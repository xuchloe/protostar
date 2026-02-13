from astropy.io import fits
import numpy as np
import emcee
from find_source import summary
import corner
import math
from astropy.coordinates import Angle
import astropy.units as units
import scipy.special as sp
import warnings
import itertools
from uncertainties import ufloat
import sigfig

def p_model(p_params, u, v):
    i0, l0, m0 = p_params
    return i0 * np.exp(-2*np.pi*1j*(u*l0 + v*m0))

def c_model(c_params, u, v):
    s0, l0, m0, vis_sigma = c_params
    return s0 * np.exp(-0.5*(u**2 + v**2)/vis_sigma**2) * np.exp(-2*np.pi*1j*(u*l0 + v*m0))

def g_model(g_params, u, v):
    s0, l0, m0, vis_sigma, ratio, vis_theta = g_params
    return s0 * np.exp(-0.5*((u*np.cos(vis_theta)-v*np.sin(vis_theta))**2 + (u*np.sin(vis_theta)+v*np.cos(vis_theta))**2/ratio**2)/vis_sigma**2) \
            * np.exp(-2*np.pi*1j*(u*l0 + v*m0))

def d_model(d_params, u, v):
    s0, l0, m0, vis_r = d_params
    return s0 * 2*vis_r/np.sqrt(u**2+v**2) * sp.j1(np.sqrt(u**2+v**2)/vis_r) * np.exp(-2*np.pi*1j*(u*l0 + v*m0))

def p_p0(peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers):
    p0 = np.zeros((n_walkers, 3))
    for i in range(n_walkers):
        p0[i,0] = np.random.uniform(0.95*peak, 1.05*peak)
        p0[i,1] = np.random.uniform(-rad_pix/2+rad_coord[0], rad_pix/2+rad_coord[0])
        p0[i,2] = np.random.uniform(-rad_pix/2+rad_coord[1], rad_pix/2+rad_coord[1])
    return p0

def c_p0(peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers):
    p0 = np.zeros((n_walkers, 4))
    source_area = rad_barea * total_flux / peak
    sigma = np.sqrt(source_area / (2*np.pi))
    vis_sigma = 1/(2*np.pi*sigma)
    for i in range(n_walkers):
        p0[i,0] = np.random.uniform(0.95*total_flux, 1.2*total_flux) # biased higher to account for missing small baselines
        p0[i,1] = np.random.uniform(-rad_pix/2+rad_coord[0], rad_pix/2+rad_coord[0])
        p0[i,2] = np.random.uniform(-rad_pix/2+rad_coord[1], rad_pix/2+rad_coord[1])
        p0[i,3] = np.random.uniform(0.95*vis_sigma, 1.05*vis_sigma)
    return p0

def g_p0(peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers):
    p0 = np.zeros((n_walkers, 6))
    source_area = rad_barea * total_flux / peak
    sigma = np.sqrt(source_area / (2*np.pi))
    vis_sigma = 1/(2*np.pi*sigma)
    for i in range(n_walkers):
        p0[i,0] = np.random.uniform(0.95*total_flux, 1.2*total_flux) # biased higher to account for missing small baselines
        p0[i,1] = np.random.uniform(-rad_pix/2+rad_coord[0], rad_pix/2+rad_coord[0])
        p0[i,2] = np.random.uniform(-rad_pix/2+rad_coord[1], rad_pix/2+rad_coord[1])
        p0[i,3] = np.random.uniform(0.95*vis_sigma, 1.05*vis_sigma)
        p0[i,4] = np.random.uniform(0, 1)
        while p0[i,4] == 0:
            p0[i,4] = np.random.uniform(0, 1)
        p0[i,5] = np.random.uniform(-np.pi/2, np.pi/2)
    return p0

def d_p0(peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers):
    p0 = np.zeros((n_walkers, 4))
    source_area = rad_barea * total_flux / peak
    r = np.sqrt(source_area / np.pi)
    vis_r = 1/(math.pi*r)
    for i in range(n_walkers):
        p0[i,0] = np.random.uniform(0.95*total_flux, 1.2*total_flux) # biased higher to account for missing small baselines
        p0[i,1] = np.random.uniform(-rad_pix/2+rad_coord[0], rad_pix/2+rad_coord[0])
        p0[i,2] = np.random.uniform(-rad_pix/2+rad_coord[1], rad_pix/2+rad_coord[1])
        p0[i,3] = np.random.uniform(0.95*vis_r, 1.05*vis_r)
    return p0

def all_p1(med_sd, resolved, point_intensity, c_flux, c_sigma, rad_position, rad_bmaj, rad_pix, n_walkers, chain):
    if point_intensity is not None:
        med_sd[0] = (point_intensity, med_sd[0][1])
    n_params = len(med_sd)
    if c_flux is not None and n_params == 6:
        med_sd[0] = (c_flux, med_sd[0][1])
    if c_sigma is not None and n_params == 6:
        med_sd[3] = (c_sigma, med_sd[3][1])
    p1 = np.zeros((n_walkers, n_params))
    for i in range(n_walkers):
        for j in range(n_params):
            if resolved:
                if j in [0, 3, 4]:  # ensure non-negative flux, width parameter, ratio
                    p1[i,j] = np.random.uniform(max(-2*med_sd[j][1]+med_sd[j][0], 0), 2*med_sd[j][1]+med_sd[j][0])
                    if j == 4:
                        if p1[i,j] == 0:
                            p1[i,j] = np.random.uniform(0.01, 0.05)  # avoid zero ratio
                        if p1[i,j] > 1:
                            p1[i,j] = np.random.uniform(0.96, 1.0)  # cap ratio at 1
                elif j == 5 and med_sd[j][1] > 10 * np.pi/180:  # vis_theta standard devation > 10 degrees
                    vis_theta_samples = [params[j] for params in chain]
                    neg_vis_thetas = [theta for theta in vis_theta_samples if theta < 0]
                    pos_vis_thetas = [theta for theta in vis_theta_samples if theta >= 0]
                    neg_med = np.median(neg_vis_thetas) if neg_vis_thetas else None
                    pos_med = np.median(pos_vis_thetas) if pos_vis_thetas else None
                    if neg_med is not None and pos_med is not None:
                        if abs(abs(neg_med) - pos_med) < 10 * np.pi/180:
                            theta_guess = -np.pi/2
                        else:
                            theta_guess = np.median(vis_theta_samples)
                        p1[i,j] = np.random.uniform(max(theta_guess - np.pi/36, -np.pi/2), theta_guess + np.pi/36)
                    else: # at least all between -90 and 0 or all between 0 and 90 degrees
                        p1[i,j] = np.random.uniform(-2*med_sd[j][1]+med_sd[j][0], 2*med_sd[j][1]+med_sd[j][0])
                else:
                    p1[i,j] = np.random.uniform(-2*med_sd[j][1]+med_sd[j][0], 2*med_sd[j][1]+med_sd[j][0])
            else: # source is unresolved, most likely a point source
                if j == 0:  # ensure non-negative flux
                    p1[i,j] = np.random.uniform(max(-2*med_sd[j][1]+med_sd[j][0], 0), 2*med_sd[j][1]+med_sd[j][0])
                if j == 1:
                    p1[i,j] = np.random.uniform(-rad_pix/2+rad_position[0], rad_pix/2+rad_position[0])
                if j == 2:
                    p1[i,j] = np.random.uniform(-rad_pix/2+rad_position[1], rad_pix/2+rad_position[1])
                if j == 3:
                    p1[i,j] = np.random.uniform(med_sd[j][0], med_sd[j][0]*2) # width parameter larger than best fit to simulate point source
                if j == 4:
                    p1[i,j] = np.random.uniform(0.75, 1.0) # bias ratio towards 1 for unresolved source
                    while p1[i,4] == 0:
                        p1[i,4] = np.random.uniform(0, 1)
                if j == 5:
                    p1[i,j] = np.random.uniform(-np.pi/2, np.pi/2)
    return p1

def p_prior(params, vis_priors, rad_bmaj, rad_bmin):
    i0, l0, m0 = params
    if vis_priors is not None:
        i0_priors = vis_priors[0]
        l0_priors = vis_priors[1]
        m0_priors = vis_priors[2]
        i0_min, i0_max = i0_priors
        l0_min, l0_max = l0_priors
        m0_min, m0_max = m0_priors

        if i0_min is not None and i0 < i0_min:
            return -np.inf
        if i0_max is not None and i0 > i0_max:
            return -np.inf
        if type(i0_priors) is tuple:
            if i0_min is not None and i0 == i0_min:
                return -np.inf
            if i0_max is not None and i0 == i0_max:
                return -np.inf
        if l0_min is not None and l0 < l0_min:
            return -np.inf
        if l0_max is not None and l0 > l0_max:
            return -np.inf
        if type(l0_priors) is tuple:
            if l0_min is not None and l0 == l0_min:
                return -np.inf
            if l0_max is not None and l0 == l0_max:
                return -np.inf
        if m0_min is not None and m0 < m0_min:
            return -np.inf
        if m0_max is not None and m0 > m0_max:
            return -np.inf
        if type(m0_priors) is tuple:
            if m0_min is not None and m0 == m0_min:
                return -np.inf
            if m0_max is not None and m0 == m0_max:
                return -np.inf
    return 0.0

def c_prior(params, vis_priors, rad_bmaj, rad_bmin):
    s0, l0, m0, vis_sigma = params
    if vis_sigma <= 0: # hardcoded prior
        return -np.inf
    if vis_priors is not None:
        i0_priors = vis_priors[0]
        l0_priors = vis_priors[1]
        m0_priors = vis_priors[2]
        vis_sigma_priors = vis_priors[3]
        i0_min, i0_max = i0_priors
        l0_min, l0_max = l0_priors
        m0_min, m0_max = m0_priors
        vis_sigma_min, vis_sigma_max = vis_sigma_priors

        # convert s0 to i0 for priors
        source_area = 1/(2*np.pi*vis_sigma**2)
        rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))
        n_beams = rad_barea / source_area
        i0 = s0 / n_beams
        if i0_min is not None and i0 < i0_min:
            return -np.inf
        if i0_max is not None and i0 > i0_max:
            return -np.inf
        if type(i0_priors) is tuple:
            if i0_min is not None and i0 == i0_min:
                return -np.inf
            if i0_max is not None and i0 == i0_max:
                return -np.inf
        if l0_min is not None and l0 < l0_min:
            return -np.inf
        if l0_max is not None and l0 > l0_max:
            return -np.inf
        if type(l0_priors) is tuple:
            if l0_min is not None and l0 == l0_min:
                return -np.inf
            if l0_max is not None and l0 == l0_max:
                return -np.inf
        if m0_min is not None and m0 < m0_min:
            return -np.inf
        if m0_max is not None and m0 > m0_max:
            return -np.inf
        if type(m0_priors) is tuple:
            if m0_min is not None and m0 == m0_min:
                return -np.inf
            if m0_max is not None and m0 == m0_max:
                return -np.inf
        if vis_sigma_min is not None and vis_sigma < vis_sigma_min:
            return -np.inf
        if vis_sigma_max is not None and vis_sigma > vis_sigma_max:
            return -np.inf
        if type(vis_sigma_priors) is tuple:
            if vis_sigma_min is not None and vis_sigma == vis_sigma_min:
                return -np.inf
            if vis_sigma_max is not None and vis_sigma == vis_sigma_max:
                return -np.inf
    return 0.0

def g_prior(params, vis_priors, rad_bmaj, rad_bmin):
    s0, l0, m0, vis_sigma, ratio, vis_theta = params
    # hardcoded priors
    if vis_sigma <= 0:
        return -np.inf
    if ratio <= 0:
        return -np.inf
    if ratio > 1:
        return -np.inf
    if vis_theta < -np.pi/2:
        return -np.inf
    if vis_theta > np.pi/2:
        return -np.inf
    if vis_priors is not None:
        i0_priors = vis_priors[0]
        l0_priors = vis_priors[1]
        m0_priors = vis_priors[2]
        vis_sigma_priors = vis_priors[3]
        ratio_priors = vis_priors[4]
        vis_theta_priors = vis_priors[5]
        i0_min, i0_max = i0_priors
        l0_min, l0_max = l0_priors
        m0_min, m0_max = m0_priors
        vis_sigma_min, vis_sigma_max = vis_sigma_priors
        ratio_min, ratio_max = ratio_priors
        vis_theta_min, vis_theta_max = vis_theta_priors

        # convert s0 to i0 for priors
        source_area = 1/(2*np.pi*vis_sigma**2*ratio)
        rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))
        n_beams = rad_barea / source_area
        i0 = s0 / n_beams
        if i0_min is not None and i0 < i0_min:
            return -np.inf
        if i0_max is not None and i0 > i0_max:
            return -np.inf
        if type(i0_priors) is tuple:
            if i0_min is not None and i0 == i0_min:
                return -np.inf
            if i0_max is not None and i0 == i0_max:
                return -np.inf
        if l0_min is not None and l0 < l0_min:
            return -np.inf
        if l0_max is not None and l0 > l0_max:
            return -np.inf
        if type(l0_priors) is tuple:
            if l0_min is not None and l0 == l0_min:
                return -np.inf
            if l0_max is not None and l0 == l0_max:
                return -np.inf
        if m0_min is not None and m0 < m0_min:
            return -np.inf
        if m0_max is not None and m0 > m0_max:
            return -np.inf
        if type(m0_priors) is tuple:
            if m0_min is not None and m0 == m0_min:
                return -np.inf
            if m0_max is not None and m0 == m0_max:
                return -np.inf
        if vis_sigma_min is not None and vis_sigma < vis_sigma_min:
            return -np.inf
        if vis_sigma_max is not None and vis_sigma > vis_sigma_max:
            return -np.inf
        if type(vis_sigma_priors) is tuple:
            if vis_sigma_min is not None and vis_sigma == vis_sigma_min:
                return -np.inf
            if vis_sigma_max is not None and vis_sigma == vis_sigma_max:
                return -np.inf
        if ratio_min is not None and ratio < ratio_min:
            return -np.inf
        if ratio_max is not None and ratio > ratio_max:
            return -np.inf
        if type(ratio_priors) is tuple:
            if ratio_min is not None and ratio == ratio_min:
                return -np.inf
            if ratio_max is not None and ratio == ratio_max:
                return -np.inf
        if vis_theta_min is not None and vis_theta < vis_theta_min:
            return -np.inf
        if vis_theta_max is not None and vis_theta > vis_theta_max:
            return -np.inf
        if type(vis_theta_priors) is tuple:
            if vis_theta_min is not None and vis_theta == vis_theta_min:
                return -np.inf
            if vis_theta_max is not None and vis_theta == vis_theta_max:
                return -np.inf
    return 0.0

def d_prior(params, vis_priors, rad_bmaj, rad_bmin):
    s0, l0, m0, vis_r = params
    if vis_r <= 0: # hardcoded prior
        return -np.inf
    if vis_priors is not None:
        i0_priors = vis_priors[0]
        l0_priors = vis_priors[1]
        m0_priors = vis_priors[2]
        vis_r_priors = vis_priors[3]
        i0_min, i0_max = i0_priors
        l0_min, l0_max = l0_priors
        m0_min, m0_max = m0_priors
        vis_r_min, vis_r_max = vis_r_priors

        # convert s0 to i0 for priors
        source_area = 1/(4*np.pi*vis_r**2)
        rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))
        n_beams = rad_barea / source_area
        i0 = s0 / n_beams
        if i0_min is not None and i0 < i0_min:
            return -np.inf
        if i0_max is not None and i0 > i0_max:
            return -np.inf
        if type(i0_priors) is tuple:
            if i0_min is not None and i0 == i0_min:
                return -np.inf
            if i0_max is not None and i0 == i0_max:
                return -np.inf
        if l0_min is not None and l0 < l0_min:
            return -np.inf
        if l0_max is not None and l0 > l0_max:
            return -np.inf
        if type(l0_priors) is tuple:
            if l0_min is not None and l0 == l0_min:
                return -np.inf
            if l0_max is not None and l0 == l0_max:
                return -np.inf
        if m0_min is not None and m0 < m0_min:
            return -np.inf
        if m0_max is not None and m0 > m0_max:
            return -np.inf
        if type(m0_priors) is tuple:
            if m0_min is not None and m0 == m0_min:
                return -np.inf
            if m0_max is not None and m0 == m0_max:
                return -np.inf
        if vis_r_min is not None and vis_r < vis_r_min:
            return -np.inf
        if vis_r_max is not None and vis_r > vis_r_max:
            return -np.inf
        if type(vis_r_priors) is tuple:
            if vis_r_min is not None and vis_r == vis_r_min:
                return -np.inf
            if vis_r_max is not None and vis_r == vis_r_max:
                return -np.inf
    return 0.0

def log_likelihood(model, re, im, u, v, w):
    return -0.5 * np.sum(w * ((re - model.real)**2 + (im - model.imag)**2))

P_PARAMS = ['i0', 'l0', 'm0']
C_PARAMS = ['s0', 'l0', 'm0', 'vis_sigma']
G_PARAMS = ['s0', 'l0', 'm0', 'vis_sigma', 'ratio', 'vis_theta']
D_PARAMS = ['s0', 'l0', 'm0', 'vis_r']

SOURCE_TYPES = {'p': [3, p_p0, p_prior, p_model, P_PARAMS], \
                'c': [4, c_p0, c_prior, c_model, C_PARAMS], \
                'g': [6, g_p0, g_prior, g_model, G_PARAMS], \
                'd': [4, d_p0, d_prior, d_model, D_PARAMS]}

def log_probability(params, sources, vis_priors, re, im, u, v, w, rad_bmaj, rad_bmin):
    log_prior = 0.0
    model = 0.0
    start = 0
    l0_list = []
    m0_list = []
    for i in range(len(sources)):
        source = sources[i]
        n_params = SOURCE_TYPES[source][0]
        prior_func = SOURCE_TYPES[source][2]
        model_func = SOURCE_TYPES[source][3]
        source_params = params[start:start+n_params]
        l0_list.append(source_params[1])
        m0_list.append(source_params[2])
        lp = prior_func(source_params, vis_priors[i], rad_bmaj, rad_bmin)
        if not np.isfinite(lp):
            return -np.inf
        log_prior += lp
        model += model_func(source_params, u, v)
        start += n_params
    indices = list(range(len(l0_list))) # list of indices for l0_list and m0_list
    pairs = itertools.permutations(indices, 2)
    for (idx1, idx2) in pairs:
        dist = np.sqrt((l0_list[idx1] - l0_list[idx2])**2 + (m0_list[idx1] - m0_list[idx2])**2)
        if dist < rad_bmaj/10:  # 1/10 of beam major axis as threshold distance to consider sources too close
            return -np.inf
    log_likelihood_value = log_likelihood(model, re, im, u, v, w)
    return log_prior + log_likelihood_value

def round_tuple(tup):
    rounded_err = sigfig.round(float(tup[1]), sigfigs=3)
    str_err = str(rounded_err)
    places = 0
    if '.' in str_err:
        decimal = str_err.split('.')[1]
        places = len(decimal)
    return (round(float(tup[0]), places), rounded_err)

def limiting_value(param_name, tuple, param_chain, n_walkers, param_priors):
    median = tuple[0]
    sd = tuple[1]
    last_n = param_chain[-n_walkers:]

    # adjusting for hardcoded priors
    if param_priors is None:
        if param_name in ['vis_sigma', 'vis_r', 'ratio']:
            param_priors = [0, None]
        if param_name == 'vis_theta':
            param_priors = [-np.pi/2, np.pi/2]
    if param_priors is not None:
        if param_priors[0] is None:
            if param_name in ['vis_sigma', 'vis_r', 'ratio']:
                param_priors = [0, param_priors[1]]
            if param_name == 'vis_theta':
                param_priors = [-np.pi/2, param_priors[1]]
        if param_priors[1] is None:
            if param_name == 'vis_theta':
                param_priors = [param_priors[0], np.pi/2]
            if param_name == 'ratio':
                param_priors = [param_priors[0], 1]
        if param_priors[0] is not None:
            if param_name in ['vis_sigma', 'vis_r', 'ratio']:
                param_priors = [max(param_priors[0], 0), param_priors[1]]
            if param_name == 'vis_theta':
                param_priors = [max(param_priors[0], -np.pi/2), np.pi/2]
        if param_priors[1] is not None:
            if param_name == 'ratio':
                param_priors = [param_priors[0], min(param_priors[1], 1)]
            if param_name == 'vis_theta':
                param_priors = [param_priors[0], min(param_priors[1], np.pi/2)]

    if all(last_n > median): # running away to higher values
        return ('upper', np.percentile(param_chain, 99.7))
    elif all(last_n < median):  # running away to lower values
        return ('lower', np.percentile(param_chain, 0.3))

    if param_priors is not None:
        if param_priors[0] is not None and all(last_n < param_priors[0] + sd/2): # hugging lower prior
            return ('lower prior', np.percentile(param_chain, 0.3))
        elif param_priors[1] is not None and all(last_n > param_priors[1] - sd/2): # hugging upper prior
            return ('upper prior', np.percentile(param_chain, 99.7))

    if param_name == 'vis_theta' and param_priors == [-np.pi/2, np.pi/2]: # special case for angle wrapping
        neg_vis_thetas = [theta for theta in param_chain if theta < 0]
        pos_vis_thetas = [theta for theta in param_chain if theta >= 0]
        neg_med = np.median(neg_vis_thetas) if neg_vis_thetas else None
        pos_med = np.median(pos_vis_thetas) if pos_vis_thetas else None
        if neg_med is not None and pos_med is not None:
            if abs(abs(neg_med) - pos_med) < 10 * np.pi/180:
                return('angle wrap', -np.pi/2)
    return None

def uv_fit(fits_file: str, sources: list, priors: list = None, clean_output=True, corner_plot=True, additional_runs: int = 2):
    # priors = [[(i0_min, i0_max), (l0_min, l0_max), (m0_min, m0_max), (width_param_min, width_param_max), (ratio_min, ratio_max), (theta_min, theta_max)], ...]
    # but (tuple) for exclusive and [list] for inclusive
    # TODO: documentation
    '''
    Fit UV data from a FITS file with specified source types using MCMC.

    Parameters
    ----------
    fits_file (str): Path to the UV FITS file.
    sources (list): List of source types to fit. Each source type should be one of
                    'p' (point), 'c' (circular gaussian), 'g' (gaussian), 'd' (disk), or 'any' (try all and pick best fit).
    '''
    # Check additional_runs
    if additional_runs < 0:
        raise ValueError("additional_runs must be a non-negative integer.")

    # Check priors format
    if priors is not None:
        if len(priors) != len(sources):
            raise ValueError("Length of priors must match length of sources.")
        for i in range(len(priors)):
            if priors[i] is not None:
                if type(priors[i]) is not list:
                    raise ValueError("Each element in priors must be None or a list corresponding to a source.")
                if len(priors[i]) != 6:
                    raise ValueError("Each prior list must have 6 elements corresponding to ranges for peak, RA, declination, width parameter, ratio, and angle.")
                for i in range(len(priors[i])):
                    if priors[i][i] is not None:
                        if type(priors[i][i]) not in [list, tuple]:
                            raise ValueError("Each prior range must be None, a list (inclusive), or a tuple (exclusive).")
                        if len(priors[i][i]) != 2:
                            raise ValueError("Each prior range list or tuple must have exactly two elements: min and max.")

    # Check input source types
    if len(sources) == 0:
        raise ValueError("No sources specified. Try specifying one or more sources of type \
                         'p' (point), 'c' (circular gaussian), 'g' (gaussian), 'd' (disk), or 'any' (try all and pick best fit).")
    for source in sources:
        if source not in SOURCE_TYPES and source != 'any':
            raise ValueError(f"Source type '{source}' is not recognized. Source type must be one of the following: \
                            'p' (point), 'c' (circular gaussian), 'g' (gaussian), 'd' (disk), or 'any' (try all and pick best fit).")

    vis_priors = []
    if priors is None:
        priors = [None] * len(sources)
    for i in range(len(priors)):
        mini_vis_priors = []
        if priors[i] is not None:
            for j in range(len(priors[i])):
                if priors[i][j] is not None:
                    if j == 0:  # peak flux density, keep as is
                        mini_vis_priors.append(priors[i][j])
                    elif j in [1, 2]:  # l0, m0
                        # convert from arcsec to radian
                        rad_min = float(Angle(priors[i][j][0], units.arcsec).to(units.radian).value)
                        rad_max = float(Angle(priors[i][j][1], units.arcsec).to(units.radian).value)
                        if type(priors[i][j]) is tuple:
                            mini_vis_priors.append((rad_min, rad_max))
                        else:
                            mini_vis_priors.append([rad_min, rad_max])
                    elif j == 3:  # width parameter
                        rad_min = float(Angle(priors[i][j][0], units.arcsec).to(units.radian).value)
                        rad_max = float(Angle(priors[i][j][1], units.arcsec).to(units.radian).value)
                        vis_min = 1/(2*np.pi*rad_max)
                        vis_max = 1/(2*np.pi*rad_min)
                        if type(priors[i][j]) is tuple:
                            mini_vis_priors.append((vis_min, vis_max))
                        else:
                            mini_vis_priors.append([vis_min, vis_max])
                    elif j == 4:  # ratio
                        mini_vis_priors.append(priors[i][j])
                    elif j == 5:  # angle
                        mini_vis_priors.append(priors[i][j] * np.pi/180) # convert from degrees to radians
                else:
                    mini_vis_priors.append(None)
            vis_priors.append(mini_vis_priors)
        else: # nothing to convert
            vis_priors.append([[None, None]] * 6)

    # Extract data from fits file
    file = fits.open(fits_file)
    cdelt1 = file[0].header['CDELT1']
    cunit1 = file[0].header['CUNIT1']
    naxis1 = file[0].header['NAXIS1']
    data = file[1].data

    summ = summary(fits_file, plot=False)
    bmaj = file[0].header['BMAJ'] # cunit1
    bmin = file[0].header['BMIN'] # cunit1
    rad_bmaj = Angle(bmaj, cunit1).to(units.radian).value
    rad_bmin = Angle(bmin, cunit1).to(units.radian).value
    rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))
    rad_pix = float(Angle(cdelt1, cunit1).to(units.radian).value)
    int_peaks = summ['int_peak_val']
    int_coords = summ['int_peak_coord']
    ext_peaks = summ['ext_peak_val']
    ext_coords = summ['ext_peak_coord']

    int_info = list(zip(int_peaks, int_coords))
    if type(ext_peaks) is list:
        ext_info = list(zip(ext_peaks, ext_coords))
    else:
        ext_info = []
    all_peaks = int_info + ext_info # list of tuples (peak_value, (l_coord, m_coord))
    all_peaks.sort(reverse=True) # sort by peak value
    n_peaks = len(all_peaks)
    if n_peaks < len(sources):
        warnings.warn(f"Number of detected peaks ({n_peaks}) is less than number of sources to fit ({len(sources)}).")

    vis = np.array(data)
    freq_bin, u, v, re, im, w = [], [], [], [], [], []
    for row in vis:
        freq_bin_data, u_data, v_data, re_data, im_data, w_data = row
        freq_bin.append(int(freq_bin_data))
        u.append(int(u_data))
        v.append(int(v_data))
        re.append(float(re_data/w_data))
        im.append(float(im_data/w_data))
        w.append(float(w_data))

    # Adding in conjugate half of data
    freq_bin *= 2
    neg_u = [-1 * val for val in u]
    u += neg_u
    neg_v = [-1 * val for val in v]
    v += neg_v
    re *= 2
    neg_im = [-1 * val for val in im]
    im += neg_im
    w *= 2

    freq_bin = np.array(freq_bin)
    u = np.array(u)
    v = np.array(v)
    re = np.array(re)
    im = np.array(im)
    w = np.array(w)

    file.close() # good practice

    # Estimate total flux from small baselines
    small_baselines = []
    q = np.sqrt(u**2 + v**2)
    baseline_indices = np.argsort(q, axis=None)
    small_baselines_indices = baseline_indices[:len(baseline_indices)//20]  # smallest 5% of baselines
    for i in small_baselines_indices:
        small_baselines.append(np.sqrt(im[i]**2 + re[i]**2))
    total_flux_median = np.median(small_baselines)
    total_flux_mean = np.mean(small_baselines)
    total_flux_sd = np.std(small_baselines)
    if abs(total_flux_median - total_flux_mean) > total_flux_sd:
        total_flux = total_flux_median
    else:
        total_flux = total_flux_mean

    # All possible permutations
    n_sources = len(sources)
    sample_space = list(SOURCE_TYPES.keys()) * n_sources
    all_permutations = list(itertools.permutations(sample_space, n_sources))

    for i in range(n_sources):
        if sources[i] != 'any':
            all_permutations = [p for p in all_permutations if p[i] == sources[i]] # remove unwanted permutations
    all_permutations = list(set(all_permutations)) # remove duplicates

    all_results = []
    for permutation in all_permutations:
        # Calculate n_params and n_walkers
        n_params = 0
        for i in range(n_sources):
            source = permutation[i]
            n_params += SOURCE_TYPES[source][0]
        n_walkers = 2 * n_params

        # Initial guesses
        for i in range(n_sources):
            source = permutation[i]
            peak = all_peaks[i][0] if i < n_peaks else all_peaks[-1][0]
            coord0 = all_peaks[i][1] if i < n_peaks else all_peaks[-1][1]
            rad_coord = (float(Angle(coord0[0], units.arcsec).to(units.radian).value), float(Angle(coord0[1], units.arcsec).to(units.radian).value))
            if i == 0:
                p0 = SOURCE_TYPES[source][1](peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers)
            else:
                mini_p0 = SOURCE_TYPES[source][1](peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers)
                if i >= n_peaks: # edit l0, m0 initial guesses
                    for j in range(n_walkers):
                        mini_p0[j,1] = np.random.uniform(-naxis1/2*rad_pix, naxis1/2*rad_pix)
                        mini_p0[j,2] = np.random.uniform(-naxis1/2*rad_pix, naxis1/2*rad_pix)
                p0 = np.append(p0, mini_p0, axis=1)

        # Set up and run MCMC
        n_steps = 100
        sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=(permutation, vis_priors, re, im, u, v, w, rad_bmaj, rad_bmin))
        try:
            state = sampler.run_mcmc(p0, n_steps)
        except emcee.autocorr.AutocorrError:
            pass
        tau = sampler.get_autocorr_time(quiet=True)
        if np.isnan(tau).all():
            warnings.warn(f"Autocorrelation time for first run of {permutation} could not be estimated; all values are NaN.", RuntimeWarning)
            all_results.append({'permutation': permutation, 'n_params': n_params, 'result': None, 'chi2': np.inf, 'chain': None})
            continue
        int_tau = math.ceil(np.nanmax(tau))
        steps_to_50_tau = abs(int_tau * 50 - n_steps)
        sampler.run_mcmc(state, steps_to_50_tau)
        chain = sampler.get_chain(discard = int_tau * 10, flat=True)
        log_probs = sampler.get_log_prob(discard = int_tau * 10, flat=True)
        max_prob_index = np.argmax(log_probs)

        # Find parameter estimates and uncertainties and calculate chi2
        result = {}
        model = 0.0
        start = 0
        for i in range(n_sources):
            source = permutation[i]
            n_source_params = SOURCE_TYPES[source][0]
            source_chain = chain[:, start:start+n_source_params]
            source_result = {'type': source}
            temp_medians = [] # to store medians
            temp_bests = {} # to store best values (that maximimize probability)
            temp_max_probs = [] # to store max prob values
            for j in range(n_source_params):
                samples = source_chain[:, j]
                temp_max_probs.append(samples[max_prob_index])
                samples_med = np.median(samples)
                samples_sd = np.nanstd(samples)
                param_name = SOURCE_TYPES[source][4][j]
                source_result[param_name] = (float(samples_med), float(samples_sd))
                temp_bests[param_name] = float(samples[max_prob_index])
                temp_medians.append(samples_med)
            source_result['best'] = temp_bests
            model += SOURCE_TYPES[source][3](temp_max_probs, u, v)
            result[f'source_{i+1}'] = source_result
            start += n_source_params
        chi2 = float(np.sum(w * ((re - model.real)**2 + (im - model.imag)**2)))

        all_results.append({'permutation': permutation, 'n_params': n_params, 'result': result, 'chi2': chi2, 'chain': chain})

    # Rank permutations by chi2
    all_results.sort(key=lambda x: x['chi2']) # lowest to highest chi2
    best_perm = all_results[0]['permutation']
    best_result = all_results[0]['result']

    # Do it again with refined initial guesses, if requested
    for reps in range(additional_runs):  # two additional refinement iterations
        second_results = []
        for permutation_info in all_results:
            # Calculate n_params and n_walkers
            permutation = permutation_info['permutation']
            chain = permutation_info['chain']
            n_params = 0
            for i in range(n_sources):
                source = permutation[i]
                n_params += SOURCE_TYPES[source][0]
            n_walkers = 2 * n_params

            # New initial guesses from previous results
            for i in range(n_sources):
                source = permutation[i]
                coord0 = all_peaks[i][1] if i < n_peaks else all_peaks[-1][1]
                rad_coord = (float(Angle(coord0[0], units.arcsec).to(units.radian).value), float(Angle(coord0[1], units.arcsec).to(units.radian).value))
                if permutation_info['result'] is None: # fitting didn't happen, so can't actually refine
                    peak = all_peaks[i][0] if i < n_peaks else all_peaks[-1][0]
                    if i == 0:
                        p1 = SOURCE_TYPES[source][1](peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers)
                    else:
                        mini_p1 = SOURCE_TYPES[source][1](peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers)
                        if i >= n_peaks: # edit l0, m0 initial guesses
                            for j in range(n_walkers):
                                mini_p1[j,1] = np.random.uniform(-naxis1/2*rad_pix, naxis1/2*rad_pix)
                                mini_p1[j,2] = np.random.uniform(-naxis1/2*rad_pix, naxis1/2*rad_pix)
                        p1 = np.append(p1, mini_p1, axis=1)
                else:
                    rad_position = best_result[f'source_{i+1}']['best']['l0'], best_result[f'source_{i+1}']['best']['m0']
                    source_result = permutation_info['result'][f'source_{i+1}']
                    best_params = []
                    med_sd = []
                    point_intensity = None
                    for param_name in SOURCE_TYPES[source][4]:
                        med_sd.append(source_result[param_name])
                        best_params.append(source_result[param_name][0])

                    # use p fitting to help c/g/d fitting if p chi2 was better than c/g/d chi2 in first run
                    resolved = True
                    if source != 'p':
                        if 1/(2*np.pi*best_params[3]) < rad_bmaj/2:  # conditions for unresolved source
                            resolved = False
                    if not resolved:
                        temp_perm = best_perm[:i] + ('p',) + best_perm[i+1:]
                        if temp_perm == best_perm:
                            point_intensity = best_result[f'source_{i+1}']['best']['i0']
                        else:
                            point_intensity = uv_fit(fits_file, list(temp_perm), priors=priors, clean_output=True, corner_plot=False, additional_runs=0)[0]['result'][f'source_{i+1}']['i0'][0]

                    # use c fitting to help g fitting if c chi2 was better than g chi2 in first run
                    c_flux = None
                    c_sigma = None
                    if source == 'g' and best_perm[i] == 'c':
                        c_flux = best_result[f'source_{i+1}']['best']['s0']
                        c_sigma = best_result[f'source_{i+1}']['best']['vis_sigma']
                    if i == 0:
                        p1 = all_p1(med_sd, resolved, point_intensity, c_flux, c_sigma, rad_position, rad_bmaj, rad_pix, n_walkers, chain)
                    else:
                        p1 = np.append(p1, all_p1(med_sd, resolved, point_intensity, c_flux, c_sigma, rad_position, rad_bmaj, rad_pix, n_walkers, chain), axis=1)

                    # edit vis_priors if unresolved source
                    if not resolved:
                        if vis_priors[i] is None:
                            vis_priors[i] = [(None, None)] * 6
                        vis_priors[i][1] = (-rad_pix+rad_coord[0], rad_pix+rad_coord[0]) # l0 within one pixel of image domain result
                        vis_priors[i][2] = (-rad_pix+rad_coord[1], rad_pix+rad_coord[1]) # m0 within one pixel of image domain result

            # Set up and run MCMC
            n_steps = 100
            sampler1 = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=(permutation, vis_priors, re, im, u, v, w, rad_bmaj, rad_bmin))
            try:
                state = sampler1.run_mcmc(p1, n_steps)
            except emcee.autocorr.AutocorrError:
                pass
            tau = sampler1.get_autocorr_time(quiet=True)
            if np.isnan(tau).all():
                warnings.warn(f"Autocorrelation time for second run of {permutation} could not be estimated; all values are NaN.", RuntimeWarning)
                second_results.append({'permutation': permutation, 'n_params': n_params, 'result': None, 'chi2': np.inf, 'chain': None})
                continue
            int_tau = math.ceil(np.nanmax(tau))
            steps_to_50_tau = abs(int_tau * 50 - n_steps)
            sampler1.run_mcmc(state, steps_to_50_tau)
            chain1 = sampler1.get_chain(discard = int_tau * 10, flat=True)
            log_probs1 = sampler1.get_log_prob(discard = int_tau * 10, flat=True)
            max_prob_index1 = np.argmax(log_probs1)

            # Find parameter estimates and uncertainties and calculate chi2
            result = {}
            model = 0.0
            start = 0
            for i in range(n_sources):
                source = permutation[i]
                n_source_params = SOURCE_TYPES[source][0]
                source_chain = chain1[:, start:start+n_source_params]
                source_result = {'type': source}
                temp_medians = [] # to store medians for chi2 calculation
                temp_bests = {} # to store best values (that maximimize probability)
                temp_max_probs = [] # to store max prob values
                for j in range(n_source_params):
                    samples = source_chain[:, j]
                    temp_max_probs.append(samples[max_prob_index1])
                    samples_med = np.median(samples)
                    samples_sd = np.nanstd(samples)
                    param_name = SOURCE_TYPES[source][4][j]
                    source_result[param_name] = (float(samples_med), float(samples_sd))
                    temp_bests[param_name] = float(samples[max_prob_index1])
                    temp_medians.append(samples_med)
                source_result['best'] = temp_bests
                model += SOURCE_TYPES[source][3](temp_max_probs, u, v)
                result[f'source_{i+1}'] = source_result
                start += n_source_params
            chi2 = float(np.sum(w * ((re - model.real)**2 + (im - model.imag)**2)))
            second_results.append({'permutation': permutation, 'n_params': n_params, 'result': result, 'chi2': chi2, 'chain': chain1})
        all_results = []
        for permutation_info in second_results:
            all_results.append(permutation_info)
        # Rank permutations by chi2
        all_results.sort(key=lambda x: x['chi2']) # lowest to highest chi2
        best_perm = all_results[0]['permutation']
        best_result = all_results[0]['result']

    # Bayesian Information Criterion
    n = len(re)
    for permutation_info in all_results:
        k = permutation_info['n_params']
        chi2 = permutation_info['chi2']
        if chi2 is np.inf:
            permutation_info['bic'] = np.inf
            continue
        bic = k * np.log(n) + chi2
        permutation_info['bic'] = float(bic)
    all_results.sort(key=lambda x: x['bic']) # lowest to highest BIC

    if clean_output:
        for permutation_info in all_results:
            result = permutation_info['result']
            start = 0
            permutation_chain = permutation_info['chain']
            if result is None:
                continue
            for i in range(n_sources):
                source_key = f'source_{i+1}'
                source_result = result[source_key]
                source_type = source_result['type']
                source_params = SOURCE_TYPES[source_type][4]
                n_source_params = SOURCE_TYPES[source_type][0]
                n_walkers = 2 * n_source_params
                source_chain = permutation_chain[:, start:start+n_source_params]
                source_priors = vis_priors[i]

                # convert l0, m0 to ra, dec in arcsec
                ra_chain = source_chain[:, 1]
                dec_chain = source_chain[:, 2]
                ra_prior = source_priors[1]
                dec_prior = source_priors[2]
                ra_limit = limiting_value('l0', source_result['l0'], ra_chain, n_walkers, ra_prior)
                dec_limit = limiting_value('m0', source_result['m0'], dec_chain, n_walkers, dec_prior)
                source_result['ra'] = round_tuple(tuple([float(Angle(l, units.radian).to(units.arcsec).value) for l in source_result['l0']]))
                if ra_limit is not None:
                    source_result['ra'] = (source_result['ra'][0], source_result['ra'][1], \
                                           (ra_limit[0], sigfig.round(float(Angle(ra_limit[1], units.radian).to(units.arcsec).value), sigfigs=3)))
                source_result['dec'] = round_tuple(tuple([float(Angle(m, units.radian).to(units.arcsec).value) for m in source_result['m0']]))
                if dec_limit is not None:
                    source_result['dec'] = (source_result['dec'][0], source_result['dec'][1], \
                                            (dec_limit[0], sigfig.round(float(Angle(dec_limit[1], units.radian).to(units.arcsec).value), sigfigs=3)))
                del source_result['l0']
                del source_result['m0']

                if source_type != 'p': # convert flux to peak and convert visibility width to image width in arcsec
                    flux_chain = source_chain[:, 0]
                    width_chain = source_chain[:, 3]
                    flux_prior = source_priors[0]
                    width_prior = source_priors[3]
                    flux_limit = limiting_value('s0', source_result['s0'], flux_chain, n_walkers, flux_prior)
                    width_limit = limiting_value(source_params[3], source_result[source_params[3]], width_chain, n_walkers, width_prior)
                    uwidth_scale = ufloat(source_result[source_params[3]][0], source_result[source_params[3]][1]) # vis_sigma or vis_r
                    uimg_width_scale = 1/(2*np.pi*uwidth_scale)
                    if not (1/(2*np.pi*source_result[source_params[3]][0]) < rad_bmaj/2): # resolved source
                        uflux = ufloat(source_result['s0'][0], source_result['s0'][1])
                        if source_type == 'c':
                            usource_area = 2 * np.pi * uimg_width_scale**2
                        elif source_type == 'g':
                            uratio = ufloat(source_result['ratio'][0], source_result['ratio'][1])
                            usource_area = 2 * np.pi * uimg_width_scale**2 / uratio
                        elif source_type == 'd':
                            usource_area = np.pi * uimg_width_scale**2
                        n_beams = usource_area / rad_barea
                        peak = uflux / n_beams
                        # Modify dictionary
                        del source_result['s0'] # remove flux values
                        source_result['i0'] = round_tuple((peak.n, peak.s))
                        if flux_limit is not None:
                            source_result['i0'] = (source_result['i0'][0], source_result['i0'][1], \
                                                   (flux_limit[0], sigfig.round(float(flux_limit[1] / n_beams.n), sigfigs=3)))
                    else: # unresolved source
                        source_result['i0'] = round_tuple((source_result['s0'][0], source_result['s0'][1]))
                        if flux_limit is not None:
                            source_result['i0'] = (source_result['i0'][0], source_result['i0'][1], \
                                                   (flux_limit[0], sigfig.round(flux_limit[1], sigfigs=3)))
                        del source_result['s0']
                    del source_result[source_params[3]] # remove visibility width values
                    new_width_key = source_params[3].replace('vis_', '')
                    source_result[new_width_key] = round_tuple((float(Angle(uimg_width_scale.n, units.radian).to(units.arcsec).value), \
                                                    float(Angle(uimg_width_scale.s, units.radian).to(units.arcsec).value)))
                    if width_limit is not None:
                        comment = width_limit[0]
                        if 'upper' in comment:
                            comment = comment.replace('upper', 'lower') # because of inverse relationship between visibility width and image width
                        elif 'lower' in comment:
                            comment = comment.replace('lower', 'upper') # because of inverse relationship between visibility width and image width
                        source_result[new_width_key] = (source_result[new_width_key][0], source_result[new_width_key][1], \
                                                       (comment, sigfig.round(float(Angle(1/(2*np.pi*width_limit[1]), units.radian).to(units.arcsec).value), sigfigs=3)))
                else: # point source, just round
                    source_result['i0'] = round_tuple(source_result['i0'])

                if source_type == 'g': # convert visibility theta to image theta in degrees and convert sigma and ratio into major and minor
                    theta_chain = source_chain[:, 5]
                    theta_prior = source_priors[5]
                    theta_limit = limiting_value('vis_theta', source_result['vis_theta'], theta_chain, n_walkers, theta_prior)
                    uvis_theta = ufloat(source_result['vis_theta'][0], source_result['vis_theta'][1])
                    uimg_theta = (uvis_theta * (180/np.pi) - 90)
                    del source_result['vis_theta']
                    source_result['theta'] = round_tuple((uimg_theta.n % 90, uimg_theta.s))
                    if theta_limit is not None:
                        source_result['theta'] = (source_result['theta'][0], source_result['theta'][1], \
                                                 (theta_limit[0], sigfig.round(float(theta_limit[1] * 180/np.pi), sigfigs=3)))

                    ratio_chain = source_chain[:, 4]
                    ratio_prior = source_priors[4]
                    ratio_limit = limiting_value('ratio', source_result['ratio'], ratio_chain, n_walkers, ratio_prior)
                    usigma_min = ufloat(source_result['sigma'][0], source_result['sigma'][1])
                    uratio = ufloat(source_result['ratio'][0], source_result['ratio'][1])
                    usigma_maj = usigma_min / uratio
                    del source_result['sigma']
                    del source_result['ratio']
                    source_result['sigma_maj'] = round_tuple((usigma_maj.n, usigma_maj.s))
                    source_result['sigma_min'] = round_tuple((usigma_min.n, usigma_min.s))
                    if ratio_limit is not None:
                        comment = ratio_limit[0]
                        if 'upper' in comment:
                            comment = comment.replace('upper', 'lower') # because of sigma_maj = sigma_min / ratio
                        elif 'lower' in comment:
                            comment = comment.replace('lower', 'upper') # because of sigma_maj = sigma_min / ratio
                        source_result['sigma_maj'] = (source_result['sigma_maj'][0], source_result['sigma_maj'][1], \
                                                     (comment, sigfig.round((source_result['sigma_maj'][0] * uratio / ratio_limit[1]).n, sigfigs=3)))

                del source_result['best']

                start += n_source_params

    if corner_plot:
        for j in range(len(all_results)):
            permutation_info = all_results[j]
            result = permutation_info['result']
            if result is None:
                continue
            chain = permutation_info['chain']
            start = 0
            end = 0
            for i in range(n_sources):
                source_key = f'source_{i+1}'
                source_result = result[source_key]
                source_type = source_result['type']
                n_params = SOURCE_TYPES[source_type][0]
                source_params = SOURCE_TYPES[source_type][4]
                end += n_params
                if i == n_sources-1:
                    fig = corner.corner(chain[:, start:], labels=source_params)
                else:
                    fig = corner.corner(chain[:, start:end], labels=source_params)
                fig.suptitle(f'Permutation {j+1}: {permutation_info["permutation"]}, source {i+1} of {n_sources}')
                start = end

    for permutation_info in all_results:
        del permutation_info['chain']

    return all_results
