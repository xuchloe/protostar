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

def p_model(p_params, u, v, rad_bmaj, rad_barea):
    peak, ra, dec = p_params
    return peak * np.exp(-2*np.pi*1j*(u*ra + v*dec))

def c_model(c_params, u, v, rad_bmaj, rad_barea):
    peak, ra, dec, sigma = c_params
    if sigma <= rad_bmaj / 2: # unresolved
        return peak * np.exp(-2*np.pi**2 * sigma**2 * (u**2 + v**2)) * np.exp(-2*np.pi*1j*(u*ra + v*dec))
    return peak * 2*np.pi*sigma**2 / rad_barea * np.exp(-2*np.pi**2 * sigma**2 * (u**2 + v**2)) * np.exp(-2*np.pi*1j*(u*ra + v*dec))

def g_model(g_params, u, v, rad_bmaj, rad_barea):
    peak, ra, dec, sigma, ratio, vis_theta = g_params
    if sigma <= rad_bmaj / 2: # unresolved
        return peak * np.exp(-2*np.pi**2 * sigma**2 * ((u*np.cos(vis_theta)-v*np.sin(vis_theta))**2 + \
            (u*np.sin(vis_theta)+v*np.cos(vis_theta))**2/ratio**2)) * np.exp(-2*np.pi*1j*(u*ra + v*dec))
    return peak * 2*np.pi*sigma**2 / rad_barea * np.exp(-2*np.pi**2 * sigma**2 * ((u*np.cos(vis_theta)-v*np.sin(vis_theta))**2 + \
            (u*np.sin(vis_theta)+v*np.cos(vis_theta))**2/ratio**2)) \
            * np.exp(-2*np.pi*1j*(u*ra + v*dec))

def d_model(d_params, u, v, rad_bmaj, rad_barea):
    peak, ra, dec, r = d_params
    if r <= rad_bmaj / 2: # unresolved
        return peak / (np.pi * r* np.sqrt(u**2+v**2)) * sp.j1(2*np.pi* r * np.sqrt(u**2+v**2)) * np.exp(-2*np.pi*1j*(u*ra + v*dec))
    return peak * (np.pi * r**2) / (rad_barea * np.pi * r * np.sqrt(u**2+v**2)) * sp.j1(2*np.pi* r * np.sqrt(u**2+v**2)) * np.exp(-2*np.pi*1j*(u*ra + v*dec))

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
    for i in range(n_walkers):
        p0[i,0] = np.random.uniform(0.95*peak, 1.05*peak) # biased higher to account for missing small baselines
        p0[i,1] = np.random.uniform(-rad_pix/2+rad_coord[0], rad_pix/2+rad_coord[0])
        p0[i,2] = np.random.uniform(-rad_pix/2+rad_coord[1], rad_pix/2+rad_coord[1])
        p0[i,3] = np.random.uniform(0.95*sigma, 1.05*sigma)
    return p0

def g_p0(peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers):
    p0 = np.zeros((n_walkers, 6))
    source_area = rad_barea * total_flux / peak
    sigma = np.sqrt(source_area / (2*np.pi))
    for i in range(n_walkers):
        p0[i,0] = np.random.uniform(0.95*peak, 1.05*peak) # biased higher to account for missing small baselines
        p0[i,1] = np.random.uniform(-rad_pix/2+rad_coord[0], rad_pix/2+rad_coord[0])
        p0[i,2] = np.random.uniform(-rad_pix/2+rad_coord[1], rad_pix/2+rad_coord[1])
        p0[i,3] = np.random.uniform(0.95*sigma, 1.05*sigma)
        p0[i,4] = np.random.uniform(0, 1)
        while p0[i,4] == 0:
            p0[i,4] = np.random.uniform(0, 1)
        p0[i,5] = np.random.uniform(-np.pi/2, np.pi/2)
    return p0

def d_p0(peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers):
    p0 = np.zeros((n_walkers, 4))
    source_area = rad_barea * total_flux / peak
    r = np.sqrt(source_area / np.pi)
    for i in range(n_walkers):
        p0[i,0] = np.random.uniform(0.95*peak, 1.05*peak) # biased higher to account for missing small baselines
        p0[i,1] = np.random.uniform(-rad_pix/2+rad_coord[0], rad_pix/2+rad_coord[0])
        p0[i,2] = np.random.uniform(-rad_pix/2+rad_coord[1], rad_pix/2+rad_coord[1])
        p0[i,3] = np.random.uniform(0.95*r, 1.05*r)
    return p0

def all_p1(med_sd, resolved, point_intensity, c_peak, c_sigma, rad_position, rad_bmaj, rad_pix, n_walkers, chain):
    if point_intensity is not None:
        new = (point_intensity, med_sd[0][1])
        temp = [pair for pair in med_sd]
        med_sd = tuple([new] + temp[1:])
    n_params = len(med_sd)
    if c_peak is not None and n_params == 6:
        new = (c_peak, med_sd[0][1])
        temp = [pair for pair in med_sd]
        med_sd = tuple([new] + temp[1:])
    if c_sigma is not None and n_params == 6:
        new = (c_sigma, med_sd[3][1])
        temp = [pair for pair in med_sd]
        temp1 = temp[:3]
        temp2 = temp[4:]
        med_sd = tuple(temp1 + [new] + temp2)
    p1 = np.zeros((n_walkers, n_params))
    for i in range(n_walkers):
        for j in range(n_params):
            if resolved:
                if j in [3, 4]:  # ensure non-negative width parameter, ratio
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
                if j == 0:
                    p1[i,j] = np.random.uniform(-2*med_sd[j][1]+med_sd[j][0], 2*med_sd[j][1]+med_sd[j][0])
                if j == 1:
                    p1[i,j] = np.random.uniform(-rad_pix/2+rad_position[0], rad_pix/2+rad_position[0])
                if j == 2:
                    p1[i,j] = np.random.uniform(-rad_pix/2+rad_position[1], rad_pix/2+rad_position[1])
                if j == 3:
                    p1[i,j] = np.random.uniform(med_sd[j][0]/2, med_sd[j][0]) # width parameter smaller than best fit to simulate point source
                if j == 4:
                    p1[i,j] = np.random.uniform(0.75, 1.0) # bias ratio towards 1 for unresolved source
                    while p1[i,4] == 0:
                        p1[i,4] = np.random.uniform(0, 1)
                if j == 5:
                    p1[i,j] = np.random.uniform(-np.pi/2, np.pi/2)
    return p1

def p_prior(params, vis_priors, rad_bmaj, rad_bmin):
    peak, ra, dec = params
    if vis_priors is not None:
        peak_priors = vis_priors[0]
        ra_priors = vis_priors[1]
        dec_priors = vis_priors[2]
        peak_min, peak_max = peak_priors
        ra_min, ra_max = ra_priors
        dec_min, dec_max = dec_priors

        if peak_min is not None and peak < peak_min:
            return -np.inf
        if peak_max is not None and peak > peak_max:
            return -np.inf
        if type(peak_priors) is tuple:
            if peak_min is not None and peak == peak_min:
                return -np.inf
            if peak_max is not None and peak == peak_max:
                return -np.inf
        if ra_min is not None and ra < ra_min:
            return -np.inf
        if ra_max is not None and ra > ra_max:
            return -np.inf
        if type(ra_priors) is tuple:
            if ra_min is not None and ra == ra_min:
                return -np.inf
            if ra_max is not None and ra == ra_max:
                return -np.inf
        if dec_min is not None and dec < dec_min:
            return -np.inf
        if dec_max is not None and dec > dec_max:
            return -np.inf
        if type(dec_priors) is tuple:
            if dec_min is not None and dec == dec_min:
                return -np.inf
            if dec_max is not None and dec == dec_max:
                return -np.inf
    return 0.0

def c_prior(params, vis_priors, rad_bmaj, rad_bmin):
    peak, ra, dec, sigma = params
    if sigma <= 0: # hardcoded prior
        return -np.inf
    if vis_priors is not None:
        peak_priors = vis_priors[0]
        ra_priors = vis_priors[1]
        dec_priors = vis_priors[2]
        sigma_priors = vis_priors[3]
        peak_min, peak_max = peak_priors
        ra_min, ra_max = ra_priors
        dec_min, dec_max = dec_priors
        sigma_min, sigma_max = sigma_priors

        if peak_min is not None and peak < peak_min:
            return -np.inf
        if peak_max is not None and peak > peak_max:
            return -np.inf
        if type(peak_priors) is tuple:
            if peak_min is not None and peak == peak_min:
                return -np.inf
            if peak_max is not None and peak == peak_max:
                return -np.inf
        if ra_min is not None and ra < ra_min:
            return -np.inf
        if ra_max is not None and ra > ra_max:
            return -np.inf
        if type(ra_priors) is tuple:
            if ra_min is not None and ra == ra_min:
                return -np.inf
            if ra_max is not None and ra == ra_max:
                return -np.inf
        if dec_min is not None and dec < dec_min:
            return -np.inf
        if dec_max is not None and dec > dec_max:
            return -np.inf
        if type(dec_priors) is tuple:
            if dec_min is not None and dec == dec_min:
                return -np.inf
            if dec_max is not None and dec == dec_max:
                return -np.inf
        if sigma_min is not None and sigma < sigma_min:
            return -np.inf
        if sigma_max is not None and sigma > sigma_max:
            return -np.inf
        if type(sigma_priors) is tuple:
            if sigma_min is not None and sigma == sigma_min:
                return -np.inf
            if sigma_max is not None and sigma == sigma_max:
                return -np.inf
    return 0.0

def g_prior(params, vis_priors, rad_bmaj, rad_bmin):
    peak, ra, dec, sigma, ratio, vis_theta = params
    # hardcoded priors
    if sigma <= 0:
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
        peak_priors = vis_priors[0]
        ra_priors = vis_priors[1]
        dec_priors = vis_priors[2]
        sigma_priors = vis_priors[3]
        ratio_priors = vis_priors[4]
        vis_theta_priors = vis_priors[5]
        peak_min, peak_max = peak_priors
        ra_min, ra_max = ra_priors
        dec_min, dec_max = dec_priors
        sigma_min, sigma_max = sigma_priors
        ratio_min, ratio_max = ratio_priors
        vis_theta_min, vis_theta_max = vis_theta_priors

        if peak_min is not None and peak < peak_min:
            return -np.inf
        if peak_max is not None and peak > peak_max:
            return -np.inf
        if type(peak_priors) is tuple:
            if peak_min is not None and peak == peak_min:
                return -np.inf
            if peak_max is not None and peak == peak_max:
                return -np.inf
        if ra_min is not None and ra < ra_min:
            return -np.inf
        if ra_max is not None and ra > ra_max:
            return -np.inf
        if type(ra_priors) is tuple:
            if ra_min is not None and ra == ra_min:
                return -np.inf
            if ra_max is not None and ra == ra_max:
                return -np.inf
        if dec_min is not None and dec < dec_min:
            return -np.inf
        if dec_max is not None and dec > dec_max:
            return -np.inf
        if type(dec_priors) is tuple:
            if dec_min is not None and dec == dec_min:
                return -np.inf
            if dec_max is not None and dec == dec_max:
                return -np.inf
        if sigma_min is not None and sigma < sigma_min:
            return -np.inf
        if sigma_max is not None and sigma > sigma_max:
            return -np.inf
        if type(sigma_priors) is tuple:
            if sigma_min is not None and sigma == sigma_min:
                return -np.inf
            if sigma_max is not None and sigma == sigma_max:
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
    peak, ra, dec, r = params
    if r <= 0: # hardcoded prior
        return -np.inf
    if vis_priors is not None:
        peak_priors = vis_priors[0]
        ra_priors = vis_priors[1]
        dec_priors = vis_priors[2]
        r_priors = vis_priors[3]
        peak_min, peak_max = peak_priors
        ra_min, ra_max = ra_priors
        dec_min, dec_max = dec_priors
        r_min, r_max = r_priors

        if peak_min is not None and peak < peak_min:
            return -np.inf
        if peak_max is not None and peak > peak_max:
            return -np.inf
        if type(peak_priors) is tuple:
            if peak_min is not None and peak == peak_min:
                return -np.inf
            if peak_max is not None and peak == peak_max:
                return -np.inf
        if ra_min is not None and ra < ra_min:
            return -np.inf
        if ra_max is not None and ra > ra_max:
            return -np.inf
        if type(ra_priors) is tuple:
            if ra_min is not None and ra == ra_min:
                return -np.inf
            if ra_max is not None and ra == ra_max:
                return -np.inf
        if dec_min is not None and dec < dec_min:
            return -np.inf
        if dec_max is not None and dec > dec_max:
            return -np.inf
        if type(dec_priors) is tuple:
            if dec_min is not None and dec == dec_min:
                return -np.inf
            if dec_max is not None and dec == dec_max:
                return -np.inf
        if r_min is not None and r < r_min:
            return -np.inf
        if r_max is not None and r > r_max:
            return -np.inf
        if type(r_priors) is tuple:
            if r_min is not None and r == r_min:
                return -np.inf
            if r_max is not None and r == r_max:
                return -np.inf
    return 0.0

def log_likelihood(model, re, im, u, v, w):
    return -0.5 * np.sum(w * ((re - model.real)**2 + (im - model.imag)**2))

P_PARAMS = ['peak', 'ra', 'dec']
C_PARAMS = ['peak', 'ra', 'dec', 'sigma']
G_PARAMS = ['peak', 'ra', 'dec', 'sigma', 'ratio', 'vis_theta']
D_PARAMS = ['peak', 'ra', 'dec', 'r']

SOURCE_TYPES = {'p': [3, p_p0, p_prior, p_model, P_PARAMS], \
                'c': [4, c_p0, c_prior, c_model, C_PARAMS], \
                'g': [6, g_p0, g_prior, g_model, G_PARAMS], \
                'd': [4, d_p0, d_prior, d_model, D_PARAMS]}

def log_probability(params, sources, vis_priors, re, im, u, v, w, rad_bmaj, rad_bmin):
    log_prior = 0.0
    model = 0.0
    start = 0
    ra_list = []
    dec_list = []
    for i in range(len(sources)):
        source = sources[i]
        n_params = SOURCE_TYPES[source][0]
        prior_func = SOURCE_TYPES[source][2]
        model_func = SOURCE_TYPES[source][3]
        source_params = params[start:start+n_params]
        ra_list.append(source_params[1])
        dec_list.append(source_params[2])
        lp = prior_func(source_params, vis_priors[i], rad_bmaj, rad_bmin)
        rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))
        if not np.isfinite(lp):
            return -np.inf
        log_prior += lp
        model += model_func(source_params, u, v, rad_bmaj, rad_barea)
        start += n_params
    indices = list(range(len(ra_list))) # list of indices for ra_list and dec_list
    pairs = itertools.permutations(indices, 2)
    for (idx1, idx2) in pairs:
        dist = np.sqrt((ra_list[idx1] - ra_list[idx2])**2 + (dec_list[idx1] - dec_list[idx2])**2)
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

def sigmas(param_chain):
    return (np.percentile(param_chain,2.5), np.percentile(param_chain,16), np.percentile(param_chain,50),\
            np.percentile(param_chain,84), np.percentile(param_chain, 97.5))

def auto_detect(fits_file: str, n_sources: int = None, priors: list = None, clean_output=True, corner_plot=True):
    # Assume everything is a point source

    # Check priors format
    if priors is not None:
        if n_sources is not None:
            if len(priors) != n_sources and len(priors) != 1:
                raise ValueError("Length of priors list must match n_sources, or priors must be of length 1 (one set of priors for all sources).")
        if n_sources is None:
            if len(priors) != 1:
                raise ValueError("If n_sources is not provided, priors must be None or of length 1 (one set of priors for the detected sources).")
        for i in range(len(priors)):
            if priors[i] is not None:
                if type(priors[i]) is not list:
                    raise ValueError("Each element in priors must be None or a list corresponding to a source.")
                if len(priors[i]) != 3:
                    raise ValueError("Each prior list must have 3 elements corresponding to ranges for peak, RA, and declination.")
                for j in range(len(priors[i])):
                    if priors[i][j] is not None:
                        if type(priors[i][j]) not in [list, tuple]:
                            raise ValueError("Each prior range must be None, a list (inclusive), or a tuple (exclusive).")
                        if len(priors[i][j]) != 2:
                            raise ValueError("Each prior range list or tuple must have exactly two elements: min and max.")

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
    rms = summ['conservative_rms']

    if len(int_peaks) > 2: # assume this means that source is extended instead of having more than 2 separate sources in this interior region
        int_peaks = int_peaks[:1]
        int_coords = int_coords[:1]

    # TODO: handle extended external source? or just ignore since extended sources are less likely to be real?

    int_info = list(zip(int_peaks, int_coords))
    if type(ext_peaks) is list:
        ext_info = list(zip(ext_peaks, ext_coords))
    else:
        ext_info = []
    all_peaks = int_info + ext_info # list of tuples (peak_value, (l_coord, m_coord))
    all_peaks.sort(reverse=True) # sort by peak value
    n_peaks = len(all_peaks)
    if n_sources is not None:
        if n_peaks != n_sources:
            print(f"Warning: Number of peaks detected ({n_peaks}) does not match n_sources ({n_sources}). Proceeding with {n_sources}, but results may not be realiable")
    else:
        n_sources = n_peaks

    # Clean up priors
    vis_priors = []
    if priors is None:
        priors = [None] * n_sources
    if len(priors) == 1 and n_sources > 1: # if only one set of priors provided, use for all sources
        priors = priors * n_sources
    for i in range(len(priors)):
        mini_vis_priors = []
        if priors[i] is not None:
            for j in range(len(priors[i])):
                if priors[i][j] is not None:
                    if j == 0:  # peak, keep as is
                        mini_vis_priors.append(priors[i][j])
                    else:  # ra, dec
                        # convert from arcsec to radian
                        rad_min = float(Angle(priors[i][j][0], units.arcsec).to(units.radian).value)
                        rad_max = float(Angle(priors[i][j][1], units.arcsec).to(units.radian).value)
                        if type(priors[i][j]) is tuple:
                            mini_vis_priors.append((rad_min, rad_max))
                        else:
                            mini_vis_priors.append([rad_min, rad_max])
                else:
                    mini_vis_priors.append([None, None])
            vis_priors.append(mini_vis_priors)
        else: # nothing to convert
            vis_priors.append([[None, None]] * 6)

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

    # Calculate n_params and n_walkers
    permutation = tuple(['p'] * n_sources)
    n_params = 0
    for i in range(n_sources):
        n_params += SOURCE_TYPES['p'][0]
    n_walkers = 2 * n_params

    # Initial guesses
    for i in range(n_sources):
        peak = all_peaks[i][0] if i < n_peaks else all_peaks[-1][0]
        coord0 = all_peaks[i][1] if i < n_peaks else all_peaks[-1][1]
        rad_coord = (float(Angle(coord0[0], units.arcsec).to(units.radian).value), float(Angle(coord0[1], units.arcsec).to(units.radian).value))
        if i == 0:
            p0 = SOURCE_TYPES['p'][1](peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers)
        else:
            mini_p0 = SOURCE_TYPES['p'][1](peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers)
            if i >= n_peaks: # edit ra, dec initial guesses
                for j in range(n_walkers):
                    mini_p0[j,1] = np.random.uniform(-naxis1/2*rad_pix, naxis1/2*rad_pix)
                    mini_p0[j,2] = np.random.uniform(-naxis1/2*rad_pix, naxis1/2*rad_pix)
            p0 = np.append(p0, mini_p0, axis=1)

    # Set up and run MCMC
    all_results = []
    n_steps = 100
    sampler = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=(permutation, vis_priors, re, im, u, v, w, rad_bmaj, rad_bmin))
    try:
        state = sampler.run_mcmc(p0, n_steps)
    except emcee.autocorr.AutocorrError:
        pass
    except ValueError:
        raise ValueError(f"Error encountered during MCMC run.")
    tau = sampler.get_autocorr_time(quiet=True)
    if np.isnan(tau).all():
        raise RuntimeError(f"Autocorrelation time for parameters could not be estimated; all values are NaN.")
    int_tau = math.ceil(np.nanmax(tau))
    steps_to_50_tau = abs(int_tau * 50 - n_steps)
    try:
        sampler.run_mcmc(state, steps_to_50_tau)
    except ValueError:
        raise ValueError(f"Error encountered during MCMC run.")
    chain = sampler.get_chain(discard = int_tau * 10, flat=True)
    log_probs = sampler.get_log_prob(discard = int_tau * 10, flat=True)
    max_prob_index = np.argmax(log_probs)

    # Find parameter estimates and uncertainties and calculate chi2
    result = {}
    model = 0.0
    start = 0
    for i in range(n_sources):
        n_source_params = SOURCE_TYPES['p'][0]
        source_chain = chain[:, start:start+n_source_params]
        source_result = {}
        temp_medians = [] # to store medians
        temp_bests = {} # to store best values (that maximimize probability)
        temp_max_probs = [] # to store max prob values
        for j in range(n_source_params):
            samples = source_chain[:, j]
            temp_max_probs.append(samples[max_prob_index])
            samples_med = np.median(samples)
            samples_sd = np.nanstd(samples)
            param_name = SOURCE_TYPES['p'][4][j]
            source_result[param_name] = (float(samples_med), float(samples_sd))
            temp_bests[param_name] = float(samples[max_prob_index])
            temp_medians.append(samples_med)
        source_result['best'] = temp_bests
        model += SOURCE_TYPES['p'][3](temp_max_probs, u, v, rad_bmaj, rad_barea)
        result[f'source_{i+1}'] = source_result
        start += n_source_params
    chi2 = float(np.sum(w * ((re - model.real)**2 + (im - model.imag)**2)))
    n = len(re)
    k = n_params
    red_chi2 = chi2 / (n - k)

    all_results.append({'n_sources': n_sources, 'result': result, 'reduced_chi2': red_chi2, 'chain': chain})

    if clean_output:
        result = all_results[0]['result']
        start = 0
        permutation_chain = all_results[0]['chain']
        for i in range(n_sources):
            source_key = f'source_{i+1}'
            source_result = result[source_key]
            source_params = SOURCE_TYPES['p'][4]
            n_source_params = SOURCE_TYPES['p'][0]
            n_walkers = 2 * n_source_params
            source_chain = permutation_chain[:, start:start+n_source_params]

            # peak
            peak_chain = source_chain[:, 0]
            peak_sigmas = tuple([float(sigfig.round(sigma, sigfigs=3)) for sigma in sigmas(peak_chain)])
            source_result['peak'] = (round_tuple((source_result['peak'][0], source_result['peak'][1])), peak_sigmas)

            # convert ra, dec to arcsec
            ra_chain = source_chain[:, 1]
            dec_chain = source_chain[:, 2]
            ra_sigmas = tuple([float(sigfig.round(Angle(sigma, units.radian).to(units.arcsec).value, sigfigs=3)) for sigma in sigmas(ra_chain)])
            dec_sigmas = tuple([float(sigfig.round(Angle(sigma, units.radian).to(units.arcsec).value, sigfigs=3)) for sigma in sigmas(dec_chain)])
            source_result['ra'] = (round_tuple(tuple([float(Angle(l, units.radian).to(units.arcsec).value) for l in source_result['ra']])), ra_sigmas)
            source_result['dec'] = (round_tuple(tuple([float(Angle(m, units.radian).to(units.arcsec).value) for m in source_result['dec']])), dec_sigmas)

            del source_result['best']

            start += n_source_params

    if corner_plot:
        result = all_results[0]['result']
        chain = all_results[0]['chain']
        start = 0
        end = 0
        for i in range(n_sources):
            source_key = f'source_{i+1}'
            source_result = result[source_key]
            n_params = SOURCE_TYPES['p'][0]
            source_params = SOURCE_TYPES['p'][4]
            end += n_params
            if i == n_sources-1:
                fig = corner.corner(chain[:, start:], labels=source_params)
            else:
                fig = corner.corner(chain[:, start:end], labels=source_params)
            fig.suptitle(f'Source {i+1} of {n_sources}')
            start = end

    del all_results[0]['chain']

    return all_results

def uv_fit(fits_file: str, sources: list, priors: list = None, clean_output=True, corner_plot=True, additional_runs: int = 2):
    # priors = [[(peak_min, peak_max), (ra_min, ra_max), (dec_min, dec_max), (width_param_min, width_param_max), (ratio_min, ratio_max), (theta_min, theta_max)], ...]
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
        if len(priors) != len(sources) or len(priors) != 1:
            raise ValueError("Length of priors must match length of sources, or priors must be of length 1 (one set of priors for all sources).")
        for i in range(len(priors)):
            if priors[i] is not None:
                if type(priors[i]) is not list:
                    raise ValueError("Each element in priors must be None or a list corresponding to a source.")
                if len(priors[i]) != 6:
                    raise ValueError("Each prior list must have 6 elements corresponding to ranges for peak, RA, declination, width parameter, ratio, and angle.")
                for j in range(len(priors[i])):
                    if priors[i][j] is not None:
                        if type(priors[i][j]) not in [list, tuple]:
                            raise ValueError("Each prior range must be None, a list (inclusive), or a tuple (exclusive).")
                        if len(priors[i][j]) != 2:
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
    if len(priors) == 1 and len(sources) > 1: # if only one set of priors provided, use for all sources
        priors = priors * len(sources)
    for i in range(len(priors)):
        mini_vis_priors = []
        if priors[i] is not None:
            for j in range(len(priors[i])):
                if priors[i][j] is not None:
                    if j == 0:  # peak, keep as is
                        mini_vis_priors.append(priors[i][j])
                    elif j in [1, 2, 3]:  # ra, dec, width parameter
                        # convert from arcsec to radian
                        rad_min = float(Angle(priors[i][j][0], units.arcsec).to(units.radian).value)
                        rad_max = float(Angle(priors[i][j][1], units.arcsec).to(units.radian).value)
                        if type(priors[i][j]) is tuple:
                            mini_vis_priors.append((rad_min, rad_max))
                        else:
                            mini_vis_priors.append([rad_min, rad_max])
                    elif j == 4:  # ratio
                        mini_vis_priors.append(priors[i][j])
                    elif j == 5:  # angle
                        mini_vis_priors.append(priors[i][j] * np.pi/180) # convert from degrees to radians
                else:
                    mini_vis_priors.append([None, None])
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
    # TODO: resolved case, MRS
    # TODO: total flux matching image peak, conclude unresolved; or median of smallest 5% within uncertainty of median of ofther 95%

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
                if i >= n_peaks: # edit ra, dec initial guesses
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
        except ValueError:
            print(f"Error encountered during MCMC run for permutation {permutation}. Skipping this permutation.")
            all_results.append({'permutation': permutation, 'n_params': n_params, 'result': None, 'chi2': np.inf, 'chain': None})
            continue
        tau = sampler.get_autocorr_time(quiet=True)
        if np.isnan(tau).all():
            warnings.warn(f"Autocorrelation time for first run of {permutation} could not be estimated; all values are NaN.", RuntimeWarning)
            all_results.append({'permutation': permutation, 'n_params': n_params, 'result': None, 'chi2': np.inf, 'chain': None})
            continue
        int_tau = math.ceil(np.nanmax(tau))
        steps_to_50_tau = abs(int_tau * 50 - n_steps)
        try:
            sampler.run_mcmc(state, steps_to_50_tau)
        except ValueError:
            print(f"Error encountered during second MCMC run for permutation {permutation}. Skipping this permutation.")
            all_results.append({'permutation': permutation, 'n_params': n_params, 'result': None, 'chi2': np.inf, 'chain': None})
            continue
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
            model += SOURCE_TYPES[source][3](temp_max_probs, u, v, rad_bmaj, rad_barea)
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
                        if i >= n_peaks: # edit ra, dec initial guesses
                            for j in range(n_walkers):
                                mini_p1[j,1] = np.random.uniform(-naxis1/2*rad_pix, naxis1/2*rad_pix)
                                mini_p1[j,2] = np.random.uniform(-naxis1/2*rad_pix, naxis1/2*rad_pix)
                        p1 = np.append(p1, mini_p1, axis=1)
                else:
                    rad_position = best_result[f'source_{i+1}']['best']['ra'], best_result[f'source_{i+1}']['best']['dec']
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
                        if best_params[3] < rad_bmaj/2:  # conditions for unresolved source
                            resolved = False
                    if not resolved:
                        temp_perm = best_perm[:i] + ('p',) + best_perm[i+1:]
                        if temp_perm == best_perm:
                            temp = best_result[f'source_{i+1}']['best']['peak']
                        else:
                            temp = uv_fit(fits_file, list(temp_perm), priors=priors, clean_output=True, corner_plot=False, additional_runs=0)[0]['result'][f'source_{i+1}']['peak'][0]
                            point_intensity = temp if type(temp) is float else temp[0]

                    # use c fitting to help g fitting if c chi2 was better than g chi2 in first run
                    c_peak = None
                    c_sigma = None
                    if source == 'g' and best_perm[i] == 'c':
                        c_peak = best_result[f'source_{i+1}']['best']['peak']
                        c_sigma = best_result[f'source_{i+1}']['best']['sigma']
                    if i == 0:
                        p1 = all_p1(med_sd, resolved, point_intensity, c_peak, c_sigma, rad_position, rad_bmaj, rad_pix, n_walkers, chain)
                    else:
                        p1 = np.append(p1, all_p1(med_sd, resolved, point_intensity, c_peak, c_sigma, rad_position, rad_bmaj, rad_pix, n_walkers, chain), axis=1)

                    # edit vis_priors if unresolved source
                    if not resolved:
                        if vis_priors[i] is None:
                            vis_priors[i] = [(None, None)] * 6
                        vis_priors[i][1] = (-rad_pix+rad_coord[0], rad_pix+rad_coord[0]) # ra within one pixel of image domain result
                        vis_priors[i][2] = (-rad_pix+rad_coord[1], rad_pix+rad_coord[1]) # dec within one pixel of image domain result

            # Set up and run MCMC
            n_steps = 100
            sampler1 = emcee.EnsembleSampler(n_walkers, n_params, log_probability, args=(permutation, vis_priors, re, im, u, v, w, rad_bmaj, rad_bmin))
            try:
                state = sampler1.run_mcmc(p1, n_steps)
            except emcee.autocorr.AutocorrError:
                pass
            except ValueError:
                print(f"Error encountered during second MCMC run for permutation {permutation}. Skipping this permutation.")
                second_results.append({'permutation': permutation, 'n_params': n_params, 'result': None, 'chi2': np.inf, 'chain': None})
                continue
            tau = sampler1.get_autocorr_time(quiet=True)
            if np.isnan(tau).all():
                warnings.warn(f"Autocorrelation time for second run of {permutation} could not be estimated; all values are NaN.", RuntimeWarning)
                second_results.append({'permutation': permutation, 'n_params': n_params, 'result': None, 'chi2': np.inf, 'chain': None})
                continue
            int_tau = math.ceil(np.nanmax(tau))
            steps_to_50_tau = abs(int_tau * 50 - n_steps)
            try:
                sampler1.run_mcmc(state, steps_to_50_tau)
            except ValueError:
                print(f"Error encountered during second MCMC run for permutation {permutation}. Skipping this permutation.")
                second_results.append({'permutation': permutation, 'n_params': n_params, 'result': None, 'chi2': np.inf, 'chain': None})
                continue
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
                model += SOURCE_TYPES[source][3](temp_max_probs, u, v, rad_bmaj, rad_barea)
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

                # peak
                peak_chain = source_chain[:, 0]
                peak_sigmas = tuple([float(sigfig.round(sigma, sigfigs=3)) for sigma in sigmas(peak_chain)])
                source_result['peak'] = (round_tuple((source_result['peak'][0], source_result['peak'][1])), peak_sigmas)

                # convert ra, dec to arcsec
                ra_chain = source_chain[:, 1]
                dec_chain = source_chain[:, 2]
                ra_sigmas = tuple([float(sigfig.round(Angle(sigma, units.radian).to(units.arcsec).value, sigfigs=3)) for sigma in sigmas(ra_chain)])
                dec_sigmas = tuple([float(sigfig.round(Angle(sigma, units.radian).to(units.arcsec).value, sigfigs=3)) for sigma in sigmas(dec_chain)])
                source_result['ra'] = (round_tuple(tuple([float(Angle(l, units.radian).to(units.arcsec).value) for l in source_result['ra']])), ra_sigmas)
                source_result['dec'] = (round_tuple(tuple([float(Angle(m, units.radian).to(units.arcsec).value) for m in source_result['dec']])), dec_sigmas)

                if source_type != 'p': # convert visibility width to image width in arcsec
                    width_chain = source_chain[:, 3]
                    width_sigmas = tuple([float(sigfig.round(Angle(sigma, units.radian).to(units.arcsec).value, sigfigs=3)) for sigma in sigmas(width_chain)])
                    source_result[source_params[3]] = (round_tuple((float(Angle(source_result[source_params[3]][0], units.radian).to(units.arcsec).value), \
                                                    float(Angle(source_result[source_params[3]][1], units.radian).to(units.arcsec).value))), width_sigmas)

                if source_type == 'g': # convert visibility theta to image theta in degrees and convert sigma and ratio into major and minor
                    theta_chain = source_chain[:, 5]
                    vis_theta_sigmas = sigmas(theta_chain)
                    theta_sigmas = tuple([float(sigfig.round((theta * 180/np.pi), sigfigs=3)) for theta in vis_theta_sigmas])
                    uvis_theta = ufloat(source_result['vis_theta'][0], source_result['vis_theta'][1])
                    uimg_theta = (uvis_theta * (180/np.pi))
                    del source_result['vis_theta']
                    source_result['theta'] = (round_tuple((uimg_theta.n, uimg_theta.s)), theta_sigmas)

                    usigma_min = ufloat(source_result['sigma'][0][0], source_result['sigma'][0][1])
                    uratio = ufloat(source_result['ratio'][0], source_result['ratio'][1])
                    usigma_maj = usigma_min / uratio
                    del source_result['sigma']
                    del source_result['ratio']
                    source_result['sigma_maj'] = (round_tuple((usigma_maj.n, usigma_maj.s)), tuple([float(sigfig.round(width / uratio.n, sigfigs=3)) for width in width_sigmas]))
                    source_result['sigma_min'] = (round_tuple((usigma_min.n, usigma_min.s)), tuple([float(round(width, 3)) for width in width_sigmas]))

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
