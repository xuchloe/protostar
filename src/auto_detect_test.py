from astropy.io import fits
import numpy as np
import emcee
import corner
import math
from astropy.coordinates import Angle
import astropy.units as units
import scipy.special as sp
from scipy.stats import norm
import warnings
import itertools
from uncertainties import ufloat
from uv_fit import *
import sigfig
from matplotlib import pyplot as plt

# Source parameters -- do not change these
P_PARAMS = ['peak', 'ra', 'dec']
C_PARAMS = ['peak', 'ra', 'dec', 'sigma']
G_PARAMS = ['peak', 'ra', 'dec', 'sigma', 'ratio', 'vis_theta']
D_PARAMS = ['peak', 'ra', 'dec', 'r']

SOURCE_TYPES = {'p': [3, p_p0, p_prior, p_model, P_PARAMS], \
                'c': [4, c_p0, c_prior, c_model, C_PARAMS], \
                'g': [6, g_p0, g_prior, g_model, G_PARAMS], \
                'd': [4, d_p0, d_prior, d_model, D_PARAMS]}

def generate_synthetic_info_vis(fits_file, sources, peaks, coords, noise, widths=None, ratios=None, thetas=None):
    # peak in Jy, coords in arcsec, noise in Jy, widths in arcsec, ratios unitless, thetas in degrees
    num_sources = len(sources)
    if num_sources == 0:
        raise ValueError("At least one peak, coordinate, and source type must be provided.")
    if num_sources != len(peaks):
        raise ValueError("Length of sources and peaks must be the same.")
    if num_sources != len(coords):
        raise ValueError("Length of sources and coords must be the same.")
    for i in range(num_sources):
        if sources[i] not in SOURCE_TYPES.keys():
            raise ValueError(f"Source type '{sources[i]}' is not recognized. Source type must be one of the following: \
                            'p' (point), 'c' (circular gaussian), 'g' (gaussian), or 'd' (disk).")
        if type(coords[i]) not in [tuple, list] or len(coords[i]) != 2:
            raise ValueError(f"Coordinate for source {i+1} must be a tuple or list of length 2 representing (ra, dec) in radians.")
    if widths is not None and num_sources != len(widths):
        raise ValueError("Length of sources and widths must be the same if specifying widths.")
    if ratios is not None and num_sources != len(ratios):
        raise ValueError("Length of sources and ratios must be the same if specifying ratios.")
    if thetas is not None and num_sources != len(thetas):
        raise ValueError("Length of sources and thetas must be the same if specifying thetas.")

    num_widths = 0
    num_ratios = 0
    num_thetas = 0
    for source in sources:
        if source in ['c', 'd', 'g']:
            num_widths += 1
            if source == 'g':
                num_ratios += 1
                num_thetas += 1

    if widths is not None:
        if len(widths) != num_widths:
            raise ValueError(f"{num_widths} sources require widths, but only {len(widths)} have been given.")
    if ratios is not None:
        if len(ratios) != num_ratios:
            raise ValueError(f"{num_ratios} sources require ratios, but only {len(ratios)} have been given.")
    if thetas is not None:
        if len(thetas) != num_thetas:
            raise ValueError(f"{num_thetas} sources require thetas, but only {len(thetas)} have been given.")

    file = fits.open(fits_file)
    data = file[1].data
    cdelt1 = file[0].header['CDELT1']
    cunit1 = file[0].header['CUNIT1']
    naxis1 = file[0].header['NAXIS1']
    naxis2 = file[0].header['NAXIS2']
    bmaj = file[0].header['BMAJ']
    bmin = file[0].header['BMIN']

    arcsec_bmaj = Angle(bmaj, cunit1).to(units.arcsec).value
    search_radius = arcsec_bmaj + 2

    pix_field_center = (0,0)

    int_peaks = []
    int_coords = []
    ext_peaks = []
    ext_coords = []

    for i in range(num_sources):
        if np.sqrt((coords[i][0] - pix_field_center[0])**2 + (coords[i][1] - pix_field_center[1])**2) <= search_radius:
            int_peaks.append(peaks[i])
            int_coords.append(coords[i])
        else:
            ext_peaks.append(peaks[i])
            ext_coords.append(coords[i])

    info = {}
    info['CDELT1'] = cdelt1
    info['CUNIT1'] = cunit1
    info['NAXIS1'] = naxis1
    info['BMAJ'] = bmaj
    info['BMIN'] = bmin
    info['int_peak_val'] = int_peaks
    info['int_peak_coord'] = int_coords
    info['ext_peak_val'] = ext_peaks
    info['ext_peak_coord'] = ext_coords
    info['conservative_rms'] = noise

    vis_err = noise * np.sqrt(len(data)) * np.sqrt(2)
    weight = 1/(vis_err**2)

    vis = []
    clean_re = []
    clean_im = []

    width_counter = 0
    g_counter = 0
    for i in range(num_sources):
        l0 = Angle(coords[i][0], units.arcsec).to(units.radian).value
        m0 = Angle(coords[i][1], units.arcsec).to(units.radian).value
        model_func = SOURCE_TYPES[sources[i]][3]
        source_info = [peaks[i], l0, m0]
        if sources[i] in ['c', 'd', 'g']:
            source_info.append(Angle(widths[width_counter], units.arcsec).to(units.radian).value)
            width_counter += 1
            if sources[i] == 'g':
                source_info.append(ratios[g_counter])
                source_info.append(thetas[g_counter] * np.pi/180) # convert from degrees to radians
                g_counter += 1
        for j in range(len(data)):
            row = data[j]
            model = model_func(source_info, row['U'], row['V'], bmaj, np.pi * bmaj * bmin / (4 * np.log(2)))
            if i == 0:
                clean_re.append(model.real)
                clean_im.append(model.imag)
            else:
                clean_re[j] += model.real
                clean_im[j] += model.imag

    for i in range(len(data)):
        row = data[i]
        new_row = (row['Frequency'], row['U'], row['V'], (np.random.normal(scale=vis_err) + clean_re[i]) * weight, (np.random.normal(scale=vis_err) + clean_im[i]) * weight, weight)
        vis.append(new_row)

    return info, vis

def sim_auto_detect(info, vis, n_sources: int = None, clean_output=True, corner_plot=True):

    # Extract data from fits file
    cdelt1 = info['CDELT1']
    cunit1 = info['CUNIT1']
    naxis1 = info['NAXIS1']

    bmaj = info['BMAJ'] # cunit1
    bmin = info['BMIN'] # cunit1
    rad_bmaj = Angle(bmaj, cunit1).to(units.radian).value
    rad_bmin = Angle(bmin, cunit1).to(units.radian).value
    rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))
    rad_pix = float(Angle(cdelt1, cunit1).to(units.radian).value)
    int_peaks = info['int_peak_val']
    int_coords = info['int_peak_coord']
    ext_peaks = info['ext_peak_val']
    ext_coords = info['ext_peak_coord']
    rms = info['conservative_rms']

    if len(int_peaks) > 2: # assume this means that source is extended instead of having more than 2 separate sources in this interior region
        int_peaks = int_peaks[:1]
        int_coords = int_coords[:1]

    # TODO: handle extended external source? or just ignore since extended sources are less likely to be real?

    int_peaks.sort(reverse=True) # descending peak value
    int_info = list(zip(int_peaks, int_coords))
    if type(ext_peaks) is list:
        ext_peaks.sort(reverse=True) # descending peak value
        ext_info = list(zip(ext_peaks, ext_coords))
    else:
        ext_info = []
    all_peaks = int_info + ext_info # list of tuples (peak_value, (l_coord, m_coord)), int in descending peaks then ext in descending peaks
    n_peaks = len(all_peaks)
    if n_sources is not None:
        if n_peaks != n_sources:
            print(f"Warning: Number of peaks detected ({n_peaks}) does not match n_sources ({n_sources}). Proceeding with {n_sources}, but results may not be realiable")
    else:
        n_sources = n_peaks

    vis_priors = [[[None, None] for _ in range(6)] for _ in range(n_sources)]
    for i in range(n_sources):
        snr = all_peaks[i][0]/rms if i < n_peaks else all_peaks[-1][0]/rms
        min_position_delta = rad_bmaj/50
        position_delta = max(3*rad_bmaj/snr, min_position_delta) if snr > 0 else min_position_delta
        img_min = int(- naxis1/ 2)* rad_pix # assumes odd number of pixels and center pixel is at (0,0)
        img_max = int(naxis1/ 2)* rad_pix # assumes odd number of pixels and center pixel is at (0,0)
        temp_ra = Angle(all_peaks[i][1][0], units.arcsec).to(units.radian).value if i < n_peaks else None
        temp_dec = Angle(all_peaks[i][1][1], units.arcsec).to(units.radian).value if i < n_peaks else None
        ra_min = max(img_min, (temp_ra - position_delta)) if temp_ra is not None else img_min # loosest prior is image edges
        ra_max = min(img_max, (temp_ra + position_delta)) if temp_ra is not None else img_max
        dec_min = max(img_min, (temp_dec - position_delta)) if temp_dec is not None else img_min
        dec_max = min(img_max, (temp_dec + position_delta)) if temp_dec is not None else img_max
        vis_priors[i][1] = [ra_min, ra_max]
        vis_priors[i][2] = [dec_min, dec_max]

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

    all_results = []

    # Calculate n_params and n_walkers
    src_type = 'p' # for now to do just point sources
    permutation = tuple([src_type] * n_sources)
    n_params = 0
    for i in range(n_sources):
        n_params += SOURCE_TYPES[src_type][0]
    n_walkers = 2 * n_params

    total_flux = None
    # if src_type == 'c':
    #     total_flux = 4 * all_results[0]['result'][f'source_1']['peak'][0] # so that sigma guess becomes roughly 1 beam major fwhm

    # Initial guesses
    for i in range(n_sources):
        peak = all_peaks[i][0] if i < n_peaks else all_peaks[-1][0]
        if not (i < n_peaks):
            print(f"Source {i+1} has no corresponding peak.")
        coord0 = all_peaks[i][1] if i < n_peaks else all_peaks[-1][1]
        rad_coord = (float(Angle(coord0[0], units.arcsec).to(units.radian).value), float(Angle(coord0[1], units.arcsec).to(units.radian).value))
        if i == 0:
            p0 = SOURCE_TYPES[src_type][1](peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers)
        else:
            mini_p0 = SOURCE_TYPES[src_type][1](peak, rad_coord, rad_pix, rad_barea, total_flux, n_walkers)
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
        raise ValueError(f"Error encountered during MCMC run.")
    tau = sampler.get_autocorr_time(quiet=True)
    if np.isnan(tau).all():
        raise RuntimeError(f"Autocorrelation time for parameters could not be estimated; all values are NaN.")
    int_tau = math.ceil(np.nanmax(tau))
    steps_to_50_tau = abs(int_tau * 50 - n_steps)
    try:
        sampler.run_mcmc(state, steps_to_50_tau, skip_initial_state_check=True)
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
        n_source_params = SOURCE_TYPES[src_type][0]
        source_chain = chain[:, start:start+n_source_params]
        source_result = {'type': src_type}
        temp_medians = [] # to store medians
        temp_bests = {} # to store best values (that maximimize probability)
        temp_max_probs = [] # to store max prob values
        for j in range(n_source_params):
            samples = source_chain[:, j]
            temp_max_probs.append(samples[max_prob_index])
            samples_med = np.median(samples)
            samples_sd = np.nanstd(samples)
            param_name = SOURCE_TYPES[src_type][4][j]
            source_result[param_name] = (float(samples_med), float(samples_sd))
            temp_bests[param_name] = float(samples[max_prob_index])
            temp_medians.append(samples_med)
        source_result['best'] = temp_bests
        model += SOURCE_TYPES[src_type][3](temp_max_probs, u, v, rad_bmaj, rad_barea)
        result[f'source_{i+1}'] = source_result
        start += n_source_params
    chi2 = float(np.sum(w * ((re - model.real)**2 + (im - model.imag)**2)))
    n = len(re)
    k = n_params
    bic = k * np.log(n) + chi2
    all_results.append({'n_sources': n_sources, 'result': result, 'bic': bic, 'chain': chain})

    # all_results.sort(key=lambda x: x['bic']) # lowest to highest BIC

    if clean_output:
        result = all_results[0]['result']
        # start = 0
        permutation_chain = all_results[0]['chain']
        for i in range(n_sources):
            source_key = f'source_{i+1}'
            source_result = result[source_key]
            src_type = source_result['type']
            source_params = SOURCE_TYPES[src_type][4]
            n_source_params = SOURCE_TYPES[src_type][0]
            n_walkers = 2 * n_source_params
            source_chain = permutation_chain #[:, start:start+n_source_params]

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

            # if src_type == 'c':
            #     # convert sigma to arcsec
            #     sigma_chain = source_chain[:, 3]
            #     sigma_sigmas = tuple([float(sigfig.round(Angle(sigma, units.radian).to(units.arcsec).value, sigfigs=3)) for sigma in sigma_chain])
            #     source_result['sigma'] = (round_tuple(tuple([float(Angle(s, units.radian).to(units.arcsec).value) for s in source_result['sigma']])), sigma_sigmas)
            del source_result['best']

            # start += n_source_params

    if corner_plot:
        result = all_results[0]['result']
        chain = all_results[0]['chain']
        start = 0
        end = 0
        for i in range(n_sources):
            source_key = f'source_{i+1}'
            source_result = result[source_key]
            src_type = source_result['type']
            n_params = SOURCE_TYPES[src_type][0]
            source_params = SOURCE_TYPES[src_type][4]
            end += n_params
            if i == n_sources-1:
                fig = corner.corner(chain[:, start:], labels=source_params)
            else:
                fig = corner.corner(chain[:, start:end], labels=source_params)
            fig.suptitle(f'Source {i+1} of {n_sources}')
            start = end

    del all_results[0]['chain']

    return all_results

def score(z):
    k = int(abs(z)) # sigma bin
    numerator = 2 * (norm.cdf(k+1) - norm.cdf(k)) # probability of being in this sigma bin
    denominator = 2 * (norm.cdf(1) - norm.cdf(0)) # probability of being in 0 to 1 sigma bin, normalization factor so that score of 1 corresponds to being in 0 to 1 sigma bin
    return numerator / denominator

def average_points(fits_file, sources, peaks, coords, noise, widths=None, ratios=None, thetas=None, reps=100):
    all_points = []
    peak_pts = []
    ra_pts = []
    dec_pts = []
    pixel = 0.214255908880632
    image_edge = pixel * 194
    num_pixels = 194**2
    beam_maj = 1.96

    all_peaks = {}
    all_ras = {}
    all_decs = {}
    # all_sigmas = {}
    num_errors = 0
    for i in range(reps):
        # give some variation to parameters
        peaks = [peak * np.random.uniform(0.95, 1.05) for peak in peaks] # plus or minus 5% on snr
        coords = [(ra + np.random.uniform(-beam_maj/2, beam_maj/2), dec + np.random.uniform(-beam_maj/2, beam_maj/2)) for ra, dec in coords] # plus or minus half beam major fwhm on position

        info, vis = generate_synthetic_info_vis(fits_file, sources, peaks, coords, noise, widths, ratios, thetas)
        pts = 0
        result = sim_auto_detect(info, vis, corner_plot=False)[0]
        # try:
        #     result = sim_auto_detect(info, vis, corner_plot=False)[0]
        # except:
        #     print(f"Error occurred while running simulation for rep {i}")
        #     num_errors += 1
        #     continue
        counter = 0
        # gaussians = 0
        for key, res in result['result'].items():
            if key not in all_peaks:
                all_peaks[key] = []
            if key not in all_ras:
                all_ras[key] = []
            if key not in all_decs:
                all_decs[key] = []
            src_type = res['type']
            peak = (float(res['peak'][0][0]), float(res['peak'][0][1]))
            all_peaks[key].append(peak[0]-peaks[counter]) # the delta
            ra = (float(res['ra'][0][0]), float(res['ra'][0][1]))
            all_ras[key].append(ra[0]-coords[counter][0]) # the delta
            dec = (float(res['dec'][0][0]), float(res['dec'][0][1]))
            all_decs[key].append(dec[0]-coords[counter][1]) # the delta
            # if src_type == 'c':
            #     if key not in all_sigmas:
            #         all_sigmas[key] = []
            #     sigma = (float(res['sigma'][0][0]), float(res['sigma'][0][1]))
            #     all_sigmas[key].append(sigma[0])

            # immediately gets -1 point if spurious detection
            significance_threshold = norm.ppf(-1/num_pixels, loc=0, scale=1) # threshold for being a once in image significant detection
            z_peak = (peak[0]-peaks[counter]) / noise if peak[1] > 0 else float('inf')
            if (abs(ra[0]-coords[counter][0])>5*(beam_maj/2) or abs(dec[0]-coords[counter][1])>5*(beam_maj/2)) and z_peak >= significance_threshold: # spurious detection if at least 5 sigma off in position and sigificant peak
                pts = -1
                peak_pts.append(-1/3)
                ra_pts.append(-1/3)
                dec_pts.append(-1/3)
                counter += 1
                continue

            # peak points
            pts += score(z_peak)
            peak_pts.append(score(z_peak))

            # ra points
            z_ra = (ra[0]-coords[counter][0]) / beam_maj if ra[1] > 0 else float('inf')
            pts += score(z_ra)
            ra_pts.append(score(z_ra))

            # dec points
            z_dec = (dec[0]-coords[counter][1]) / beam_maj if dec[1] > 0 else float('inf')
            pts += score(z_dec)
            dec_pts.append(score(z_dec))

            counter += 1
        all_points.append(pts)

    # histograms of deltas
    if all_peaks:
        peak_keys = list(all_peaks.keys())
        pfig, pax = plt.subplots(nrows=1, ncols=len(peak_keys))
        if len(peak_keys) == 1:
            pax = [pax]
        for i in range(len(peak_keys)):
            key = peak_keys[i]
            pax[i].hist(all_peaks[key], bins=20, color='b', edgecolor='k')
            pax[i].set_title(f'Peak, {key}')
    if all_ras:
        ra_keys = list(all_ras.keys())
        rfig,rax = plt.subplots(nrows=1, ncols=len(ra_keys))
        if len(ra_keys) == 1:
            rax = [rax]
        for i in range(len(ra_keys)):
            key = ra_keys[i]
            rax[i].hist(all_ras[key], bins=20, color='g', edgecolor='k')
            rax[i].set_title(f'RA, {key}')
    if all_decs:
        dec_keys = list(all_decs.keys())
        dfig,dax = plt.subplots(nrows=1, ncols=len(dec_keys))
        if len(dec_keys) == 1:
            dax = [dax]
        for i in range(len(dec_keys)):
            key = dec_keys[i]
            dax[i].hist(all_decs[key], bins=20, color='r', edgecolor='k')
            dax[i].set_title(f'Dec, {key}')
    # if all_sigmas:
    #     sigma_keys = list(all_sigmas.keys())
    #     sfig,sax = plt.subplots(nrows=1, ncols=len(sigma_keys))
    #     if len(sigma_keys) == 1:
    #         sax = [sax]
    #     for i in range(len(sigma_keys)):
    #         key = sigma_keys[i]
    #         sax[i].hist(all_sigmas[key], bins=20, color='c', edgecolor='k')
    #         sax[i].set_title(f'Sigma, {key}')

    return {'average total points': round(float(np.mean(all_points)),2), 'std total points': round(float(np.std(all_points)),2),\
            'average peak points': round(float(np.mean(peak_pts)),2), 'std peak points': round(float(np.std(peak_pts)),2),\
            'average ra points': round(float(np.mean(ra_pts)),2), 'std ra points': round(float(np.std(ra_pts)),2),\
            'average dec points': round(float(np.mean(dec_pts)),2), 'std dec points': round(float(np.std(dec_pts)),2),\
            'number of errors': num_errors}

def average_by_snr(fits_file, sources, coords, noise, widths=None, ratios=None, thetas=None, reps_per_snr=50, snr_vals=[3,4,5,6,7,8,9,10,20,50,100,200,500,1000]):
    # single source only for now, can be extended to multiple sources in the future
    pixel = 0.214255908880632
    image_edge = pixel * 194
    num_pixels = 194**2
    beam_maj = 1.96
    peaks_by_snr = []
    peak_std_by_snr = []
    ras_by_snr = []
    ra_std_by_snr = []
    decs_by_snr = []
    dec_std_by_snr = []
    # sigmas_by_snr = []
    # sigma_std_by_snr = []
    # snr_vals_for_sigma = []
    points_by_snr = []
    point_std_by_snr = []

    for snr in snr_vals:
        peaks = [snr * noise]

        all_points = []
        all_peaks = {}
        all_ras = {}
        all_decs = {}
        # all_sigmas = {}
        num_errors = 0
        for i in range(reps_per_snr):
            # give some variation to parameters
            coords = [(ra + np.random.uniform(-beam_maj/2, beam_maj/2), dec + np.random.uniform(-beam_maj/2, beam_maj/2)) for ra, dec in coords] # plus or minus half beam major fwhm on position
            info, vis = generate_synthetic_info_vis(fits_file, sources, peaks, coords, noise, widths, ratios, thetas)

            pts = 0
            try:
                result = sim_auto_detect(info, vis, corner_plot=False)[0]
            except:
                print(f"Error occurred while running simulation for rep {i}")
                num_errors += 1
                continue
            counter = 0
            # gaussians = 0
            for key, res in result['result'].items():
                if key not in all_peaks:
                    all_peaks[key] = []
                if key not in all_ras:
                    all_ras[key] = []
                if key not in all_decs:
                    all_decs[key] = []
                src_type = res['type']
                peak = (float(res['peak'][0][0]), float(res['peak'][0][1]))
                all_peaks[key].append(peak[0]-peaks[counter]) # the delta
                ra = (float(res['ra'][0][0]), float(res['ra'][0][1]))
                all_ras[key].append(ra[0]-coords[counter][0]) # the delta
                dec = (float(res['dec'][0][0]), float(res['dec'][0][1]))
                all_decs[key].append(dec[0]-coords[counter][1]) # the delta
                # if src_type == 'c':
                #     if key not in all_sigmas:
                #         all_sigmas[key] = []
                #     sigma = (float(res['sigma'][0][0]), float(res['sigma'][0][1]))
                #     all_sigmas[key].append(sigma[0])

                # immediately gets -1 point if spurious detection
                significance_threshold = norm.ppf(-1/num_pixels, loc=0, scale=1) # threshold for being a once in image significant detection
                z_peak = (peak[0]-peaks[counter]) / noise if peak[1] > 0 else float('inf')
                if (abs(ra[0]-coords[counter][0])>5*(beam_maj/2) or abs(dec[0]-coords[counter][1])>5*(beam_maj/2)) and z_peak >= significance_threshold: # spurious detection if at least 5 sigma off in position and sigificant peak
                    pts = -1
                    counter += 1
                    continue

                # peak points
                pts += score(z_peak)

                # ra points
                z_ra = (ra[0]-coords[counter][0]) / beam_maj if ra[1] > 0 else float('inf')
                pts += score(z_ra)

                # dec points
                z_dec = (dec[0]-coords[counter][1]) / beam_maj if dec[1] > 0 else float('inf')
                pts += score(z_dec)

                counter += 1
            all_points.append(pts)

        if all_peaks:
            peaks_by_snr.append(np.mean(list(all_peaks.values())))
            peak_std_by_snr.append(np.std(list(all_peaks.values())))
        if all_ras:
            ras_by_snr.append(np.mean(list(all_ras.values())))
            ra_std_by_snr.append(np.std(list(all_ras.values())))
        if all_decs:
            decs_by_snr.append(np.mean(list(all_decs.values())))
            dec_std_by_snr.append(np.std(list(all_decs.values())))
        # if all_sigmas:
        #     sigmas_by_snr.append(np.mean(list(all_sigmas.values())))
        #     sigma_std_by_snr.append(np.std(list(all_sigmas.values())))
        #     snr_vals_for_sigma.append(snr)
        points_by_snr.append(round(float(np.mean(all_points)),2))
        point_std_by_snr.append(round(float(np.std(all_points)),2))

    if peaks_by_snr:
        pfig, pax = plt.subplots()
        snr_measured = [peak/noise for peak in peaks_by_snr]
        snr_std = [std/noise for std in peak_std_by_snr]
        pax.errorbar(snr_vals, snr_measured, yerr=snr_std, fmt='o', color='b', ecolor='b', capsize=5)
        pax.set_xscale('log')
        pax.set_title('Delta SNR vs True SNR')
    if ras_by_snr:
        rfig, rax = plt.subplots()
        rax.errorbar(snr_vals, ras_by_snr, yerr=ra_std_by_snr, fmt='o', color='g', ecolor='g', capsize=5)
        rax.set_xscale('log')
        rax.set_title('Delta RA vs SNR')
    if decs_by_snr:
        dfig, dax = plt.subplots()
        dax.errorbar(snr_vals, decs_by_snr, yerr=dec_std_by_snr, fmt='o', color='r', ecolor='r', capsize=5)
        dax.set_xscale('log')
        dax.set_title('Delta Dec vs SNR')
    # if sigmas_by_snr:
    #     sfig, sax = plt.subplots()
    #     sax.errorbar(snr_vals_for_sigma, sigmas_by_snr, yerr=sigma_std_by_snr, fmt='o', color='c', ecolor='c', capsize=5)
    #     sax.set_xscale('log')
    #     sax.set_title('Average Sigma vs SNR')
    if points_by_snr:
        ptsfig, ptsax = plt.subplots()
        ptsax.errorbar(snr_vals, points_by_snr, yerr=point_std_by_snr, fmt='o', color='k', ecolor='k', capsize=5)
        ptsax.set_xscale('log')
        ptsax.set_title('Average Points vs SNR')

def average_by_width(fits_file, sources, coords, peaks, noise, width_vals=[0.1,0.2,0.5,1,1.5,2,3,4,5], ratios=None, thetas=None, reps_per_width=50):
    # single source only for now, can be extended to multiple sources in the future
    pixel = 0.214255908880632
    image_edge = pixel * 194
    beam_maj = 1.96
    peaks_by_width = []
    peak_std_by_width = []
    ras_by_width = []
    ra_std_by_width = []
    decs_by_width = []
    dec_std_by_width = []
    sigmas_by_width = []
    sigma_std_by_width = []
    width_vals_for_sigma = []
    points_by_width = []
    point_std_by_width = []

    for w in width_vals:
        widths = [w]
        info, vis = generate_synthetic_info_vis(fits_file, sources, peaks, coords, noise, widths, ratios, thetas)

        all_points = []
        all_peaks = {}
        all_ras = {}
        all_decs = {}
        all_sigmas = {}
        num_errors = 0
        for i in range(reps_per_width):
            pts = 0
            result = sim_auto_detect(info, vis, corner_plot=False)[0]
            try:
                result = sim_auto_detect(info, vis, corner_plot=False)[0]
            except:
                print(f"Error occurred while running simulation for rep {i}")
                num_errors += 1
                continue
            counter = 0
            gaussians = 0
            for key, res in result['result'].items():
                if key not in all_peaks:
                    all_peaks[key] = []
                if key not in all_ras:
                    all_ras[key] = []
                if key not in all_decs:
                    all_decs[key] = []
                src_type = res['type']
                peak = (float(res['peak'][0][0]), float(res['peak'][0][1]))
                all_peaks[key].append(peak[0])
                ra = (float(res['ra'][0][0]), float(res['ra'][0][1]))
                all_ras[key].append(ra[0])
                dec = (float(res['dec'][0][0]), float(res['dec'][0][1]))
                all_decs[key].append(dec[0])
                if src_type == 'c':
                    if key not in all_sigmas:
                        all_sigmas[key] = []
                    sigma = (float(res['sigma'][0][0]), float(res['sigma'][0][1]))
                    all_sigmas[key].append(sigma[0])
                if peak[0] - 3*peak[1] > noise and abs(ra[0]) <= image_edge and abs(dec[0]) <= image_edge: # significant detection and within image bounds
                    pts += 1
                if np.sqrt((ra[0]-coords[counter][0])**2 + (dec[0]-coords[counter][1])**2) < beam_maj and abs(peak[0]-peaks[counter]) < 3*noise:
                    if src_type == 'c':
                        if widths is not None:
                            if gaussians <= len(widths):
                                if abs(sigma[0] - widths[gaussians]) < 3*noise:
                                    pts += 1
                                gaussians += 1
                    else:
                        pts += 1
                counter += 1
            all_points.append(pts)

        if all_peaks:
            peaks_by_width.append(np.mean(list(all_peaks.values())))
            peak_std_by_width.append(np.std(list(all_peaks.values())))
        if all_ras:
            ras_by_width.append(np.mean(list(all_ras.values())))
            ra_std_by_width.append(np.std(list(all_ras.values())))
        if all_decs:
            decs_by_width.append(np.mean(list(all_decs.values())))
            dec_std_by_width.append(np.std(list(all_decs.values())))
        if all_sigmas:
            sigmas_by_width.append(np.mean(list(all_sigmas.values())))
            sigma_std_by_width.append(np.std(list(all_sigmas.values())))
            width_vals_for_sigma.append(w)
        points_by_width.append(round(float(np.mean(all_points)),2))
        point_std_by_width.append(round(float(np.std(all_points)),2))

    if peaks_by_width:
        pfig, pax = plt.subplots()
        pax.errorbar(width_vals, peaks_by_width, yerr=peak_std_by_width, fmt='o', color='b', ecolor='b', capsize=5)
        pax.set_title('Measured Peaks vs True Width')
    if ras_by_width:
        rfig, rax = plt.subplots()
        rax.errorbar(width_vals, ras_by_width, yerr=ra_std_by_width, fmt='o', color='g', ecolor='g', capsize=5)
        rax.set_title('Average RA vs Width')
    if decs_by_width:
        dfig, dax = plt.subplots()
        dax.errorbar(width_vals, decs_by_width, yerr=dec_std_by_width, fmt='o', color='r', ecolor='r', capsize=5)
        dax.set_title('Average Dec vs Width')
    if sigmas_by_width:
        sfig, sax = plt.subplots()
        sax.errorbar(width_vals_for_sigma, sigmas_by_width, yerr=sigma_std_by_width, fmt='o', color='c', ecolor='c', capsize=5)
        sax.set_title('Average Sigma vs Width')
    if points_by_width:
        ptsfig, ptsax = plt.subplots()
        ptsax.errorbar(width_vals, points_by_width, yerr=point_std_by_width, fmt='o', color='k', ecolor='k', capsize=5)
        ptsax.set_title('Average Points vs Width')
