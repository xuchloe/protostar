import numpy as np
import scipy.stats
from scipy.stats import norm
import incl_excl_data

def meas_rms_prob(fits_file: str, center: list = []):
    '''Given a FITS file, (optional) list of tuples of center coordinates in units of arcsec,
    and (optional) maximum number of repetitions, return a list of dictionaries.
    The first dictionary contains the probability of the peak to noise ratio of the interior of the specified region
    and the probability of peak to noise ratio of the exterior of the specified region,
    with noise being the measured rms in the exclusion area.
    If the external probability is less than 0.001, there will be subsequent dictionaries
    that contain the probability of the peak to noise ratio of the interior of the specified region
    and the probability of peak to noise ratio of the exterior of the specified region,
    with noise being the measured rms in the exclusion area,
    but the included region is expanded to include the highest excluded peak of the previous excluded region
    and the excluded region now excludes that highest peak.
    '''
    info = incl_excl_data(fits_file, center)

    int_peak = info['int_peak_val']
    ext_peak = info['ext_peak_val']
    rms = info['rms_val']
    n_incl = info['n_incl_meas']
    n_excl = info['n_excl_meas']

    #calculate error for rms
    rms_err = rms * (n_excl)**(-1/2)

    #create normal distributions from rms and error for rms
    uncert = np.linspace(-5 * rms_err, 5 * rms_err, 100)
    uncert_pdf = norm.pdf(uncert, loc = 0, scale = rms_err)

    #sum and normalize to find probabilities
    prob_dict = info
    prob_dict['int_prob'] = float(sum((norm.cdf((-1 * int_peak)/(rms + uncert)) * n_incl) * uncert_pdf) / sum(uncert_pdf))
    prob_dict['ext_prob'] = float(sum((norm.cdf((-1 * ext_peak)/(rms + uncert)) * n_excl) * uncert_pdf) / sum(uncert_pdf))
    prob_dict['int_snr'] = float(int_peak / rms)
    prob_dict['ext_snr'] = float(ext_peak / rms)

    prob_list = [prob_dict]

    if prob_dict['ext_prob'] < 0.001:
        if center == []:
            new_center = [info['field_center'], info['ext_peak_coord']]
        else:
            center.append(info['ext_peak_coord'])
            new_center = center
        new_list = meas_rms_prob(fits_file, new_center)
        prob_list.extend(new_list)

    return prob_list
