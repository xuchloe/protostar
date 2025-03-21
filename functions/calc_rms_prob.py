import scipy.stats
from scipy.stats import norm
import incl_excl_data

def calc_rms_prob(fits_file: str, center: list = [], max_reps: int = 2):
    '''Given a FITS file, (optional) center coordinates in units of arcsec, and (optional) maximum number of repetitions
    return a list of dictionaries.
    The first dictionary contains the probability of the peak to noise ratio of the interior of the specified region
    and the probability of peak to noise ratio of the exterior of the specified region,
    with noise being the calculated rms in the exclusion area based on the expected probability of the peak value in the exclusion area.
    The second dictionary contains the probability of the peak to noise ratio of the interior of the specified region
    and the probability of peak to noise ratio of the exterior of the specified region,
    with noise being the calculated rms in the exclusion area based on the expected probability of the peak value in the exclusion area,
    but the included region is expanded to include the highest excluded peak of the previous excluded region
    and the excluded region now excludes that highest peak.
    '''

    info = incl_excl_data(fits_file, center)
    prob_dict = info

    int_peak_val = info['int_peak_val']
    ext_peak_val = info['ext_peak_val']
    n_incl_meas = info['n_incl_meas']
    n_excl_meas = info['n_excl_meas']

    excl_sigma = -1 * norm.ppf(1/n_excl_meas)
    rms_val = ext_peak_val / excl_sigma

    prob_dict['rms_val'] = float(rms_val)
    prob_dict['int_prob'] = float(norm.cdf((-1 * int_peak_val)/(rms_val))) * n_incl_meas
    prob_dict['ext_prob'] = float(norm.cdf((-1 * ext_peak_val)/(rms_val))) * n_excl_meas
    prob_dict['int_snr'] = float(int_peak_val / rms_val)
    prob_dict['ext_snr'] = float(excl_sigma)

    prob_list = [prob_dict]

    if max_reps > 1 and prob_dict['ext_prob'] < 0.001:
        n = max_reps - 1
        new_list = calc_rms_prob(fits_file, center = [info['field_center'], info['ext_peak_coord']], max_reps = n)
        prob_list.extend(new_list)

    return prob_list
