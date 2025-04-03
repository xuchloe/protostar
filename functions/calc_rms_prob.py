import scipy.stats
from scipy.stats import norm
import incl_excl_data

def calc_rms_prob(fits_file: str, center: list = []):
    '''Given a FITS file, (optional) center coordinates in units of arcsec, and (optional) maximum number of repetitions
    return a dictionary that contains the probability of the peak to noise ratio of the interior of the specified region
    and the probability of peak to noise ratio of the exterior of the specified region,
    with noise being the calculated rms in the exclusion area based on the expected probability of the peak value in the exclusion area.
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

    return prob_dict
