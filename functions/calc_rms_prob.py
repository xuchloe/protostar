import scipy.stats
from scipy.stats import norm
import incl_excl_data

def calc_rms_prob(fits_file: str, center: tuple = (float('inf'), float('inf'))):
    '''Given a FITS file and (optional) center coordinates in units of arcsec,
    return a dictionary with the probability of the peak to noise ratio of the interior of the specified circle
    and the probability of peak to noise ratio of the exterior of the specified circle,
    with noise being the calculated rms in the exclusion area based on the expected probability of the peak value in the exclusion area.
    '''

    prob_dict = {}

    info = incl_excl_data(fits_file, center)

    int_peak_val = info['int_peak_val']
    ext_peak_val = info['ext_peak_val']
    n_incl_meas = info['n_incl_meas']
    n_excl_meas = info['n_excl_meas']

    excl_sigma = -1 * norm.ppf(1/n_excl_meas)
    rms_val = ext_peak_val / excl_sigma

    prob_dict['int_prob'] = float(norm.cdf((-1 * int_peak_val)/(rms_val))) * n_incl_meas
    prob_dict['ext_prob'] = float(norm.cdf((-1 * ext_peak_val)/(rms_val))) * n_excl_meas

    return(prob_dict)
