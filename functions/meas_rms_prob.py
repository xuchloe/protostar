import numpy as np
import scipy.stats
from scipy.stats import norm
import incl_excl_data

def meas_rms_prob(fits_file: str, center: tuple = (float('inf'), float('inf'))):
    '''Given a FITS file and (optional) center coordinates in units of arcsec,
    return a dictionary with the probability of the peak to noise ratio of the interior of the specified circle
    and the probability of peak to noise ratio of the exterior of the specified circle,
    with noise being the measured rms in the exclusion area.
    '''

    info = incl_excl_data(fits_file, center)

    int_peak_val = info['int_peak_val']
    ext_peak_val = info['ext_peak_val']
    rms_val = info['rms_val']
    n_incl_meas = info['n_incl_meas']
    n_excl_meas = info['n_excl_meas']

    prob_dict = {}

    #calculate error for rms
    rms_err = rms_val * (n_excl_meas)**(-1/2)

    #create normal distributions from rms and error for rms
    uncert = np.linspace(-5 * rms_err, 5 * rms_err, 100)
    uncert_pdf = norm.pdf(uncert, loc = 0, scale = rms_err)

    #sum and normalize to find probabilities
    prob_dict['int_prob'] = float(sum((norm.cdf((-1 * int_peak_val)/(rms_val + uncert)) * n_incl_meas) * uncert_pdf) / sum(uncert_pdf))
    prob_dict['ext_prob'] = float(sum((norm.cdf((-1 * ext_peak_val)/(rms_val + uncert)) * n_excl_meas) * uncert_pdf) / sum(uncert_pdf))

    return prob_dict
