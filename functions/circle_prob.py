import numpy as np
from scipy.stats import norm
import region_stats

def circle_prob(fits_file: str, radius: float, center: tuple = (float('inf'), float('inf'))):
    '''Given a FITS file, radius in units of arcsec, and center coordinates in units of arcsec,
    return a dictionary with the probability of the peak to noise ratio of the interior of the specified circle
    and the probability of peak to noise ratio of the exterior of the specified circle.
    '''

    prob_dict = {}

    #get info on inclusion and exclusion regions
    int_info = region_stats(fits_file = fits_file, inclusion = radius, center = center)
    ext_info = region_stats(fits_file = fits_file, exclusion = radius, center = center)

    #getting values for peak, rms, axis lengths, beam size
    int_peak_val = int_info['peak']
    ext_peak_val = ext_info['peak']
    rms_val = ext_info['rms']
    x_axis = int_info['x_axis']
    y_axis = int_info['y_axis']
    beam_size = int_info['beam_size']

    #calculating number of measurements in inclusion and exclusion regions
    incl_area = np.pi * (radius**2)
    excl_area = x_axis * y_axis - incl_area
    n_incl_meas = incl_area / beam_size
    n_excl_meas = excl_area / beam_size

    #calculate error for rms
    rms_err = rms_val * (2 / (n_excl_meas - 1))**(1/4)

    #create normal distributions from rms and error for rms
    uncert = np.linspace(-5 * rms_err, 5 * rms_err, 100)
    uncert_pdf = norm.pdf(uncert, loc = 0, scale = rms_err)

    #sum and normalize to find probabilities
    prob_dict['int_prob'] = float(sum((norm.cdf((-1 * int_peak_val)/(rms_val + uncert)) * n_incl_meas) * uncert_pdf) / sum(uncert_pdf))
    prob_dict['ext_prob'] = float(sum((norm.cdf((-1 * ext_peak_val)/(rms_val + uncert)) * n_excl_meas) * uncert_pdf) / sum(uncert_pdf))

    return prob_dict
