import scipy.stats
import region_stats

def circle_cdf(fits_file: str, radius: float, center: tuple = (float('inf'), float('inf'))):
    '''Given a FITS file, radius in units of arcsec, and center coordinates in units of arcsec,
    return a dictionary with the probability of the peak to noise ratio of the interior of the specified circle
    and the probability of peak to noise ratio of the exterior of the specified circle.
    '''

    prob_dict = {}

    int_info = region_stats(fits_file = fits_file, inclusion = radius, center = center)
    ext_info = region_stats(fits_file = fits_file, exclusion = radius, center = center)

    int_peak_val = int_info['peak']
    ext_peak_val = ext_info['peak']
    rms_val = ext_info['rms']

    int_sigma = int_peak_val / rms_val
    ext_sigma = ext_peak_val / rms_val

    prob_dict['int_prob'] = float(scipy.stats.norm.cdf(-1 * int_sigma))
    prob_dict['ext_prob'] = float(scipy.stats.norm.cdf(-1 * ext_sigma))
    return prob_dict
