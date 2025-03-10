import meas_rms_prob
import calc_rms_prob

def significance(fits_file: str, version: str, threshold: float):
    '''Given a FITS file, version ('meas_rms_prob' or 'calc_rms_prob'), and threshold probability,
    return a Boolean of whether source emission is considered significant for the given threshold.
    '''
    if not (threshold >= 0 and threshold <= 1):
        raise ValueError("threshold must be between 0 and 1, inclusive")

    if not (version == 'meas_rms_prob' or version == 'calc_rms_prob'):
        raise ValueError("version must be either 'meas_rms_prob' or 'calc_rms_prob'")

    if version == 'meas_rms_prob':
        return meas_rms_prob(fits_file)['int_prob'] <= threshold

    else:
        return calc_rms_prob(fits_file)['int_prob'] <= threshold
