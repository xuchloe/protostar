import meas_rms_prob
import calc_rms_prob

def significance(fits_file: str, version: str, threshold: float):
    '''Given a FITS file, version ('meas_rms_prob' or 'calc_rms_prob'), and threshold probability,
    return a list of Booleans (if version = 'meas_rms_prob') or Boolean (if version = 'calc_rms_prob')
    of whether source emission is considered significant for the given threshold.
    Each entry of the list corresponds to the external probabilities using different regions.
    There will be more than one entry in the list if the first external probability is less than 0.001.
    '''
    if not (threshold >= 0 and threshold <= 1):
        raise ValueError("threshold must be between 0 and 1, inclusive")

    if not (version == 'meas_rms_prob' or version == 'calc_rms_prob'):
        raise ValueError("version must be either 'meas_rms_prob' or 'calc_rms_prob'")

    bool_list = []

    if version == 'meas_rms_prob':
        for i in range(len(meas_rms_prob(fits_file))):
            dict = meas_rms_prob(fits_file)[i]
            prob = dict['int_prob']
            bool_list.append(prob <= threshold)
        return bool_list

    else:
        dict = calc_rms_prob(fits_file)
        prob = dict['int_prob']
        bool = prob <= threshold
        return bool
