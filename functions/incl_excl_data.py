from astropy.io import fits
import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
import fits_data_index
import region_stats

def incl_excl_data(fits_file: str, center: tuple = (float('inf'), float('inf'))):
    '''Given a FITS file and (optional) center coordinates in units of arcsec,
    return a dictionary with the peak flux value of the inclusion area, peak flux value of the exclusion area, rms value of the exclusion area,
    number of measurements in the inclusion area, and number of measurements in the exclusion area of the specified circle.
    '''

    i = fits_data_index(fits_file)

    #open FITS file
    try:
        file = fits.open(fits_file)
    except:
        print(f'Unable to open {fits_file}')

    #extract data array
    info = file[i]

    radius = float((info.header['BMAJ'] * (Angle(1, info.header['CUNIT1'])).to(u.arcsec) / u.arcsec) + 5) #major axis + 5 arcsec

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

    return {'int_peak_val': int_peak_val, 'ext_peak_val': ext_peak_val, 'rms_val': rms_val, \
            'n_incl_meas': n_incl_meas, 'n_excl_meas': n_excl_meas}
