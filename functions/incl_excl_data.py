from astropy.io import fits
import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
import fits_data_index
import region_stats

def incl_excl_data(fits_file: str, center: list = []):
    '''Given a FITS file and (optional) list of tuples of center coordinates in units of arcsec,
    return a dictionary with the field center coordinates as tuple, peak flux value of the inclusion area, coordinates of this peak as tuple,
    peak flux value of the exclusion area, coordinates of this peak as tuple, rms value of the exclusion area,
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

    #get radius, inclusion, exclusion lists for interior and exterior
    radius = [float((info.header['BMAJ'] * (Angle(1, info.header['CUNIT1'])).to(u.arcsec) / u.arcsec) + 5)] #major axis + 5 arcsec
    int_excl = [0]
    ext_incl = [float('inf')]
    if len(center) > 1:
        radius = radius * len(center)
        int_excl = int_excl * len(center)
        ext_incl = ext_incl * len(center)

    #get info on inclusion and exclusion regions
    int_info = region_stats(fits_file = fits_file, inclusion = radius, exclusion = int_excl, center = center)
    ext_info = region_stats(fits_file = fits_file, inclusion = ext_incl, exclusion = radius, center = center)

    #getting values for peak, rms, axis lengths, beam size
    info_dict = {}
    info_dict['int_peak_val'] = int_info['peak']
    info_dict['field_center'] = int_info['field_center']
    info_dict['int_peak_coord'] = int_info['peak_coord']
    info_dict['ext_peak_coord'] = ext_info['peak_coord']
    info_dict['ext_peak_val'] = ext_info['peak']
    info_dict['rms_val'] = ext_info['rms']
    x_axis = int_info['x_axis']
    y_axis = int_info['y_axis']
    beam_size = int_info['beam_size']

    #calculating number of measurements in inclusion and exclusion regions
    incl_area = np.pi * (radius[0]**2)
    excl_area = x_axis * y_axis - incl_area
    info_dict['n_incl_meas'] = incl_area / beam_size
    info_dict['n_excl_meas'] = excl_area / beam_size

    return info_dict
