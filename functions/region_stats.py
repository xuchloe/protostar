from astropy.io import fits
import numpy as np
from astropy.coordinates import Angle
import astropy.units as u
import fits_data_index

def region_stats(fits_file: str, center: list = [], radius: list = [], invert: bool = False):
    '''Given a FITS file, list of center coordinates in units of pixels,
    list of radii in units of arcsec (include measurements within this radius of the center),
    and Boolean of whether to invert (if True, becomes exclude instead of include),
    return a dictionary with floats of the maximum flux (in Jy), coordinates of field center (in pixels),
    coordinates of maximum flux (in pixels), rms (in Jy), beam size (in arcsec^2),
    x axis length (in arcsec), and y axis length (in arcsec) in the specified region.
    If no center given, will eventually default to center of ((length of x-axis)/2, (length of y-axis)/2), rounded up.
    '''

    if center != [] and len(center) != len(radius):
        raise IndexError ('Center list and radius list lengths do not match')

    i = fits_data_index(fits_file)

    #open FITS file
    try:
        file = fits.open(fits_file)
    except:
        print(f'Unable to open {fits_file}')

    #extract data array
    info = file[i]
    data = info.data

    #getting dimensions for array
    try:
        dims = data.shape
        x_dim = dims[1]
        y_dim = dims[2]
    except:
        print('Data dimension error')

    x_dist_array = np.tile(np.arange(x_dim),(y_dim, 1)) #array of each pixel's horizontal distance (in pixels) from y-axis
    y_dist_array = x_dist_array.T #array of each pixel's vertical distance (in pixels) from x-axis

    #keep center pixel coordinates if specified, set to default if unspecified
    center_pix = center
    field_center = (round(x_dim/2), round(y_dim/2))
    if center == []:
        center_pix = [field_center]
        if len(radius) > 1:
            center_pix = center_pix * len(radius)

    #find units of axes
    x_unit = info.header['CUNIT1']
    y_unit = info.header['CUNIT2']

    #find cell size (units of arcsec)
    x_cell_size = (Angle(info.header['CDELT1'], x_unit)).to(u.arcsec)
    y_cell_size = (Angle(info.header['CDELT2'], y_unit)).to(u.arcsec)
    y_cell_size.to(u.arcsec)

    #find major axis (units of arcsec), minor axis (units of arcsec), beam size (units of arcsec^2)
    beam_size = ((np.pi/4) * info.header['BMAJ'] * info.header['BMIN'] * Angle(1, x_unit) * Angle(1, y_unit) / np.log(2)).to(u.arcsec**2)

    #find axis sizes
    x_axis_size = info.header['NAXIS1'] * x_cell_size
    y_axis_size = info.header['NAXIS2'] * y_cell_size

    #distance from center array
    dist_from_center =((((x_dist_array - center_pix[0][0])*x_cell_size)**2 + ((y_dist_array - center_pix[0][1])*y_cell_size)**2)**0.5)

    #boolean mask and apply
    mask = (dist_from_center <= radius[0] * u.arcsec)
    if len(center) > 1:
        for j in range(1, len(center)):
            dist_from_center = ((((x_dist_array - center_pix[j][0])*x_cell_size)**2 + ((y_dist_array - center_pix[j][1])*y_cell_size)**2)**0.5)
            mask = np.logical_or(mask, (dist_from_center <= radius[j] * u.arcsec))

    if invert:
        mask = np.logical_not(mask)

    masked_data = data[0][mask]

    #get peak, rms, beam_size values
    try:
        peak = float(max(masked_data))
    except ValueError:
        print('No values after mask applied. Check inclusion and exclusion radii.')

    #find coordinates of peak
    peak_pix = np.where(data[0] == peak)
    x = peak_pix[1][0]
    y = peak_pix[0][0]
    peak_coord = (int(x_dist_array[0][x]), int(y_dist_array[y][0]))

    rms = float((np.var(masked_data))**0.5)

    stats = {'peak': peak, 'field_center': field_center, 'peak_coord': peak_coord, 'rms': rms, 'beam_size': float(beam_size / (u.arcsec**2)),\
              'x_axis': float(x_axis_size / u.arcsec), 'y_axis': float(y_axis_size / u.arcsec)}

    return stats
