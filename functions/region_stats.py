from astropy.io import fits
import numpy as np
from astropy.coordinates import Angle
import astropy.units as u

def region_stats(fits_file: str, exclusion: float = 0, inclusion: float = float('inf'), center: tuple = (float('inf'), float('inf'))):
    '''Given a FITS file, exclusion radius in units of arcsec (exclude area within this radius),
    inclusion radius in units of arcsec (include area within this radius),
    and center coordinates in units of arcsec,
    return a dictionary with floats of the maximum flux (in Jy), rms (in Jy), beam size (in arcsec^2),
    x axis length (in arcsec), and y axis length (in arcsec) in the specified region.
    If no exclusion radius given, default to 0.
    If no inclusion radius given, default to infinity.
    If no center given, will eventually default to center of ((length of x-axis)/2, (length of y-axis)/2), rounded up.
    '''

    file_index = 0

    #open FITS file
    try:
        file = fits.open(fits_file)
    except:
        print(f'Unable to open {fits_file}')

    #extract data array
    while True:
        #going through the indices of file to find the array
        try:
            info = file[file_index]
            data = info.data
            if isinstance(data, np.ndarray):
                break
            else:
                file_index += 1
        except:
            print(f'Error in extracting data from {fits_file}')

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
    if center == (float('inf'), float('inf')):
        center_pix = (round(x_dim/2), round(y_dim/2))

    #find units of axes
    x_unit = info.header['CUNIT1']
    y_unit = info.header['CUNIT2']

    #find cell size (units of arcsec)
    x_cell_size = (Angle(info.header['CDELT1'], x_unit)).to(u.arcsec)
    y_cell_size = (Angle(info.header['CDELT2'], y_unit)).to(u.arcsec)
    y_cell_size.to(u.arcsec)

    #find beam size (units of arcsec^2)
    beam_size = (info.header['BEAMSIZE'] * Angle(1, x_unit) * Angle(1, y_unit)).to(u.arcsec**2)

    #find axis sizes
    x_axis_size = info.header['NAXIS1'] * x_cell_size
    y_axis_size = info.header['NAXIS2'] * y_cell_size

    #distance from center array
    dist_from_center =((((x_dist_array - center_pix[0])*x_cell_size)**2 + \
                        ((y_dist_array - center_pix[1])*y_cell_size)**2)**0.5) #array of each pixel's distance from center_pix

    #boolean mask and apply
    mask = (dist_from_center >= exclusion * u.arcsec) & (dist_from_center <= inclusion * u.arcsec)
    masked_data = data[0][mask]

    #get peak, rms, beam_size values
    try:
        peak = float(max(masked_data))
    except ValueError:
        print('No values after mask applied. Check inclusion and exclusion radii.')

    rms = float((np.var(masked_data))**0.5)

    stats = {'peak': peak, 'rms': rms, 'beam_size': float(beam_size / (u.arcsec**2)),\
              'x_axis': float(x_axis_size / u.arcsec), 'y_axis': float(y_axis_size / u.arcsec)}

    return stats
