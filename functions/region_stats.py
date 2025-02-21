
from astropy.io import fits
import numpy as np

def region_stats(fits_file: str, exclusion: float = 0, inclusion: float = float('inf')):
    '''Given a FITS file, exclusion radius (exclude pixels within this radius),
    and inclusion radius (include pixels within this radius),
    return a dictionary with the maximum flux and rms in the specified region.
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
            data = file[file_index].data
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

    x_dist_array = np.tile(np.arange(x_dim),(y_dim, 1)) #array of each pixel's horizontal distance from y-axis
    y_dist_array = x_dist_array.T #array of each pixel's vertical distance from x-axis
    center_pix = (x_dim // 2, y_dim // 2)
    dist_from_center =(((x_dist_array - center_pix[0])**2 + \
                        (y_dist_array - center_pix[1])**2)**0.5) #array of each pixel's distance from center_pix

    mask = (dist_from_center >= exclusion) & (dist_from_center <= inclusion)
    masked_data = data[0][mask]

    try:
        peak = float(max(masked_data))
    except ValueError:
        print('No values after mask applied. Check inclusion and exclusion radii.')

    rms = float(np.var(masked_data))

    stats = {'peak': peak, 'rms': rms}

    return stats
