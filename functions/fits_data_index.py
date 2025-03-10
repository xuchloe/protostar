# %%
from astropy.io import fits
import numpy as np

def fits_data_index(fits_file: str):
    '''Given a FITS file, return the index of the file where the data array is'''

    file_index = 0

    #open FITS file
    try:
        file = fits.open(fits_file)
    except:
        print(f'Unable to open {fits_file}')

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
            print(f'Error in locating data index of {fits_file}')

    return file_index
