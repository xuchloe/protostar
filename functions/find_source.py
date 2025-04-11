from astropy.io import fits
import numpy as np
import scipy.stats
from scipy.stats import norm
from astropy.coordinates import Angle
import astropy.units as u
import matplotlib.pyplot as plt

def fits_data_index(fits_file: str):
    '''Given a FITS file, return the index of the file where the data array is'''

    file_index = 0

    #open FITS file
    try:
        file = fits.open(fits_file)
    except:
        print(f'Unable to open {fits_file}')

    info = file[file_index]
    data = info.data
    while data is None:
        #going through the indices of file to find the array
        try:
            file_index += 1
            info = file[file_index]
            data = info.data
        except:
            print(f'Error in locating data index of {fits_file}')

    return file_index

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
    if len(center) > 1:
        radius = radius + ([radius[0] - 5.0] * (len(center) - 1))

    #get info on inclusion and exclusion regions
    int_info = region_stats(fits_file = fits_file, radius = radius, center = center)
    ext_info = region_stats(fits_file = fits_file, radius = radius, center = center, invert=True)

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
    incl_area = np.pi * ((radius[0]**2) + ((radius[0] - 5.0)**2) * (len(center)- 1))
    excl_area = x_axis * y_axis - incl_area
    info_dict['n_incl_meas'] = incl_area / beam_size
    info_dict['n_excl_meas'] = excl_area / beam_size

    return info_dict

def meas_rms_prob(fits_file: str, center: list = [], rms: float = None, reps: bool = False, recursion: bool = True):
    '''Given a FITS file, (optional) list of tuples of center coordinates in units of arcsec,
    (optional) rms value, and (optional) choice to use recursion, return a list of dictionaries.
    The first dictionary contains the probability of the peak to noise ratio of the interior of the specified region
    and the probability of peak to noise ratio of the exterior of the specified region,
    with noise being the measured rms in the exclusion area.
    If the external probability is less than 0.001, there will be subsequent dictionaries
    that contain the probability of the peak to noise ratio of the interior of the specified region
    and the probability of peak to noise ratio of the exterior of the specified region,
    with noise being the measured rms in the exclusion area,
    but the included region is expanded to include the highest excluded peak of the previous excluded region
    and the excluded region now excludes that highest peak.
    '''
    info = incl_excl_data(fits_file, center)
    if rms is not None:
        info['rms_val'] = rms

    if reps: #keeping int_peak_val and int_peak coord in the original search area
        initial_info = incl_excl_data(fits_file, [center[0]])
        info['int_peak_val'] = initial_info['int_peak_val']
        info['int_peak_coord'] = initial_info['int_peak_coord']

    int_peak = info['int_peak_val']
    ext_peak = info['ext_peak_val']
    rms = info['rms_val']
    n_incl = info['n_incl_meas']
    n_excl = info['n_excl_meas']

    #calculate error for rms
    rms_err = rms * (n_excl)**(-1/2)

    #create normal distributions from rms and error for rms
    uncert = np.linspace(-5 * rms_err, 5 * rms_err, 100)
    uncert_pdf = norm.pdf(uncert, loc = 0, scale = rms_err)

    #sum and normalize to find probabilities
    prob_dict = info
    prob_dict['int_prob'] = float(sum((norm.cdf((-1 * int_peak)/(rms + uncert)) * n_incl) * uncert_pdf) / sum(uncert_pdf))
    prob_dict['ext_prob'] = float(sum((norm.cdf((-1 * ext_peak)/(rms + uncert)) * n_excl) * uncert_pdf) / sum(uncert_pdf))
    prob_dict['int_snr'] = float(int_peak / rms)
    prob_dict['ext_snr'] = float(ext_peak / rms)

    prob_list = [prob_dict]

    if prob_dict['ext_prob'] < 0.001 and recursion:
        reps = True
        if center == []:
            new_center = [info['field_center'], info['ext_peak_coord']]
        else:
            center.append(info['ext_peak_coord'])
            new_center = center
        new_list = meas_rms_prob(fits_file, new_center, rms = None, reps = reps, recursion = True)
        prob_list.extend(new_list)

    #using better rms value for calculating probability of peak when just looking in initial area
    elif len(prob_list) > 1:
        new_list = meas_rms_prob(fits_file, center = [prob_list[0]['field_center']], rms = prob_list[-1]['rms_val'], \
                                     reps = False, recursion = False)
        new_list.extend(prob_list[1:])
        prob_list = new_list

    return prob_list

def calc_rms_prob(prob_list: list):
    '''Given a list output from meas_rms_prob(), return the list output with an appended dictionary
    that contains the probability of the peak to noise ratio of the interior of the specified region
    and the probability of peak to noise ratio of the exterior of the specified region,
    calculated rms, calculated interior peak to noise ratio, and calculated exterior peak to noise ratio
    with noise being the calculated rms in the exclusion area based on the expected probability of the peak value in the exclusion area.
    '''
    info = prob_list[-1]

    int_peak_val = info['int_peak_val']
    ext_peak_val = info['ext_peak_val']
    n_incl_meas = info['n_incl_meas']
    n_excl_meas = info['n_excl_meas']

    excl_sigma = -1 * norm.ppf(1/n_excl_meas)
    rms_val = ext_peak_val / excl_sigma

    prob_dict= {}

    prob_dict['calc_rms_val'] = float(rms_val)
    prob_dict['calc_int_prob'] = float(norm.cdf((-1 * int_peak_val)/(rms_val))) * n_incl_meas
    prob_dict['calc_ext_prob'] = float(norm.cdf((-1 * ext_peak_val)/(rms_val))) * n_excl_meas
    prob_dict['calc_int_snr'] = float(int_peak_val / rms_val)
    prob_dict['calc_ext_snr'] = float(excl_sigma)

    prob_list.append(prob_dict)

    return prob_list

def summary(fits_file: str, info_list: bool = True, plot: bool = True):
    '''Given a FITS file, (optional) choice to return a list of information, and (optional) choice to show a plot,
    return a list of source information if requested and a plot of source information if requested.
    '''
    m_info = meas_rms_prob(fits_file)
    info = (calc_rms_prob(meas_rms_prob(fits_file)))

    if plot:
        image_data = fits.getdata(fits_file)
        shape = image_data.shape

        if len(shape) > 2:
            image_data = image_data[0]

        x_coords = [m_info[0]['int_peak_coord'][0]]
        y_coords = [m_info[0]['int_peak_coord'][1]]

        print(len(m_info))
        print(m_info)
        if len(m_info) > 1:
            for i in range(len(m_info)-1):
                x_coords.append(m_info[i]['ext_peak_coord'][0])
                y_coords.append(m_info[i]['ext_peak_coord'][1])

        plt.plot(x_coords, y_coords, 'o')
        plt.imshow(image_data)
        plt.colorbar()

    if info_list:
        return info
