from astropy.io import fits
import numpy as np
import scipy.stats
from scipy.stats import norm
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import glob


def fits_data_index(fits_file: str):
    '''
    Finds the location of a FITS file's data array.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file to be searched.

    Returns
    -------
    int
        The index of the data array in the FITS file.
    '''

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
    '''
    Finds the statistics of a region of an image.

    The region can be the union of circles or the complement of such a region.

    The statistics are the region's maximum flux in Jy and its coordinates in pixels, the region's rms in Jy,
    the coordinates in pixels of the image's center, the image's beam size in arcseconds squared,
    the image's x- and y-axis lengths in arcseconds, and the area included in the mask in arcseconds squared.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    center : list (optional)
        A list of center coordinates in units of pixels.
        If no center coordinates are given, eventually defaults to ((length of x-axis)/2, (length of y-axis)/2), rounded up.
    radius : list (optional)
        A list of search radii in units of arcsec.
        If no radius list is given, defaults to an empty list.
    invert : bool (optional)
        Whether to swap the inclusion and exclusion regions.
        If no value is given, defaults to False.

    Returns
    -------
    dict
        A dictionary with:
            float
                The region's maximum flux in Jy.
            tuple (int, int)
                The coordinates in pixels of the region's maximum flux.
            float
                The region's rms in Jy.
            tuple (int, int)
                The coordinates in pixels of the image's center.
            float
                The image's beam size in arcseconds squared.
            float
                The image's x-axis length in arcsec.
            float
                The image's y-axis length in arcsec.
            float
                The area included in the mask in arcseconds squared.

    Raises
    ------
    IndexError
        If center list and radius list are of different lengths.
    '''

    if center != [] and len(center) != len(radius):
        raise IndexError ('Center list and radius list are of different lengths')

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

    incl_area = float(mask.sum() * x_cell_size * y_cell_size / (u.arcsec)**2)

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
              'x_axis': float(x_axis_size / u.arcsec), 'y_axis': float(y_axis_size / u.arcsec), 'incl_area': incl_area}

    return stats


def incl_excl_data(fits_file: str, center: list = [], radius_buffer: float = 5.0):
    '''
    Finds statistics of an inclusion region and its complement, the exclusion region.

    The inclusion region can be the union of circles or the complement of such a region.

    The statistics are the inclusion region's maximum flux in Jy and its coordinates in pixels,
    the exclusion region's maximum flux in Jy and its coordinates in pixels, the exclusion region's rms in Jy,
    the number of measurements in the inclusion region, the number of measurements in the exclusion region,
    the coordinates in pixels of the image's center, and the radii in pixels of the inclusion zones.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    center : list (optional)
        A list of center coordinates in units of pixels.
        If no center coordinates are given, eventually defaults to ((length of x-axis)/2, (length of y-axis)/2), rounded up.
    radius_buffer : float (optional)
        The amount of buffer, in arcsec, to add to the beam FWHM to get the initial search radius.
        If no value is given, defaults to 5 arcsec.

    Returns
    -------
    dict
        A dictionary with:
            float
                The inclusion region's maximum flux in Jy.
            tuple (int, int)
                The coordinates in pixels of the inclusion region's maximum flux.
            float
                The exclusion region's maximum flux in Jy.
            tuple (int, int)
                The coordinates in pixels of the exclusion region's maximum flux.
            float
                The exclusion region's rms in Jy.
            float
                The number of measurements in the inclusion region.
            float
                The number of measurements in the exclusion region.
            tuple (int, int)
                The coordinates in pixels of the image's center.
            list
                A list with:
                    float(s)
                        The radii in pixels of inclusion zones.
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
    beam_fwhm = float((info.header['BMAJ'] * (Angle(1, info.header['CUNIT1'])).to(u.arcsec) / u.arcsec)) #in arcsec
    radius = [beam_fwhm + radius_buffer]
    if len(center) > 1:
        radius = radius + ([beam_fwhm] * (len(center) - 1))

    #get info on inclusion and exclusion regions
    int_info = region_stats(fits_file=fits_file, radius=radius, center=center)
    ext_info = region_stats(fits_file=fits_file, radius=radius, center=center, invert=True)

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
    incl_area = int_info['incl_area']
    excl_area = ext_info['incl_area']
    info_dict['n_incl_meas'] = incl_area / beam_size
    info_dict['n_excl_meas'] = excl_area / beam_size

    pix_radius = [] #list of radii in pixels
    for r in range(len(radius)):
        pix_rad = (Angle(radius[r], u.arcsec).to(info.header['CUNIT1']) / info.header['CDELT1']) / info.header['CUNIT1']
        pix_radius.append(float(pix_rad))
    info_dict['radius'] = pix_radius

    return info_dict


def get_prob_image_rms(fits_file: str, center: list = [], rms: float = None, recursion: bool = True,\
                       radius_buffer: float = 5.0, ext_threshold: float = 0.001):
    '''
    Using the exclusion region's rms taken directly from the image,
    finds the probability of detecting the inclusion region's maximum flux if there were no source in the inclusion region,
    the probability of detecting the exclusion region's maximum flux if there were no source in the exclusion region, and other statistics.

    If the external probability is less than 0.001, updates the inclusion region to include a circle around the external peak.

    The other statisitcs are the inclusion region's maximum flux in Jy and its coordinates in pixels,
    the exclusion region's maximum flux in Jy and its coordinates in pixels, the exclusion region's rms in Jy,
    the number of measurements in the inclusion region, the number of measurements in the exclusion region,
    the coordinates in pixels of the image's center, and the radii in pixels of the inclusion zones,
    the inclusion region's signal to noise ratio, and the external region's signal to noise ratio.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    center : list (optional)
        A list of center coordinates in units of pixels.
        If no center coordinates are given, eventually defaults to ((length of x-axis)/2, (length of y-axis)/2), rounded up.
    radius_buffer : float (optional)
        The amount of buffer, in arcsec, to add to the beam FWHM to get the initial search radius.
        If no value is given, defaults to 5 arcsec.
    ext_threshold : float (optional)
        The probability that an external peak must be below for it to be considered an external source.
        If no value is given, defaults to 0.001.
    rms : float (optional)
        An rms value in Jy.
        If no value is given, eventually defaults to the rms calculated by incl_excl_data.
    recursion : bool (optional)
        Whether to use recursion to find significant external peaks, if any.
        If no value is given, defaults to True.

    Returns
    -------
    list
        A list with:
            dict (possibly multiple)
                A dictionary with:
                    float
                        The probability of detecting the inclusion region's maximum flux if there were no source in the inclusion region.
                    float
                        The probability of detecting the exclusion region's maximum flux if there were no source in the exclusion region.
                    float
                        The inclusion region's maximum flux in Jy.
                    tuple (int, int)
                        The coordinates in pixels of the inclusion region's maximum flux.
                    float
                        The exclusion region's maximum flux in Jy.
                    tuple (int, int)
                        The coordinates in pixels of the exclusion region's maximum flux.
                    float
                        The exclusion region's rms in Jy.
                    float
                        The number of measurements in the inclusion region.
                    float
                        The number of measurements in the exclusion region.
                    tuple (int, int)
                        The coordinates in pixels of the image's center.
                    list
                        A list with:
                            float(s)
                                The radii in pixels of inclusion zones.
                    float
                        The inclusion region's signal to noise ratio.
                    float
                        The exclusion region's signal to noise ratio.
    '''
    info = incl_excl_data(fits_file, center, radius_buffer)
    if rms is not None:
        info['rms_val'] = rms

    #keeping int_peak_val and int_peak coord in the original search area
    initial_info = incl_excl_data(fits_file, [], radius_buffer)
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

    if prob_dict['ext_prob'] < ext_threshold and recursion:
        if center == []:
            new_center = [info['field_center'], info['ext_peak_coord']]
        else:
            center.append(info['ext_peak_coord'])
            new_center = center
        new_list = get_prob_image_rms(fits_file, new_center, rms=None, recursion=True, \
                                      radius_buffer=radius_buffer, ext_threshold=ext_threshold)
        prob_list.extend(new_list)

    #using better rms value for calculating probability of peak when just looking in initial area
    elif len(prob_list) > 1:
        new_list = get_prob_image_rms(fits_file, center=[prob_list[0]['field_center']], rms=prob_list[-1]['rms_val'], \
                                     recursion=False, radius_buffer=radius_buffer, ext_threshold=ext_threshold)
        new_list.extend(prob_list[1:])
        prob_list = new_list

    return prob_list


def get_prob_rms_est_from_ext(prob_list: list):
    '''
    Using the rms estimated from the value of the exclusion region's maximum flux,
    finds the probability of detecting the inclusion region's maximum flux if there were no source in the inclusion region,
    the probability of detecting the exclusion region's maximum flux if there were no source in the exclusion region, and other statistics.

    The estimated rms is that the probability of finding such an external peak,
    assuming no source in the exclusion region, is 1.
    Note: this implies that the external probability will always be 1.

    The other statistics include the following as calculated using the rms estimated as described above:
    the exclusion region's rms in Jy, the inclusion region's signal to noise ratio,
    and the external region's signal to noise ratio.

    The remaining statisitcs include the following as calculated using the rms taken directly from the image:
    the inclusion region's maximum flux in Jy and its coordinates in pixels,
    the exclusion region's maximum flux in Jy and its coordinates in pixels, the exclusion region's rms in Jy,
    the number of measurements in the inclusion region, the number of measurements in the exclusion region,
    the coordinates in pixels of the image's center, and the radii in pixels of the inclusion zones,
    the inclusion region's signal to noise ratio, and the external region's signal to noise ratio.

    Parameters
    ----------
    prob_list : list
        The list of statistics, as outputted by get_prob_image_rms(), for an image.

    Returns
    -------
    list
        A list with:
            dict(s)
                A dictionary with the following, found using the rms taken directly from the image:
                    float
                        The probability of detecting the inclusion region's maximum flux if there were no source in the inclusion region.
                    float
                        The probability of detecting the exclusion region's maximum flux if there were no source in the exclusion region.
                    float
                        The inclusion region's maximum flux in Jy.
                    tuple (int, int)
                        The coordinates in pixels of the inclusion region's maximum flux.
                    float
                        The exclusion region's maximum flux in Jy.
                    tuple (int, int)
                        The coordinates in pixels of the exclusion region's maximum flux.
                    float
                        The exclusion region's rms in Jy.
                    float
                        The number of measurements in the inclusion region.
                    float
                        The number of measurements in the exclusion region.
                    tuple (int, int)
                        The coordinates in pixels of the image's center.
                    list
                        A list with:
                            float(s)
                                The radii in pixels of inclusion zones.
                    float
                        The inclusion region's signal to noise ratio.
                    float
                        The exclusion region's signal to noise ratio.
            dict
                A dictionary with the following, found using the rms estimated as described above:
                    float
                        The probability of detecting the inclusion region's maximum flux if there were no source in the inclusion region.
                    float
                        The probability of detecting the exclusion region's maximum flux if there were no source in the exclusion region.
                    float
                        The exclusion region's rms in Jy.
                    float
                        The inclusion region's signal to noise ratio.
                    float
                        The exclusion region's signal to noise ratio.
    '''
    info = prob_list[-1]

    int_peak_val = info['int_peak_val']
    ext_peak_val = info['ext_peak_val']
    n_incl_meas = info['n_incl_meas']
    n_excl_meas = info['n_excl_meas']

    excl_sigma = -1 * norm.ppf(1/n_excl_meas)
    rms_val = ext_peak_val / excl_sigma

    prob_dict = {}

    prob_dict['calc_rms_val'] = float(rms_val)
    prob_dict['calc_int_prob'] = float(norm.cdf((-1 * int_peak_val)/(rms_val))) * n_incl_meas
    prob_dict['calc_ext_prob'] = float(norm.cdf((-1 * ext_peak_val)/(rms_val))) * n_excl_meas
    prob_dict['calc_int_snr'] = float(int_peak_val / rms_val)
    prob_dict['calc_ext_snr'] = float(excl_sigma)

    prob_list.append(prob_dict)

    return prob_list


def summary(fits_file: str, radius_buffer: float = 5.0, ext_threshold: float = 0.001,\
            short_dict: bool = True, full_list: bool = False, plot: bool = True, save_path: str = ''):
    '''
    Summarizes an image's statistics into a shorter dictionary, a more detailed dictionary, and/or a plot,
    with an option to save the plot as a png.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    radius_buffer : float (optional)
        The amount of buffer, in arcsec, to add to the beam FWHM to get the initial search radius.
        If no value is given, defaults to 5 arcsec.
    ext_threshold : float (optional)
        The probability that an external peak must be below for it to be considered an external source.
        If no value is given, defaults to 0.001.
    short_dict : bool (optional)
        Whether to return the short dictionary of statistics.
        If no value is given, defaults to True.
    full_list : bool (optional)
        Whether to return the more detailed list of statistics.
        If no value is given, defaults to False.
    plot : bool (optional)
        Whether to plot the image and statistics.
        If no value is given, defaults to True.
    save_path : str (optional)
        The path to which the plot will be saved.
        If no value is given, defaults to '' and no image is saved.

    Returns
    -------
    dict (if requested)
        A shorter dictionary with:
            float
                The probability, found using the rms taken directly from the image,
                of detecting the inclusion region's maximum flux if there were no source in the inclusion region.
            list
                A list with:
                    float(s)
                        The probabilities, found using the rms taken directly from the image,
                        of detecting the exclusion regions' maximum flux if there were no source in the exclusion regions.
                        If there are multiple entries in this list,
                        they are the probabilities as the exclusion region becomes increasingly small
                        as external peaks deemed significant are added to the inclusion region.
            float
                The inclusion region's maximum flux in Jy.
            tuple (float, float)
                The coordinates in relative arcsec of the inclusion region's maximum flux.
            list
                A list of with:
                    float(s)
                        The exclusion regions' maximum fluxes in Jy.
                        If there are multiple entries in this list,
                        they are the maxmimum fluxes as the exclusion region becomes increasingly small
                        as external peaks deemed significant are added to the inclusion region.
            list
                A list with:
                    tuple(s) (float, float)
                        The coordinates in relative arcsec of the exclusion regions' maximum fluxes.
                        If there are multiple entires in this list,
                        they are the coordinates as the exclusion region becomes increasingly small
                        as external peaks deemed significant are added to the inclusion region.
            float
                The exclusion region's rms in Jy. This uses the final (smallest) exclusion region.
            float
                The number of measurements in the inclusion region.
            float
                The number of measurements in the exclusion region.
            tuple (int, int)
                The coordinates in relative arcsec of the image's center. Should be (0, 0).
            list
                A list with:
                    float(s):
                        The radii in arcsec of inclusion zones.
            float
                The inclusion region's signal to noise ratio.
            list
                A list with:
                    float(s)
                        The exclusion regions' signal to noise ratios.
            float
                The probability, found using the rms estimated from the value of the exclusion region's maximum flux,
                of detecting the inclusion region's maximum flux if there were no source in the inclusion region.
            float
                The probability, found using the rms estimated from the value of the exclusion region's maximum flux,
                of detecting the exclusion region's maximum flux if there were no source in the exclusion region.
            float
                The rms in Jy estimated from the value of the exclusion region's maximum flux.
            float
                The inclusion region's signal to noise ratio,
                found using the rms estimated from the value of the exclusion region's maximum flux.
            float
                The exclusion region's signal to noise ratio,
                found using the rms estimated from the value of the exclusion region's maximum flux.
    list (if requested)
        A more detailed list with:
            dict(s)
                A dictionary with the following, found using the rms taken directly from the image:
                    float
                        The probability of detecting the inclusion region's maximum flux if there were no source in the inclusion region.
                    float
                        The probability of detecting the exclusion region's maximum flux if there were no source in the exclusion region.
                    float
                        The inclusion region's maximum flux in Jy.
                    tuple (float, float)
                        The coordinates in relative arcsec of the inclusion region's maximum flux.
                    float
                        The exclusion region's maximum flux in Jy.
                    tuple (float, float)
                        The coordinates in relative arcsec of the exclusion region's maximum flux.
                    float
                        The exclusion region's rms in Jy.
                    float
                        The number of measurements in the inclusion region.
                    float
                        The number of measurements in the exclusion region.
                    tuple (float, float)
                        The coordinates in relative arcsec of the image's center. Should be (0.0, 0.0).
                    list
                        A list with:
                            float(s)
                                The radii in arcsec of inclusion zones.
                    float
                        The inclusion region's signal to noise ratio.
                    float
                        The exclusion region's signal to noise ratio.
            dict
                A dictionary with the following, found using the rms estimated as described above:
                    float
                        The probability of detecting the inclusion region's maximum flux if there were no source in the inclusion region.
                    float
                        The probability of detecting the exclusion region's maximum flux if there were no source in the exclusion region.
                    float
                        The exclusion region's rms in Jy.
                    float
                        The inclusion region's signal to noise ratio.
                    float
                        The exclusion region's signal to noise ratio.
    '''
    m_info = get_prob_image_rms(fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold)

    info = (get_prob_rms_est_from_ext(get_prob_image_rms(fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold)))

    center = m_info[0]['field_center']

    header_data = fits.getheader(fits_file)
    pixel_scale = Angle(header_data['CDELT1'], header_data['CUNIT1']).to_value('arcsec')

    int_x_coord = np.array([m_info[0]['int_peak_coord'][0]])
    int_y_coord = np.array([m_info[0]['int_peak_coord'][1]])

    #normalize internal peak coordinates
    int_x_coord = (int_x_coord - center[0]) * pixel_scale
    int_y_coord = (int_y_coord - center[1]) * pixel_scale

    int_radius = m_info[0]['radius'][0] #in pixels

    if len(m_info) > 1:
        x_coords = []
        y_coords = []

        for i in range(len(m_info)-1):
            #normalized external peak coordinates
            x_coords.append((m_info[i]['ext_peak_coord'][0] - center[0]) * pixel_scale)
            y_coords.append((m_info[i]['ext_peak_coord'][1] - center[1]) * pixel_scale)
        ext_radius = m_info[-1]['radius'][1]

        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)

    if plot:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 24
        plt.rcParams['hatch.linewidth'] = 0.5

        image_data = fits.getdata(fits_file)
        shape = image_data.shape

        while len(shape) > 2:
            image_data = image_data[0]
            shape = image_data.shape

        plt.set_cmap('inferno')
        fig, ax = plt.subplots(figsize=(10,10))

        plt.plot(int_x_coord, int_y_coord, 'wo', fillstyle='none', markersize=15)
        plt.plot(int_x_coord, int_y_coord, 'kx', fillstyle='none', markersize=15/np.sqrt(2))

        int_circle = patches.Circle((0, 0), int_radius * pixel_scale, edgecolor='c', fill=False)
        ax.add_artist(int_circle)

        if len(m_info) > 1:
            plt.plot(x_coords, y_coords, 'ko', fillstyle='none', markersize=15)
            plt.plot(x_coords, y_coords, 'wx', fillstyle='none', markersize=15/np.sqrt(2))

            for i in range(len(x_coords)):
                ext_circle = patches.Circle((x_coords[i], y_coords[i]), ext_radius * pixel_scale, edgecolor='lime', fill=False)
                ax.add_artist(ext_circle)
        int_snr = m_info[-1]['int_snr']

        x_min = ((0 - center[0]) - 0.5) * pixel_scale
        y_min = ((0 - center[1]) - 0.5) * pixel_scale
        x_max = ((image_data.shape[0] -  center[0]) - 0.5) * pixel_scale
        y_max = ((image_data.shape[1] -  center[1]) - 0.5) * pixel_scale

        beam = patches.Ellipse((x_min*0.92, y_min*0.92), Angle(header_data['BMIN'], header_data['CUNIT1']).to_value('arcsec'),\
                               Angle(header_data['BMAJ'], header_data['CUNIT1']).to_value('arcsec'), fill=True, facecolor='w',\
                                edgecolor='k', angle=header_data['BPA'], hatch='/////', lw=1)
        ax.add_artist(beam)

        title = fits_file[fits_file.rindex('/')+1:fits_file.index('.fits')]
        ax.text(x_min*0.96, y_max*0.96, f'Source: {title}\nInternal Candidate SNR: {int_snr:.2f}', horizontalalignment='left', verticalalignment='top',\
                fontsize=24, bbox=dict(facecolor='w'))

        plt.imshow(image_data, extent=[x_min, x_max, y_min, y_max], origin='lower')

        plt.xlabel('Relative RA Offset [arcsec]', fontsize=32)
        plt.ylabel('Relative Dec Offset [arcsec]', fontsize=32)

        jy_to_mjy = lambda x, pos: '{}'.format(round(x*1000, 1))
        fmt = ticker.FuncFormatter(jy_to_mjy)

        cbar = plt.colorbar(shrink=0.8, format=fmt)
        cbar.ax.set_ylabel('Intensity [mJy/beam]', fontsize=32, rotation=270, labelpad=36)

        if save_path != '':
            try:
                file = fits_file
                while '/' in file:
                    file = file[file.index('/')+1:]
                file = file.replace('.fits', '')
                if save_path[-1] != '/':
                    save_path = save_path + '/'
                plt.savefig(f'{save_path}{file}.pdf')
            except:
                print('Error saving figure. Double check path entered.')

    ext_peaks = 'No significant external peak'
    ext_vals = 'No significant external peak'
    ext_snrs = 'No significant external peak'
    ext_probs = 'No significant external peak'

    if len(m_info) > 1:
        ext_peaks = []
        ext_vals = []
        ext_snrs = []
        ext_probs = []
    for i in range(len(m_info)-1):
        ext_peaks.append((float(x_coords[i]), float(y_coords[i])))
        ext_vals.append(m_info[i]['ext_peak_val'])
        ext_snrs.append(m_info[i]['ext_snr'])
        ext_probs.append(m_info[i]['ext_prob'])

    #convert radii from pixels to arcsec
    new_rad = []
    for j in range(len(m_info[-1]['radius'])):
        new_rad.append(float(m_info[-1]['radius'][j] * pixel_scale))

    short_info = {'int_peak_val': m_info[-1]['int_peak_val'], 'int_peak_coord': (float(int_x_coord[0]), float(int_y_coord[0])), 'int_snr': m_info[-1]['int_snr'],\
                  'calc_int_snr': info[-1]['calc_int_snr'], 'int_prob': m_info[-1]['int_prob'], 'calc_int_prob': info[-1]['calc_int_prob'],\
                  'ext_peak_val': ext_vals, 'ext_peak_coord': ext_peaks, 'ext_snr': ext_snrs,\
                  'calc_ext_snr': info[-1]['calc_ext_snr'], 'ext_prob': ext_probs, 'calc_ext_prob': info[-1]['calc_ext_prob'],\
                  'field_center': (0,0), 'rms': m_info[-1]['rms_val'], 'calc_rms_val': info[-1]['calc_rms_val'],\
                  'n_incl_meas': m_info[-1]['n_incl_meas'], 'n_excl_meas': m_info[-1]['n_excl_meas'], 'radius': new_rad}

    #normalizing coordinates in the full list
    if full_list:
        for d in info:
            for key, value in d.items():
                #convert coordinates from pixels to relative arcsec
                if type(value) == tuple:
                    new_coords = (float((value[0] - center[0]) * pixel_scale), float((value[1] - center[1]) * pixel_scale))
                    d[key] = new_coords

                #convert radii from pixels to arcsec
                new_radius = []
                if key == 'radius':
                    for k in range(len(value)):
                        new_radius.append(float(value[k] * pixel_scale))
                    d[key] = new_radius

    center = (0,0) #normalizing center coordinates

    if short_dict and full_list:
        return short_info, info

    elif full_list:
        return info

    elif short_dict:
        return short_info

    else:
        return


def significant(fits_file: str, threshold: float = 0.01, radius_buffer: float = 5.0, ext_threshold: float = 0.001):
    '''
    Finds whether a significant source was detected in a field's center region.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    threshold : float (optional)
        The threshold for a significant detection.
        If the probability of detecting the center region's maximum flux assuming no source in the image
        is less than this threshold, then the detection is deemed significant.
        If no value is given, defaults to 0.01.
    radius_buffer : float (optional)
        The amount of buffer, in arcsec, to add to the beam FWHM to get the initial search radius.
        If no value is given, defaults to 5 arcsec.
    ext_threshold : float (optional)
        The probability that an external peak must be below for it to be considered an external source.
        If no value is given, defaults to 0.001.

    Returns
    -------
    bool : Whether a significant source was detected in the field's center region.

    Raises
    ------
    ValueError
        If threshold is not between 0 and 1, inclusive.
    '''

    #make sure reasonable input
    if not (threshold >= 0 and threshold <= 1):
        raise ValueError('Threshold must be between 0 and 1, inclusive.')

    summ = summary(fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold, short_dict=True, full_list=False, plot=False)
    return (summ['int_prob'] < threshold and summ['calc_int_prob'] < threshold)


def make_catalog(fits_file: str, radius_buffer: float = 5.0, ext_threshold: float = 0.001):
    '''
    Summarizes information on any significant point sources detected in an image.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    radius_buffer : float (optional)
        The amount of buffer, in arcsec, to add to the beam FWHM to get the initial search radius.
        If no value is given, defaults to 5 arcsec.
    ext_threshold : float (optional)
        The probability that an external peak must be below for it to be considered an external source.
        If no value is given, defaults to 0.001.

    Returns
    -------
    dict
        A dictionary with:
            dict(s)
                A dictionary with:
                    str
                        The name of the target object of the observation.
                    str
                        The date and time of the observation.
                    str
                        The name of the FITS file with the image.
                    Angle
                        The restoring beam major axis.
                    Angle
                        The restoring beam minor axis.
                    Angle
                        The restoring beam position angle.
                    float
                        The uncertainty in flux density measurements. The rms excluding any significant sources and a small circular region around them.
                    float
                        The flux density of the detected point source.
                    SkyCoord
                        The location of the detected point source.
                    bool
                        Whether the detected point source is in the initial search region.
    '''

    summ = summary(fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold, short_dict=True, full_list=False, plot=False)

    header_data = fits.getheader(fits_file)
    name = header_data['OBJECT']
    obs_date_time = header_data['DATE-OBS']
    bmaj = header_data['BMAJ']
    bmin = header_data['BMIN']
    bpa = header_data['BPA']
    ctype1 = header_data['CTYPE1']
    crval1 = header_data['CRVAL1']
    cunit1 = header_data['CUNIT1']
    ctype2 = header_data['CTYPE2']
    crval2 = header_data['CRVAL2']
    cunit2 = header_data['CUNIT2']

    #assume beam axes in same units as CUNIT1 and CUNIT2 and BPA in degrees
    beam_maj_axis = Angle(bmaj, cunit1)
    beam_min_axis = Angle(bmin, cunit1)
    beam_pos_angle = Angle(bpa, u.degree)

    interesting_sources = {}
    field_info = {'field_name': name, 'obs_date_time': obs_date_time, 'file_name': fits_file[fits_file.rindex('/')+1:],\
                           'beam_maj_axis': beam_maj_axis, 'beam_min_axis': beam_min_axis, 'beam_pos_angle': beam_pos_angle,\
                           'flux_uncertainty': summ['rms']}

    n_ext_sources = 0
    if type(summ['ext_peak_val']) == list:
        n_ext_sources += len(summ['ext_peak_val'])

    ra_index = 0
    dec_index = 1

    if 'RA' in ctype1:
        ra = crval1
    elif 'RA' in ctype2:
        ra = crval2
        ra_index = 1
    else:
        raise ValueError('No RA in image')

    if 'DEC' in ctype1:
        dec = crval1
        dec_index = 0
    elif 'DEC' in ctype2:
        dec = crval2
    else:
        raise ValueError('No dec in image')

    if cunit1 != cunit2:
        raise ValueError('Axes have different units')

    center = SkyCoord(ra, dec, unit=cunit1)

    pt_source_count = 1

    if significant(fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold):
        int_info = field_info.copy()
        int_info['flux_density'] = summ['int_peak_val']

        int_ra_offset = summ['int_peak_coord'][ra_index] * u.arcsec
        int_dec_offset = summ['int_peak_coord'][dec_index] * u.arcsec
        int_info['coord'] = center.spherical_offsets_by(int_ra_offset, int_dec_offset)

        int_info['internal'] = True

        key = f'source_{pt_source_count}'
        interesting_sources[key] = int_info
        pt_source_count +=1

    for i in range(n_ext_sources):
        ext_info = field_info.copy()
        ext_info['flux_density'] = summ['ext_peak_val'][i]

        ext_ra_offset = summ['ext_peak_coord'][i][ra_index] * u.arcsec
        ext_dec_offset = summ['ext_peak_coord'][i][dec_index] * u.arcsec
        ext_info['coord'] = center.spherical_offsets_by(ext_ra_offset, ext_dec_offset)

        ext_info['internal'] = False

        key = f'source_{pt_source_count}'
        interesting_sources[key] = ext_info
        pt_source_count += 1

    if 'source_1' not in interesting_sources:
        return
    else:
        return interesting_sources


def combine_catalogs(catalog_1: dict, catalog_2: dict):
    '''
    Combines two catalogs in the format returned by make_catalog() into a single catalog of the same format.

    Parameters
    ----------
    catalog_1 : dict
        The catalog to which the other catalog will be "appended."
    catalog_2 : dict
        The catalog to "append" to the other catalog.

    Returns
    -------
    dict
        A dictionary of the combined catalogs in the same catalog format.
    '''

    shift = len(catalog_1)
    for key, value in catalog_2.items():
        new_number = int(key.replace('source_', ''))
        new_key = f'source_{new_number + shift}'
        catalog_1[new_key] = value
    return catalog_1
