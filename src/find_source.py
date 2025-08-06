from astropy.io import fits
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import norm, median_abs_deviation
from astropy.coordinates import Angle
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
import math


def fits_data_index(fits_file: str):
    '''
    Finds the location of a FITS file's image data array.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file to be searched.

    Returns
    -------
    int
        The index of the image data array in the FITS file.
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


def gaussian_theta(coord, amp, sigma, theta, mu_x, mu_y):
    '''
    Finds the value at a point on a 2D Gaussian.

    Parameters
    ----------
    coord : tuple
        The coordinate(s) of the point(s) where the first entry is the x-coordinate or a list of x-coordinates
        and the second entry is the y-coordinate or a list of y-coordinates.
    amp : float
        The factor in front of the 2D Gaussian's exponent.
    sigma : float
        The standard deviation of the 2D Gaussian.
    theta : float
        The angle of rotation of the 2D Gaussian.
    mu_x : float
        The x-value of the peak of the 2D Gaussian.
    mu_y : float
        The y-value of the peak of the 2D Gaussian.

    Returns
    -------
    float
        The value of the 2D Gaussian evaluated at the given point.
    '''

    x, y = coord
    return amp * np.exp(-(((x-mu_x)*math.cos(theta)+(y-mu_y)*math.sin(theta))**2+(-(x-mu_x)*math.sin(theta)+(y-mu_y)*math.cos(theta))**2)/(2*sigma**2))


def region_stats(fits_file: str, center: list = [], radius: list = [], invert: bool = False, Gaussian: bool = True, internal: bool = True,\
                 outer_radius: float = None):
    '''
    Finds the statistics of a region of an image.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    center : list (optional)
        A list of center coordinates in units of pixels.
        If no center coordinates are given, eventually defaults to [((length of x-axis)/2, (length of y-axis)/2)], rounded up.
    radius : list (optional)
        A list of search radii in units of arcsec.
        If no radius list is given, defaults to an empty list.
    invert : bool (optional)
        Whether to swap the inclusion and exclusion regions.
        If no value is given, defaults to False.
    Gaussian : bool (optional)
        Whether to use a 2D Gaussian fit to estimate the true maximum flux and its corresponding coordinates.
        If no value is given, defaults to True.
    internal : bool (optional)
        Whether the peak to search for is internal (in which case to use a 5x5 pixel region if using a Gaussian fit)
        or external (in which case to use a 3x3 pixel region if using a Gaussian fit).
        If no value is given, defaults to True.
    outer radius : float (optional)
        The radius outside of which everything will be excluded. This is not affected by value invert.
        If no value is given, defaults to None and will not be used to exclude data.

    Returns
    -------
    dict
        A dictionary with:
            float
                The region's maximum flux in Jy.
            tuple (int, int)
                The coordinates in pixels of the image's center.
            tuple (int, int)
                The coordinates in pixels of the region's maximum flux.
            float
                The region's rms in Jy.
            float
                The image's beam size in arcseconds squared.
            float
                The image's x-axis length in arcsec.
            float
                The image's y-axis length in arcsec.
            float
                The area included in the mask in arcseconds squared.
            float
                The area excluded by the mask in arcseconds squared.
            float
                The number of measurements included in the mask.
            float
                The number of measurements excluded by the mask.
            float
                The median absolute deviation of the flux of the image.
            float
                The standard deviation of the flux of the image, as estimated by the MAD.
            float
                The most negative flux in the image, if such a flux exists. If not, this is None.

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

    neg_peak = float(np.min(data[0]))
    if neg_peak >= 0:
        neg_peak = None

    mad = float(median_abs_deviation(data[0].flatten()))
    sd_mad = float(norm.ppf(0.84) / norm.ppf(0.75) * mad) #estimate standard deviation from MAD

    #getting dimensions for array
    x_dim = info.header['NAXIS1']
    y_dim = info.header['NAXIS2']

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

    #find beam size (unitless but in arcsec^2)
    beam_size = float(((np.pi/4) * info.header['BMAJ'] * info.header['BMIN'] * Angle(1, x_unit) * Angle(1, y_unit) / np.log(2)).to(u.arcsec**2)\
                / (u.arcsec**2))

    #find axis sizes
    x_axis_size = x_dim * x_cell_size
    y_axis_size = y_dim * y_cell_size

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

    if outer_radius is not None:
        dist_from_field_center = ((((x_dist_array - field_center[0])*x_cell_size)**2 + ((y_dist_array - field_center[1])*y_cell_size)**2)**0.5)
        outer_mask = (dist_from_field_center <= outer_radius * u.arcsec)
        mask = np.logical_and(mask, outer_mask)

    incl_area = float(mask.sum() * x_cell_size * y_cell_size / (u.arcsec)**2)
    excl_area = float(np.logical_not(mask).sum() * x_cell_size * y_cell_size / (u.arcsec)**2)

    masked_data = data[0][mask]

    #get peak
    try:
        peak = float(max(masked_data))
    except ValueError:
        print('No values after mask applied. Check inclusion and exclusion radii.')

    #find coordinates of peak
    peak_pix = np.where(data[0] == peak)
    peak_x = int(peak_pix[1][0])
    peak_y = int(peak_pix[0][0])
    peak_coord = (peak_x, peak_y)

    #fit for peak and coordinates assuming Gaussian
    #use data from 5x5 region if internal peak
    if Gaussian and internal and (peak_x - 2) >= 0 and (peak_x + 2) <= x_dim and (peak_y - 2) >= 0 and (peak_y + 2) <= y_dim:
        neg2_2 = data[0][peak_x - 2][peak_y + 2]
        neg2_1 = data[0][peak_x - 2][peak_y + 1]
        neg2_0 = data[0][peak_x - 2][peak_y]
        neg2_neg1 = data[0][peak_x - 2][peak_y - 1]
        neg2_neg2 = data[0][peak_x - 2][peak_y - 2]
        neg1_2 = data[0][peak_x - 1][peak_y + 2]
        neg1_1 = data[0][peak_x - 1][peak_y + 1]
        neg1_0 = data[0][peak_x - 1][peak_y]
        neg1_neg1 = data[0][peak_x - 1][peak_y - 1]
        neg1_neg2 = data[0][peak_x - 1][peak_y - 2]
        zero_2 = data[0][peak_x][peak_y + 2]
        zero_1 = data[0][peak_x][peak_y + 1]
        zero_neg1 = data[0][peak_x][peak_y - 1]
        zero_neg2 = data[0][peak_x][peak_y - 2]
        pos1_2 = data[0][peak_x + 1][peak_y + 2]
        pos1_1 = data[0][peak_x + 1][peak_y + 1]
        pos1_0 = data[0][peak_x + 1][peak_y]
        pos1_neg1 = data[0][peak_x + 1][peak_y - 1]
        pos1_neg2 = data[0][peak_x + 1][peak_y - 2]
        pos2_2 = data[0][peak_x + 2][peak_y + 2]
        pos2_1 = data[0][peak_x + 2][peak_y + 1]
        pos2_0 = data[0][peak_x + 2][peak_y]
        pos2_neg1 = data[0][peak_x + 2][peak_y - 1]
        pos2_neg2 = data[0][peak_x + 2][peak_y - 2]

        z_data = [neg2_2, neg2_1, neg2_0, neg2_neg1, neg2_neg2,\
                neg1_2, neg1_1, neg1_0, neg1_neg1, neg1_neg2,\
                zero_2, zero_1, peak, zero_neg1, zero_neg2,\
                pos1_2, pos1_1, pos1_0, pos1_neg1, pos1_neg2,\
                pos2_2, pos2_1, pos2_0, pos2_neg1, pos2_neg2]
        x_data = [-2]*5 + [-1]*5 + [0]*5 + [1]*5 + [2]*5
        y_data = [2, 1, 0, -1, -2]*5

        try:
            popt, pcov = curve_fit(gaussian_theta, (x_data, y_data), z_data, bounds=([peak,0,0,-1,-1],[float('inf'),float('inf'),2*np.pi,1,1]))
            amp, sigma, theta, mu_x, mu_y = popt
            peak = float(amp)
            peak_coord = (float(peak_x + mu_x), float(peak_y + mu_y))
        except RuntimeError:
            pass

    #use data from 3x3 region if external peak
    elif Gaussian and (not internal) and (peak_x - 1) >= 0 and (peak_x + 1) <= x_dim and (peak_y - 1) >= 0 and (peak_y + 1) <= y_dim:
        left_top = data[0][peak_x - 1][peak_y + 1]
        left_middle = data[0][peak_x - 1][peak_y]
        left_bottom = data[0][peak_x - 1][peak_y - 1]
        middle_top = data[0][peak_x][peak_y + 1]
        middle_bottom = data[0][peak_x][peak_y - 1]
        right_top = data[0][peak_x + 1][peak_y + 1]
        right_middle = data[0][peak_x + 1][peak_y]
        right_bottom = data[0][peak_x + 1][peak_y - 1]

        z_data = [left_top, left_middle, left_bottom, middle_top, peak, middle_bottom, right_top, right_middle, right_bottom]
        x_data = [-1]*3 + [0]*3 + [1]*3
        y_data = [1, 0, -1] * 3

        try:
            popt, pcov = curve_fit(gaussian_theta, (x_data, y_data), z_data, bounds=([peak,0,0,-1,-1],[float('inf'),float('inf'),2*np.pi,1,1]))
            amp, sigma, theta, mu_x, mu_y = popt
            peak = float(amp)
            peak_coord = (float(peak_x + mu_x), float(peak_y + mu_y))
        except RuntimeError:
            pass

    rms = float((np.var(masked_data))**0.5)

    stats = {'peak': peak, 'field_center': field_center, 'peak_coord': peak_coord, 'rms': rms, 'beam_size': beam_size,\
             'x_axis': float(x_axis_size / u.arcsec), 'y_axis': float(y_axis_size / u.arcsec), 'incl_area': incl_area, 'excl_area': excl_area,\
             'n_incl_meas': float(incl_area / beam_size), 'n_excl_meas': float(excl_area / beam_size), 'mad': mad, 'sd_mad': sd_mad,\
             'neg_peak': neg_peak}

    return stats


def calc_prob_from_rms_uncert(peak: float, rms: float, n_excl: float, n_incl: float = None):
    '''
    Estimates the probability of a value or greater occurring in some number of measurements
    of a Gaussian distribution with an imprecisely known RMS.

    Parameters
    ----------
    peak : float
        The smallest value in the range of values whose probability of occurring will be estimated.
    rms : float
        The imprecisely known RMS value.
    n_excl : float
        The number of measurements in the region from which the RMS is measured.
        If no value is given for n_incl, this is also the number of measurements
        for which the probability will be estimated.
    n_incl : float (optional)
        The number of measurements for which the probability will be estimated.

    Returns
    -------
    float
        The estimated probability.
    '''

    #calculate error for rms
    rms_err = rms * (n_excl)**(-1/2)

    #create normal distributions from rms and error for rms
    uncert = np.linspace(-5 * rms_err, 5 * rms_err, 100)
    uncert_pdf = norm.pdf(uncert, loc = 0, scale = rms_err)

    #sum and normalize to find probabilities
    if n_incl == None:
        return float(sum((norm.cdf((-1 * peak)/(rms + uncert)) * n_excl) * uncert_pdf) / sum(uncert_pdf))
    else:
        return float(sum((norm.cdf((-1 * peak)/(rms + uncert)) * n_incl) * uncert_pdf) / sum(uncert_pdf))


def prob_dict_from_rms_uncert(fits_file: str, center: list = [], threshold: float = 0.01, radius_buffer: float = 5.0,\
                              ext_threshold: float = None):
    '''
    Finds the probabilities of the internal and external peaks, as well as other relevant statistics of an image.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    center : list (optional)
        A list of center coordinates in units of pixels.
        If no center coordinates are given, first defaults to [((length of x-axis)/2, (length of y-axis)/2)], rounded up.
    threshold : float (optional)
        The maximum probability, assuming no source in the image, for a significant internal detection.
        If no value is given, defaults to 0.01.
    radius_buffer : float (optional)
        The amount of buffer, in arcsec, to add to the beam FWHM to get the initial search radius.
        If no value is given, defaults to 5 arcsec.
    ext_threshold : float (optional)
        The probability that an external peak must be below for it to be considered an external source.
        If no value is given, defaults to 1e-3, 1e-6, or 1e-12, depending on the SNR of the internal peak.

    Returns
    -------
    dict
        A dictionary with:
            tuple (int, int)
                The coordinates in pixels of the image's center.
            float
                The image's rms in Jy.
            float
                The median absolute deviation of the flux of the image.
            float
                The standard deviation of the flux of the image, as estimated by the MAD.
            float
                The number of measurements included in the mask.
            float
                The number of measurements excluded by the mask.
            float
                The length of the beam major axis in arcsec.
            float
                The radius of the initial inclusion region in arcsec.
            float
                The most negative flux in the image, if such a flux exists. If not, this is None.
            list
                A list with:
                    float(s)
                        The flux of the brightest internal peak and the fluxes of the remaining significant internal peaks,
                        if these exist.
            list
                A list with:
                    tuple(s) (int, int)
                        The coordinates in pixels of the brightest internal peak and the remaining significant internal peaks,
                        if these exist.
            list
                A list with:
                    float(s)
                        The probability/probabilities of the brightest internal peak and the remaining significant internal peaks,
                        if these exist.
            list
                A list with:
                    float(s)
                        The signal to noise ratios of the brightest internal peak and the remaining significant internal peaks,
                        if these exist.
            list
                A list with:
                    float(s)
                        The flux(es) the significant external peak(s), if these exist.

ext_peak_coord: []
ext_prob: []
ext_snr: []
next_ext_peak: 0.11062327027320862
    '''

    i = fits_data_index(fits_file)

    #open FITS file
    try:
        file = fits.open(fits_file)
    except:
        print(f'Unable to open {fits_file}')

    #extract data array
    info = file[i]

    beam_fwhm = float((info.header['BMAJ'] * (Angle(1, info.header['CUNIT1'])).to(u.arcsec) / u.arcsec)) #unitless but in arcsec
    search_radius = beam_fwhm + radius_buffer #unitless but in arcsec

    #search for brightest internal peak
    int_stats1 = region_stats(fits_file=fits_file, center=center, radius=[search_radius], invert=False, Gaussian=False, internal=True)
    int_coord1 = int_stats1['peak_coord']
    int_peak1 = int_stats1['peak']
    n_incl = int_stats1['n_incl_meas'] #should be the same for all internal peaks
    field_center = int_stats1['field_center'] #in pixels
    mad = int_stats1['mad'] #should be the same for all peaks
    sd_mad = int_stats1['sd_mad'] #should be the same for all peaks
    neg_peak = int_stats1['neg_peak']

    #find external peaks and get their info
    center = [field_center]
    radius = [search_radius]

    ext_stats1 = region_stats(fits_file=fits_file, center=center, radius=radius, invert=True, Gaussian=False, internal=False)
    n_excl = ext_stats1['n_incl_meas'] #should be the same for all external peaks
    ext_peak1 = ext_stats1['peak']
    rms = ext_stats1['rms'] #can be changed later as we exclude more peaks
    ext_prob1 = calc_prob_from_rms_uncert(peak=ext_peak1, rms=rms, n_excl=n_excl)

    prob_dict = {'field_center': field_center, 'rms_val': None, 'mad': mad, 'sd_mad': sd_mad, 'n_incl_meas': n_incl, 'n_excl_meas': n_excl,\
                 'fwhm': beam_fwhm, 'incl_radius': search_radius, 'neg_peak': neg_peak,\
                 'int_peak_val': [], 'int_peak_coord': [], 'int_prob': [], 'int_snr': [],\
                 'ext_peak_val': [], 'ext_peak_coord': [], 'ext_prob': [], 'ext_snr': [], 'next_ext_peak': None}

    #update ext_threshold if needed
    int_snr1 = int_peak1 / rms
    if ext_threshold == None:
        if int_snr1 < 20:
            ext_threshold = 1e-3
        elif int_snr1 < 100:
            ext_threshold = 1e-6
        else:
            ext_threshold = 1e-12

    if ext_prob1 < ext_threshold:
        ext_significant = True
    else:
        prob_dict['next_ext_peak'] = ext_peak1
        ext_significant = False

    while ext_significant:
        ext_stats = region_stats(fits_file=fits_file, center=center, radius=radius, invert=True, Gaussian=False, internal=False)
        peak = ext_stats['peak']
        rms = ext_stats['rms']

        ext_prob = calc_prob_from_rms_uncert(peak=peak, rms=rms, n_excl=n_excl)
        if ext_prob < ext_threshold:
            ext_stats = region_stats(fits_file=fits_file, center=center, radius=radius, invert=True, Gaussian=True, internal=False)
            coord = ext_stats['peak_coord']
            peak = ext_stats['peak']
            ext_prob = calc_prob_from_rms_uncert(peak=peak, rms=rms, n_excl=n_excl)
            prob_dict['ext_peak_val'].append(peak)
            prob_dict['ext_peak_coord'].append(coord)
            prob_dict['ext_prob'].append(ext_prob)
            prob_dict['ext_snr'].append(peak / rms)
            center.append(coord)
            radius.append(beam_fwhm)
        else:
            prob_dict['next_ext_peak'] = peak
            ext_significant = False

    prob_dict['rms_val'] = rms

    #find prob for 1st internal peak using updated rms
    prob_dict['int_peak_val'].append(int_peak1)
    prob_dict['int_peak_coord'].append(int_coord1)
    int_prob1 = calc_prob_from_rms_uncert(peak=int_peak1, rms=rms, n_excl=n_excl, n_incl=n_incl)
    prob_dict['int_prob'].append(int_prob1)
    prob_dict['int_snr'].append(int_peak1 / rms)

    if threshold == None:
        threshold = 0.01
    int_significant = (int_prob1 < threshold)

    #treat 1st internal peak kind of like an external peak and get rid of search radius so we can look inside
    center = [int_coord1]
    radius = [beam_fwhm]

    #find internal peaks in addition to 1st internal peak
    while int_significant:
        int_stats = region_stats(fits_file=fits_file, center=center, radius=radius, invert=True, Gaussian=False, internal=True,\
                                 outer_radius=search_radius)
        int_peak = int_stats['peak']
        int_prob = calc_prob_from_rms_uncert(peak=int_peak, rms=rms, n_excl=n_excl, n_incl=n_incl)
        if int_prob < threshold and (int_peak > int_snr1 / 100):
            int_stats = region_stats(fits_file=fits_file, center=center, radius=radius, invert=True, Gaussian=True, internal=True,\
                                     outer_radius=search_radius)
            int_coord = int_stats['peak_coord']
            int_peak = int_stats['peak']
            int_prob = calc_prob_from_rms_uncert(peak=int_peak, rms=rms, n_excl=n_excl, n_incl=n_incl)
            prob_dict['int_peak_val'].append(int_peak)
            prob_dict['int_peak_coord'].append(int_coord)
            prob_dict['int_prob'].append(int_prob)
            prob_dict['int_snr'].append(int_peak / rms)
            center.append(int_coord)
            radius.append(beam_fwhm)
        else:
            int_significant = False

    return prob_dict


def get_prob_rms_est_from_ext(prob_dict: dict):
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
    int_peak_val = prob_dict['int_peak_val']
    ext_peak_val = prob_dict['next_ext_peak']
    n_incl_meas = prob_dict['n_incl_meas']
    n_excl_meas = prob_dict['n_excl_meas']

    excl_sigma = -1 * norm.ppf(1/n_excl_meas)
    old_rms_val = ext_peak_val / excl_sigma
    prob_dict['calc_rms_val'] = float(old_rms_val)

    sigma = norm.ppf(1/(n_incl_meas + n_excl_meas))
    neg_peak = prob_dict['neg_peak']

    if neg_peak is not None:
        rms_val = neg_peak / sigma
        prob_dict['neg_peak_rms_val'] = float(rms_val)
    else:
        prob_dict['neg_peak_rms_val'] = None
        rms_val = old_rms_val

    prob_dict['calc_ext_prob'] = float(norm.cdf((-1 * ext_peak_val)/(rms_val))) * n_excl_meas
    prob_dict['calc_ext_snr'] = float(excl_sigma)
    for i in range(len(int_peak_val)):
        if i == 0:
            prob_dict['calc_int_prob'] = [float(norm.cdf((-1 * int_peak_val[i])/(rms_val))) * n_incl_meas]
            prob_dict['calc_int_snr'] = [float(int_peak_val[i] / rms_val)]
        else:
            prob_dict['calc_int_prob'].append(float(norm.cdf((-1 * int_peak_val[i])/(rms_val))) * n_incl_meas)
            prob_dict['calc_int_snr'].append(float(int_peak_val[i] / rms_val))

    return prob_dict


def summary(fits_file: str, threshold: float = 0.01, radius_buffer: float = 5.0, ext_threshold: float = None,\
            short_dict: bool = True, plot: bool = True, save_path: str = ''):
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
    info = (get_prob_rms_est_from_ext(prob_dict_from_rms_uncert(fits_file=fits_file, threshold=threshold, radius_buffer=radius_buffer,\
                                                                ext_threshold=ext_threshold)))

    center = info['field_center']

    header_data = fits.getheader(fits_file)
    pixel_scale = Angle(header_data['CDELT1'], header_data['CUNIT1']).to_value('arcsec')

    int_x_coords = []
    int_y_coords = []
    int_peak_coords = info['int_peak_coord']
    n_int_peaks = len(int_peak_coords)
    for i in range(n_int_peaks):
        #normalized internal peak coordinates
        int_x_coords.append((int_peak_coords[i][0] - center[0]) * pixel_scale)
        int_y_coords.append((int_peak_coords[i][1] - center[1]) * pixel_scale)
    int_x_coords = np.array(int_x_coords)
    int_y_coords = np.array(int_y_coords)

    incl_radius = info['incl_radius'] #unitless but in arcsec already

    # get most conservative rms and internal snr
    hdul = fits.open(fits_file)
    noise = None
    try:
        noise_col = hdul[1].columns[2]
        if noise_col.name == 'Noise Est':
            if noise_col.unit == 'mJy':
                noise = float(hdul[1].data[0][2] * 1e3) # into Jy
            elif noise_col.unit == 'Jy':
                noise = float(hdul[1].data[0][2])
    except:
        pass
    rms_list = [info['rms_val'], info['sd_mad'], info['calc_rms_val'], info['neg_peak_rms_val']] # all in Jy
    if info['neg_peak_rms_val'] is not None:
        rms_list.append(info['neg_peak_rms_val'])
    if noise is not None:
        rms_list.append(noise)
    conservative_rms = max(rms_list) # in Jy
    conservative_snr = round(info['int_peak_val'][0] / conservative_rms, 3)

    x_coords = []
    y_coords = []
    ext_peak_coords = info['ext_peak_coord']
    n_ext_peaks = len(ext_peak_coords)
    for i in range(n_ext_peaks):
        #normalized external peak coordinates
        x_coords.append((ext_peak_coords[i][0] - center[0]) * pixel_scale)
        y_coords.append((ext_peak_coords[i][1] - center[1]) * pixel_scale)

    fwhm = info['fwhm']

    if plot:
        #plt.rcParams['font.family'] = 'serif'
        #plt.rcParams['font.serif'] = ['Times New Roman']
        plt.rcParams['font.size'] = 15
        plt.rcParams['hatch.linewidth'] = 0.5
        plt.rcParams['figure.dpi'] = 60

        image_data = fits.getdata(fits_file)
        shape = image_data.shape

        while len(shape) > 2:
            image_data = image_data[0]
            shape = image_data.shape

        plt.set_cmap('inferno')
        fig, ax = plt.subplots(figsize=(6.7,5.1))

        plt.plot(int_x_coords, int_y_coords, 'wo', fillstyle='none', markersize=15)
        plt.plot(int_x_coords, int_y_coords, 'kx', fillstyle='none', markersize=15/np.sqrt(2))

        for i in range(n_int_peaks):
            int_circle = patches.Circle((int_x_coords[i], int_y_coords[i]), fwhm * pixel_scale, edgecolor='lime', fill=False)
            ax.add_artist(int_circle)

        int_circle = patches.Circle((0, 0), incl_radius, edgecolor='c', fill=False)
        ax.add_artist(int_circle)

        if n_ext_peaks > 0:
            x_coords = np.array(x_coords)
            y_coords = np.array(y_coords)
            plt.plot(x_coords, y_coords, 'ko', fillstyle='none', markersize=15)
            plt.plot(x_coords, y_coords, 'wx', fillstyle='none', markersize=15/np.sqrt(2))

            for i in range(n_ext_peaks):
                ext_circle = patches.Circle((x_coords[i], y_coords[i]), fwhm * pixel_scale, edgecolor='lime', fill=False)
                ax.add_artist(ext_circle)

        int_snr = info['int_snr'][0]

        x_min = ((0 - center[0]) - 0.5) * pixel_scale
        y_min = ((0 - center[1]) - 0.5) * pixel_scale
        x_max = ((image_data.shape[0] -  center[0]) - 0.5) * pixel_scale
        y_max = ((image_data.shape[1] -  center[1]) - 0.5) * pixel_scale

        beam = patches.Ellipse((x_min*0.88, y_min*0.92), Angle(header_data['BMIN'], header_data['CUNIT1']).to_value('arcsec'),\
                               Angle(header_data['BMAJ'], header_data['CUNIT1']).to_value('arcsec'), fill=True, facecolor='w',\
                                edgecolor='k', angle=header_data['BPA'], hatch='/////', lw=1)
        ax.add_artist(beam)

        title = fits_file[fits_file.rindex('/')+1:fits_file.index('.fits')]
        ax.text(x_min*0.96, y_max*0.96, f'Source: {title}\nInternal Candidate SNR: {conservative_snr}', horizontalalignment='left', verticalalignment='top',\
                fontsize=10, bbox=dict(facecolor='w'))

        plt.imshow(image_data, extent=[x_min, x_max, y_min, y_max], origin='lower')

        plt.xlabel('Relative RA Offset [arcsec]', fontsize=15)
        plt.ylabel('Relative Dec Offset [arcsec]', fontsize=15)

        jy_to_mjy = lambda x, pos: '{}'.format(round(x*1000, 1))
        fmt = ticker.FuncFormatter(jy_to_mjy)

        cbar = plt.colorbar(shrink=0.8, format=fmt)
        cbar.ax.set_ylabel('Intensity [mJy/beam]', fontsize=15, rotation=270, labelpad=24)

        if save_path != '':
            try:
                file = fits_file
                while '/' in file:
                    file = file[file.index('/')+1:]
                file = file.replace('.fits', '')
                if ext_threshold == None:
                    ext_threshold = 'default'
                file += f'_rb{radius_buffer}_et{ext_threshold}'
                if save_path[-1] != '/':
                    save_path = save_path + '/'
                plt.savefig(f'{save_path}{file}.jpg')
            except:
                print('Error saving figure. Double check path entered.')

    if short_dict:
        short_info = info

        int_peaks = []
        for i in range(n_int_peaks):
            int_peaks.append((float(int_x_coords[i]), float(int_y_coords[i])))

        ext_peaks = []
        for i in range(n_ext_peaks):
            ext_peaks.append((float(x_coords[i]), float(y_coords[i])))

        if n_ext_peaks == 0:
            ext_peaks = 'No significant external peak'
            short_info['ext_peak_val'] = 'No significant external peak'
            short_info['ext_snr'] = 'No significant external peak'
            short_info['ext_prob'] = 'No significant external peak'

        short_info = info
        short_info['int_peak_coord'] = int_peaks
        short_info['ext_peak_coord'] = ext_peaks
        short_info['field_center'] = (0,0)
        short_info['conservative_rms'] = conservative_rms
        short_info['conservative_snr'] = conservative_snr

        del short_info['next_ext_peak']

        return short_info

    else:
        return


def significant(fits_file: str, threshold: float = 0.01, radius_buffer: float = 5.0, ext_threshold: float = None):
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

    summ = summary(fits_file=fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold, short_dict=True, plot=False)
    return (summ['int_prob'][0] < threshold and summ['calc_int_prob'][0] < threshold)
