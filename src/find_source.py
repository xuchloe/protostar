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

_RMS_UNCERT_SIGMA = 5
_RMS_UNCERT_SAMPLES = 100

def _fits_data_index(fits_file: str) -> int:
    """Return the index of the first HDU containing image data.

    Parameters
    ----------
    fits_file : str
        The path to the FITS file.

    Returns
    -------
    int
        The index of the first HDU whose `data` attribute is not `None`.

    Raises
    ------
    OSError
        If the FITS file cannot be opened.
    ValueError
        If the FITS file contains no HDU with data.
    """
    try:
        with fits.open(fits_file) as hdulist:
            # Iterate through the HDUs until one containing data is found.
            # Assume the first HDU with data contains the image.
            for file_index, hdu in enumerate(hdulist):
                if hdu.data is not None:
                    return file_index
    except OSError as err:
        raise OSError(
            f"Unable to open FITS file: {fits_file}"
            ) from err
    raise ValueError(
        f"No HDU containing image data found in {fits_file}."
        )

def _gaussian_2d(
    coord: tuple,
    amp: float,
    sigma: float,
    mu_x: float,
    mu_y: float,
):
    """Evaluate an isotropic 2D Gaussian at one or more coordinates.

    Parameters
    ----------
    coord : tuple
        A tuple `(x, y)` containing the x- and y-coordinates at which to
        evaluate the Gaussian. `x` and `y` may be scalars or array-like.
    amp : float
        The amplitude of the Gaussian.
    sigma : float
        The standard deviation of the Gaussian.
    mu_x : float
        The x-coordinate of the Gaussian center.
    mu_y : float
        The y-coordinate of the Gaussian center.

    Returns
    -------
    float or ndarray
        The Gaussian evaluated at the given coordinates.
    """

    x, y = coord
    return amp * np.exp(-((x - mu_x) ** 2 + (y - mu_y) ** 2) / (2 * sigma ** 2))


def _region_stats(
    fits_file: str,
    radius: list,
    center: list | None = None,
    invert: bool = False,
    gaussian: bool = True,
    internal: bool = True,
    outer_radius: float | None = None,
) -> dict:
    """Find the statistics of a region of an image.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    radius : list
        Sequence of radii, in arcseconds, corresponding to each center
        coordinate.
        If centers are provided, `center` and `radius` must have the same
        length.
    center : list | None, optional
        Sequence of `(x, y)` pixel coordinates defining the centers of the
        circular regions.
        If `None` or empty, the image center is used.
    invert : bool, optional
        Whether to swap the inclusion and exclusion regions.
    gaussian : bool, optional
        Whether to use a 2D Gaussian fit to estimate the true maximum flux and
        its corresponding coordinates.
        If `False`, the brightest pixel is returned without subpixel refinement.
    internal : bool, optional
        Determines the fitting window used for Gaussian refinement.
        If `True`, a 5x5 pixel neighborhood is used. Otherwise, a 3x3
        neighborhood is used.
    outer_radius : float | None, optional
        If provided, restrict the mask to a circular region centered on the
        image center with this radius, in arcseconds.
        Pixels outside this region are excluded regardless of `invert`.

    Returns
    -------
    dict
        Dictionary with the following keys:
        `peak` : float
            Peak flux density in Jy.
        `field_center` : tuple of 2 float
            Image center in pixel coordinates.
        `peak_coord` : tuple of 2 float
            Pixel coordinates of the peak. Integer pixel coordinates are
            returned if Gaussian fitting is not performed or fails; subpixel
            coordinates are returned if Gaussian fitting succeeds.
        `rms` : float
            RMS of the region, in Jy.
        `beam_area` : float
            The image's beam area, in arcsec^2.
        `x_axis` : float
            The image's x-axis length in arcsec.
        `y_axis` : float
            The image's y-axis length in arcsec.
        `incl_area` : float
            The area included in the mask, in arcsec^2.
        `excl_area` : float
            The area excluded by the mask, in arcsec^2.
        `n_incl_meas` : float
            Estimated number of independent beam areas included in the mask.
        `n_excl_meas` : float
            Estimated number of independent beam areas excluded by the mask.
        `mad` : float
            The median absolute deviation of the image flux density, in Jy.
        `sd_mad` : float
            The standard deviation, as estimated by the MAD, of the image flux
            density, in Jy.
        `neg_peak` : float | None
            Most negative pixel flux density, in Jy, in the image.
            `None` if no negative pixel values are present.

    Raises
    ------
    OSError
        If the FITS file cannot be opened.
    ValueError
        If `center` and `radius` have different lengths, or if the applied mask
        contains no pixels.

    Notes
    -----
    When `gaussian` is `True`, the peak flux and position are refined using a
    bounded two-dimensional Gaussian fit around the brightest pixel. A 5x5
    pixel window is used for internal peaks and a 3x3 pixel window for external
    peaks. If the fitting window does not fit within the image or the fit
    fails, the pixel maximum and its integer coordinates are returned.
    """

    if center:
        if len(center) != len(radius):
            raise ValueError(
                f"'center' and 'radius' must have the same length. "
                f"(got {len(center)} and {len(radius)})."
            )
    i = _fits_data_index(fits_file)

    try:
        with fits.open(fits_file) as hdulist:
            image_hdu = hdulist[i]
            data = image_hdu.data
    except OSError as err:
        raise OSError(f'Unable to open {fits_file}') from err

    # Record the most negative pixel value for image diagnostics.
    neg_peak = float(np.min(data[0]))
    if neg_peak >= 0:
        neg_peak = None

    mad = float(median_abs_deviation(data[0].flatten()))
    # Convert the MAD to an equivalent Gaussian standard deviation.
    sd_mad = float(norm.ppf(0.84) / norm.ppf(0.75) * mad)

    x_dim = image_hdu.header['NAXIS1']
    y_dim = image_hdu.header['NAXIS2']

    x_dist_array = np.tile(np.arange(x_dim), (y_dim, 1))
    y_dist_array = x_dist_array.T

    # Keep center pixel coordinates if specified or set to default if
    # unspecified.
    center_pix = center
    field_center = ((x_dim - 1) / 2, (y_dim - 1) / 2)
    if not center:
        center_pix = [field_center]
        if len(radius) > 1:
            center_pix = center_pix * len(radius)

    # Find the units of the axes.
    x_unit = image_hdu.header['CUNIT1']
    y_unit = image_hdu.header['CUNIT2']

    # Find the cell size, in arcsec.
    x_delt = Angle(abs(image_hdu.header['CDELT1']), x_unit)
    x_cell_size = x_delt.to(u.arcsec).value
    y_delt = Angle(abs(image_hdu.header['CDELT2']), y_unit)
    y_cell_size = y_delt.to(u.arcsec).value

    # Find the beam area, in arcsec^2.
    beam_area = float(
        ((np.pi / 4) * image_hdu.header['BMAJ'] * image_hdu.header['BMIN']
        * (Angle(1, x_unit) * Angle(1, y_unit) / np.log(2))
        .to(u.arcsec ** 2)).value
    )

    # Find the axis sizes, in arcsec.
    x_axis_size = x_dim * x_cell_size
    y_axis_size = y_dim * y_cell_size

    # Compute the distance of every pixel from the first search center.
    dist_from_center = (
        (((x_dist_array - center_pix[0][0]) * x_cell_size) ** 2
        + ((y_dist_array - center_pix[0][1]) * y_cell_size) ** 2) ** 0.5
    )

    # Create the inclusion mask.
    mask = (dist_from_center <= radius[0])
    if len(center_pix) > 1:
        # Combine masks from multiple circular regions.
        for j in range(1, len(center)):
            dist_from_center = (
                (((x_dist_array - center_pix[j][0]) * x_cell_size) ** 2
                + ((y_dist_array - center_pix[j][1]) * y_cell_size) ** 2)
                ** 0.5
            )
            mask = np.logical_or(mask, (dist_from_center <= radius[j]))

    if invert:
        mask = np.logical_not(mask)

    if outer_radius is not None:
        # Restrict the mask to the central circular region used for the search.
        dist_from_field_center = (
            (((x_dist_array - field_center[0]) * x_cell_size) ** 2
            + ((y_dist_array - field_center[1]) * y_cell_size) ** 2) ** 0.5
        )
        outer_mask = (dist_from_field_center <= outer_radius)
        mask = np.logical_and(mask, outer_mask)

    incl_area = float(mask.sum() * x_cell_size * y_cell_size)
    excl_area = float(np.logical_not(mask).sum() * x_cell_size * y_cell_size)

    masked_data = data[0][mask]

    # Get the peak flux density.
    try:
        peak = float(np.max(masked_data))
    except ValueError:
        raise ValueError(
            "No values remain after the mask was applied. "
            "Check inclusion and exclusion radii."
        )

    # Find the coordinates of the peak.
    # Use the first occurrence within the mask if multiple pixels share the
    # maximum value.
    peak_pix = peak_pix = np.where(mask & (data[0] == peak))
    peak_x = int(peak_pix[1][0])
    peak_y = int(peak_pix[0][0])
    peak_coord = (peak_x, peak_y)

    # Refine the peak value and location using a bounded 2D Gaussian fit.
    # Use data from a 5x5 region if the peak is internal and if this region
    # fits in the image.
    data_array = np.array(data[0])
    if (
        gaussian
        and internal
        and (peak_x - 2) >= 0
        and (peak_x + 2) < x_dim
        and (peak_y - 2) >= 0
        and (peak_y + 2) < y_dim
    ):
        z_data = data_array[peak_y - 2:peak_y + 3, peak_x - 2:peak_x + 3]
        z_data = z_data.flatten()
        y_data = [-2] * 5 + [-1] * 5 + [0] * 5 + [1] * 5 + [2] * 5
        x_data = [-2, -1, 0, 1, 2] * 5

        try:
            popt, _ = curve_fit(
                _gaussian_2d, (x_data, y_data), z_data,
                bounds=(
                    [peak, 0, -1, -1],
                    [float('inf'), float('inf'), 1, 1]
                )
            )
            amp, _, mu_x, mu_y = popt
            peak = float(amp)
            peak_coord = (float(peak_x + mu_x), float(peak_y + mu_y))
        except RuntimeError:
            pass  # Subpixel fitting failed; use pixel values instead.

    # Use data from a 3x3 region if the peak is external and if this region
    # fits in the image.
    elif (
        gaussian
        and (not internal)
        and (peak_x - 1) >= 0
        and (peak_x + 1) < x_dim
        and (peak_y - 1) >= 0
        and (peak_y + 1) < y_dim
    ):
        z_data = data_array[peak_y - 1:peak_y + 2, peak_x - 1:peak_x + 2]
        z_data = z_data.flatten()
        y_data = [-1] * 3 + [0] * 3 + [1] * 3
        x_data = [-1, 0, 1] * 3

        try:
            popt, _ = curve_fit(
                _gaussian_2d, (x_data, y_data), z_data,
                bounds=(
                    [peak, 0, -1, -1],
                    [float('inf'), float('inf'), 1, 1]
                )
            )
            amp, _, mu_x, mu_y = popt
            peak = float(amp)
            peak_coord = (float(peak_x + mu_x), float(peak_y + mu_y))
        except RuntimeError:
            pass  # Subpixel fitting failed; use pixel values instead.

    rms = float(np.sqrt(np.var(masked_data)))

    stats = {
        'peak': peak,
        'field_center': field_center,
        'peak_coord': peak_coord,
        'rms': rms,
        'beam_area': beam_area,
        'x_axis': float(x_axis_size),
        'y_axis': float(y_axis_size),
        'incl_area': incl_area,
        'excl_area': excl_area,
        'n_incl_meas': float(incl_area / beam_area),
        'n_excl_meas': float(excl_area / beam_area),
        'mad': mad,
        'sd_mad': sd_mad,
        'neg_peak': neg_peak
    }

    return stats


def _expected_exceedances_from_rms_uncertainty(
    peak: float,
    rms: float,
    n_excl_meas: float,
    n_incl_meas: float | None = None,
) -> float:
    """Estimate the expected number of independent noise measurements greater
    than or equal to `peak`, accounting for uncertainty in the estimated RMS.

    Parameters
    ----------
    peak : float
        The threshold value. The expected number of independent noise
        measurements greater than or equal to this value is estimated.
    rms : float
        The estimated RMS of the Gaussian distribution.
    n_excl_meas : float
        The effective number of independent measurements contributing to the
        RMS estimate. The suffix 'excl' indicates that these measurements come
        from an exclusion region that may differ from the region over which the
        expected number is estimated.
    n_incl_mease : float | None, optional
        The effective number of independent measurements over which the expected
        number is evaluated.
        If `None` or empty, `n_excl_meas` is used.

    Returns
    -------
    float
        The estimated expected number of independent noise measurements greater
        than or equal to `peak`, accounting for uncertainty in the estimated RMS.

    Raises
    ------
    ValueError
        If `rms`, `n_excl_mease`, or `n_incl_meas` (when provided) is not
        positive.

    Notes
    -----
    The RMS uncertainty is incorporated into the expected number estimate
    rather than assuming the measured RMS is exact. The uncertainty in the RMS
    estimate is modeled as a Gaussian distribution with standard deviation
    `rms / sqrt(n_excl_meas)`. The RMS may be estimated from one set of measurements
    (the exclusion region) while the expected number may be evaluated over
    another set of measurements (the inclusion region). The input distribution
    is assumed to be Gaussian. The expected number is estimated by numerically
    marginalizing over the RMS uncertainty using a Gaussian weighting function
    sampled at 100 evenly spaced points spanning ±5 standard deviations.
    """

    if rms <= 0:
        raise ValueError(f"'rms' must be positive. Got {rms}.")
    if n_excl_meas <= 0:
        raise ValueError(f"'n_excl_meas' must be positive. Got {n_excl_meas}.")
    if n_incl_meas is not None and n_incl_meas <= 0:
        raise ValueError(f"'n_incl_meas' must be positive. Got {n_incl_meas}.")

    # Estimate RMS uncertainty assuming Gaussian noise statistics.
    rms_err = rms / np.sqrt(n_excl_meas)

    # Evaluate the expected exceedance count over possible RMS values, weighted
    # by the assumed Gaussian distribution of RMS uncertainty.
    uncert = np.linspace(
        -_RMS_UNCERT_SIGMA * rms_err,
        _RMS_UNCERT_SIGMA * rms_err,
        _RMS_UNCERT_SAMPLES
    )
    uncert_pdf = norm.pdf(uncert, loc=0, scale=rms_err)

    # Marginalize the probability over the RMS uncertainty distribution and
    # calculate expectation.
    if n_incl_meas is None:
        n_incl_meas = n_excl_meas
    return float(
        np.sum(norm.cdf(-peak / (rms + uncert)) * n_incl_meas * uncert_pdf)
        / np.sum(uncert_pdf)
    )


def _statistics_from_rms_uncertainty(
    fits_file: str,
    center: list | None = None,
    threshold: float = 0.01,
    radius_buffer: float = 5.0,
    ext_threshold: float | None = None,
) -> dict:
    """Calculate expected numbers of independent noise measurements with flux
    densities greater than or equal to internal peaks and significant external
    peaks, along with related peak statistics for an image.

    Parameters
    ----------
    fits_file : str
        The path of the FITS file that contains the image.
    center : list | None, optional
        A list of center coordinates in units of pixels.
        If `None` or empty, field center coordinates are used.
    threshold : float, optional
        The maximum expected number of independent noise measurements with flux
        densities greater than or equal to an internal peak for the peak to be
        considered significant, assuming no source is present in the image.
    radius_buffer : float, optional
        The amount of buffer, in arcsec, to add to the beam FWHM to get the
        initial search radius.
    ext_threshold : float | None, optional
        The maximum expected number of independent noise measurements with flux
        densities greater than or equal to an external peak for the peak to be
        considered significant, assuming no source is present in the image.
        If no value is given, `1e-3`, `1e-6`, or `1e-12` is used, depending on
        the signal-to-noise ratio of the brightest internal peak calculated
        using the initial external-region RMS estimate.

    Returns
    -------
    dict
        Dictionary with the following keys:
        `field_center` : tuple of 2 float
            Image center in pixel coordinates.
        `rms_val` : float
            The estimated RMS, in Jy, of the image, excluding circular
            neighborhoods around flux densities that were considered to be
            significant.
        `mad` : float
            The median absolute deviation of the image flux density, in Jy.
        `sd_mad` : float
            The standard deviation of image flux density, in Jy, as estimated
            by the MAD.
        `n_incl_meas` : float
            The number of measurements (beams) included in the mask.
        `n_excl_meas` : float
            The number of measurements (beams) excluded by the mask.
        `fwhm` : float
            The beam major axis FWHM, in arcsec.
        `incl_radius` : float
            The radius, in arcsec, of the initial inclusion region.
        `neg_peak` : float | None
            Most negative pixel flux density, in Jy, in the image.
            `None` if no negative pixel values are present.
        `int_peak_val` : list of float
            The flux density, in Jy, of the brightest internal peak and the
            flux densities, in Jy, of the remaining significant internal peaks,
            if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `int_peak_coord` : list of tuple of 2 int or float
            The pixel coordinates of the brightest internal peak and the
            remaining significant internal peaks, if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `int_exp_exceed` : list of float
            The expected number of independent noise measurements with flux
            densities greater than or equal to the brightest internal peak and
            the remaining significant internal peaks, if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `int_snr` : list of float
            The signal to noise ratios of the brightest internal peak and the
            remaining significant internal peaks, if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `ext_peak_val` : list of float
            The flux densities, in Jy, of the significant external peaks, if
            these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant external peaks are found.
        `ext_peak_coord` : list of tuple of 2 int or float
            The pixel coordinates of the significant external peaks, if these
            exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant external peaks are found.
        `ext_exp_exceed` : list of float
            The expected number of independent noise measurements with flux
            densities greater than or equal to the significant external peaks,
            if these exist.
            Peaks are arranged in decreasing brightness.
        `ext_snr` : list of float
            The signal to noise ratios of the external peaks, if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant external peaks are found.
        `next_ext_peak` : float
            The flux density, in Jy, of the brightest non-significant external
            peak.

    Raises
    ------
    OSError
        If the FITS file cannot be opened.
    ValueError
        Propagated from _expected_exceedances_from_rms_uncertainty(), if `rms`,
        `n_excl_meas`, or `n_incl_meas` (when provided) is not positive.

    Notes
    -----
    The RMS is estimated iteratively by identifying statistically significant
    external peaks and excluding circular regions with radii equal to the beam
    FWHM around those peaks. Using the updated RMS estimate, the expected
    number of independent noise measurements with flux densities greater than
    or equal to those of internal peaks is then evaluated.

    To reduce false detections caused by the point spread function of bright
    sources, empirical thresholds are applied. If external significance
    threshold `ext_threshold` is `None`, it is set to `1e-3`, `1e-6`, or
    `1e-12` depending on whether the SNR of the brightest internal peak is
    below `20`, between `20` and `100`, or at least `100`, respectively.
    Additional internal peaks are required to have flux densities greater
    than `1/100` of the brightest internal peak.

    Expected numbers are calculated assuming Gaussian statistics.
    """

    i = _fits_data_index(fits_file)

    # Open FITS file and extract image HDU.
    # Extract beam information from image HDU.
    try:
        with fits.open(fits_file) as hdulist:
            hdu = hdulist[i]
            beam_fwhm = float(
                (Angle(hdu.header['BMAJ'], hdu.header['CUNIT1']))
                .to(u.arcsec).value
            )
    except OSError as err:
        raise OSError(
            f"Unable to open FITS file: {fits_file}"
            ) from err

    search_radius = beam_fwhm + radius_buffer

    # Search for the brightest internal peak.
    int_stats1 = _region_stats(
        fits_file=fits_file,
        radius=[search_radius],
        center=center,
        invert=False,
        gaussian=False,
        internal=True
    )
    int_peak1 = int_stats1['peak']
    int_coord1 = int_stats1['peak_coord']
    # `n_incl_meas` is the effective number of independent beam areas in the
    # internal search region and is therefore the same for all internal peaks.
    n_incl_meas = int_stats1['n_incl_meas']
    field_center = int_stats1['field_center']
    # `mad` and `sd_mad` are evaluated using the external region and therefore
    # are the same for all peaks.
    mad = int_stats1['mad']
    sd_mad = int_stats1['sd_mad']
    neg_peak = int_stats1['neg_peak']

    # Find external peaks and get their info.
    center = [field_center]
    radius = [search_radius]
    ext_stats1 = _region_stats(
        fits_file=fits_file,
        radius=radius,
        center=center,
        invert=True,
        gaussian=False,
        internal=False
    )
    # `n_excl_meas` is the effective number of independent beam areas in the
    # external region and is therefore the same for all external peaks.
    n_excl_meas = ext_stats1['n_incl_meas']
    ext_peak1 = ext_stats1['peak']
    rms = ext_stats1['rms']  # May change as we exclude more external peaks.
    ext_exp_exceed1 = _expected_exceedances_from_rms_uncertainty(
        peak=ext_peak1,
        rms=rms,
        n_excl_meas=n_excl_meas
    )

    prob_dict = {
        'field_center': field_center,
        'rms_val': None,
        'mad': mad,
        'sd_mad': sd_mad,
        'n_incl_meas': n_incl_meas,
        'n_excl_meas': n_excl_meas,
        'fwhm': beam_fwhm,
        'incl_radius': search_radius,
        'neg_peak': neg_peak,
        'int_peak_val': [],
        'int_peak_coord': [],
        'int_exp_exceed': [],
        'int_snr': [],
        'ext_peak_val': [],
        'ext_peak_coord': [],
        'ext_exp_exceed': [],
        'ext_snr': [],
        'next_ext_peak': None
    }

    # Update ext_threshold based on SNR of internal peak, if needed.
    # This prevents spurious detections due to the point spread function.
    int_snr1 = int_peak1 / rms
    if ext_threshold is None:
        if int_snr1 < 20:
            ext_threshold = 1e-3
        elif int_snr1 < 100:
            ext_threshold = 1e-6
        else:
            ext_threshold = 1e-12

    ext_significant = ext_exp_exceed1 < ext_threshold
    if not ext_significant:
        prob_dict['next_ext_peak'] = ext_peak1

    # Find significant external peaks, if they exist, and exclude them from
    # the region where RMS is measured.
    while ext_significant:
        ext_stats = _region_stats(
            fits_file=fits_file,
            radius=radius,
            center=center,
            invert=True,
            # Not fitting with Gaussian because a significant external peak
            # might not exist.
            gaussian=False,
            internal=False
        )
        peak = ext_stats['peak']
        rms = ext_stats['rms']

        ext_exp_exceed = _expected_exceedances_from_rms_uncertainty(
            peak=peak,
            rms=rms,
            n_excl_meas=n_excl_meas
        )
        if ext_exp_exceed < ext_threshold:
            ext_stats = _region_stats(
                fits_file=fits_file,
                radius=radius,
                center=center,
                invert=True,
                # Fit with Gaussian once we know that a significant external
                # peak actually does exist.
                gaussian=True,
                internal=False
            )
            coord = ext_stats['peak_coord']
            peak = ext_stats['peak']
            ext_exp_exceed = _expected_exceedances_from_rms_uncertainty(
                peak=peak,
                rms=rms,
                n_excl_meas=n_excl_meas
            )
            prob_dict['ext_peak_val'].append(peak)
            prob_dict['ext_peak_coord'].append(coord)
            prob_dict['ext_exp_exceed'].append(ext_exp_exceed)
            prob_dict['ext_snr'].append(peak / rms)
            center.append(coord)
            radius.append(beam_fwhm)
        else:
            prob_dict['next_ext_peak'] = peak
            ext_significant = False

    prob_dict['rms_val'] = rms

    # Find expected number of independent noise measurements with flux
    # densities greater than or equal to the first internal peak, using the
    # updated RMS estimate.
    int_exp_exceed1 = _expected_exceedances_from_rms_uncertainty(
        peak=int_peak1,
        rms=rms,
        n_excl_meas=n_excl_meas,
        n_incl_meas=n_incl_meas
    )

    int_significant = int_exp_exceed1 < threshold

    # Gaussian interpolation for a significant first internal peak to get a
    # better estimate of its flux and coordinates, using the updated RMS.
    if int_significant:
        int_stats_final = _region_stats(
            fits_file=fits_file,
            radius=[search_radius],
            center=center,
            invert=False,
            gaussian=True,
            internal=True
        )
        int_coord_final = int_stats_final['peak_coord']
        int_peak_final = int_stats_final['peak']
        prob_dict['int_peak_val'].append(int_peak_final)
        prob_dict['int_peak_coord'].append(int_coord_final)
        int_exp_exceed1 = _expected_exceedances_from_rms_uncertainty(
            peak=int_peak_final,
            rms=rms,
            n_excl_meas=n_excl_meas,
            n_incl_meas=n_incl_meas
        )
        prob_dict['int_exp_exceed'].append(int_exp_exceed1)
        prob_dict['int_snr'].append(int_peak_final / rms)

    # Treat the first internal peak like an external peak, in the sense that we
    # only exclude a small area around this peak so that we can look for
    # any additional sources inside the internal region.
    if int_significant:
        center = [int_coord_final]
    radius = [beam_fwhm]
    # Find any internal peaks in addition to the first internal peak.
    while int_significant:
        int_stats = _region_stats(
            fits_file=fits_file,
            radius=radius,
            center=center,
            invert=True,
            # As before, don't fit to Gaussian if we don't yet know that a
            # significant source exists.
            gaussian=False,
            internal=True,
            outer_radius=search_radius
        )
        int_peak = int_stats['peak']
        int_exp_exceed = _expected_exceedances_from_rms_uncertainty(
            peak=int_peak,
            rms=rms,
            n_excl_meas=n_excl_meas,
            n_incl_meas=n_incl_meas
        )
        if (
            int_exp_exceed < threshold
            # Additional condition to mitigate false positives due to the point
            # spread function of very bright internal sources.
            and int_peak > int_peak_final / 100
        ):
            int_stats = _region_stats(
                fits_file=fits_file,
                radius=radius,
                center=center,
                invert=True,
                gaussian=True,
                internal=True,
                outer_radius=search_radius
            )
            int_coord = int_stats['peak_coord']
            int_peak = int_stats['peak']
            int_exp_exceed = _expected_exceedances_from_rms_uncertainty(
                peak=int_peak,
                rms=rms,
                n_excl_meas=n_excl_meas,
                n_incl_meas=n_incl_meas
            )
            prob_dict['int_peak_val'].append(int_peak)
            prob_dict['int_peak_coord'].append(int_coord)
            prob_dict['int_exp_exceed'].append(int_exp_exceed)
            prob_dict['int_snr'].append(int_peak / rms)
            center.append(int_coord)
            radius.append(beam_fwhm)
        else:
            int_significant = False

    return prob_dict


def _statistics_from_extreme_peaks(prob_dict: dict) -> dict:
    """Calculate expected numbers of independent noise measurements with flux
    densities greater than or equal to internal peaks and the brightest
    non-significant external peak, along with related peak statistics.

    Parameters
    ----------
    prob_dict : dict
        A dictionary of image statistics, as returned by
        _statistics_from_rms_uncertainty().

    Returns
    -------
    dict
        The input dictionary, modified in place to include additional
        expected numbers and signal-to-noise statistics. The following keys are
        present:
        `field_center` : tuple of 2 float
            Image center in pixel coordinates.
        `rms_val` : float
            The estimated RMS, in Jy, of the image, excluding circular
            neighborhoods around flux densities that were considered to be
            significant.
        `mad` : float
            The median absolute deviation of the image flux density, in Jy.
        `sd_mad` : float
            The standard deviation of image flux density, in Jy, as estimated
            by the MAD.
        `n_incl_meas` : float
            The number of measurements (beams) included in the mask.
        `n_excl_meas` : float
            The number of measurements (beams) excluded by the mask.
        `fwhm` : float
            The beam major axis FWHM, in arcsec.
        `incl_radius` : float
            The radius, in arcsec, of the initial inclusion region.
        `neg_peak` : float | None
            Most negative pixel flux density, in Jy, in the image.
            `None` if no negative pixel values are present.
        `int_peak_val` : list of float
            The flux density, in Jy, of the brightest internal peak and the
            flux densities, in Jy, of the remaining significant internal peaks,
            if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `int_peak_coord` : list of tuple of 2 int or float
            The pixel coordinates of the brightest internal peak and the
            remaining significant internal peaks, if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `int_exp_exceed` : list of float
            The expected number of independent noise measurements with flux
            densities greater than or equal to the brightest internal peak and
            the remaining significant internal peaks, if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `int_snr` : list of float
            The signal to noise ratios of the brightest internal peak and the
            remaining significant internal peaks, if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `ext_peak_val` : list of float
            The flux densities, in Jy, of the significant external peaks, if
            these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant external peaks are found.
        `ext_peak_coord` : list of tuple of 2 int or float
            The pixel coordinates of the significant external peaks, if these
            exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant external peaks are found.
        `ext_exp_exceed` : list of float
            The expected number of independent noise measurements with flux
            densities greater than or equal to the significant external peaks,
            if these exist.
            Peaks are arranged in decreasing brightness.
        `ext_snr` : list of float
            The signal to noise ratios of the external peaks, if these
            exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant external peaks are found.
        `next_ext_peak` : float
            The flux density, in Jy, of the brightest non-significant external
            peak.
        `calc_rms_val` : float
            The RMS, in Jy, for which the expected number of independent noise
            measurements greater than or equal to `next_ext_peak` is one over
            the external region.
        neg_peak_rms_val : float | None
            The RMS, in Jy, for which the expected number of independent noise
            measurements less than or equal to the image's most negative pixel is one.
            `None` if no negative pixel values are present.
        `calc_ext_exp_exceed` : float
            The expected number of independent noise measurements with flux
            densities greater than or equal to the brightest non-significant
            external peak, calculated using the more conservative of
            `calc_rms_val` and `neg_peak_rms_val` when the latter is available.
        `calc_ext_snr` : float
            The SNR of the brightest non-significant external peak, calculated
            with the more conservative (smaller value) of `calc_rms_val` and
            `neg_peak_rms_val`.
        `calc_int_exp_exceed` : list of float
            The expected number of independent noise measurements with flux
            densities greater than or equal to the brightest internal peak and
            the remaining significant internal peaks, if these exist,
            calculated using the more conservative of `calc_rms_val` and
            `neg_peak_rms_val` when the latter is available.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.
        `calc_int_snr` : list of float
            The signal to noise ratios, calculated using the more
            conservative of `calc_rms_val` and `neg_peak_rms_val` when the
            latter is available, of the brightest internal peak and the
            remaining significant internal peaks, if these exist.
            Peaks are arranged in decreasing brightness.
            Empty if no significant internal peaks are found.

    Raises
    ------
    ValueError
        If `prob_dict['next_ext_peak']` is less than 0 or
        `prob_dict['n_excl_meas']` is less than or equal to 1.

    Notes
    -----
    Expected numbers are calculated assuming Gaussian statistics.

    This function modifies `prob_dict` in place by adding the keys
    `calc_rms_val`, `neg_peak_rms_val`, `calc_ext_exp_exceed`, `calc_ext_snr`,
    `calc_int_exp_exceed`, and `calc_int_snr`.
    """
    int_peak_val = prob_dict['int_peak_val']
    next_ext_peak = prob_dict['next_ext_peak']
    if next_ext_peak is None:
        raise ValueError(
            "Cannot conduct statistical calculations because no "
            "non-significant external peak was found ('next_ext_peak' is "
            "`None`)."
        )
    if next_ext_peak < 0:
        raise ValueError(
            "Negative 'next_ext_peak' will result in incorrect statistical "
            "calculations. "
            f"Got ({next_ext_peak})."
        )
    n_incl_meas = prob_dict['n_incl_meas']
    n_excl_meas = prob_dict['n_excl_meas']
    if n_excl_meas <= 1:
        raise ValueError(
            "'n_excl_meas' less than or equal to 1 will result in incorrect "
            "statistical calculations. "
            f"Got ({n_excl_meas})."
        )

    excl_z_score = -norm.ppf(1 / n_excl_meas)
    excl_rms_val = next_ext_peak / excl_z_score
    prob_dict['calc_rms_val'] = float(excl_rms_val)

    img_z_score = norm.ppf(1 / (n_incl_meas + n_excl_meas))
    neg_peak = prob_dict['neg_peak']

    if neg_peak is not None:
        neg_peak_rms_val = neg_peak / img_z_score
        prob_dict['neg_peak_rms_val'] = float(neg_peak_rms_val)
        # Choose the more conservative of `excl_rms_val` and
        # `neg_peak_rms_val`.
        rms_val = (
            excl_rms_val if excl_rms_val < neg_peak_rms_val
            else neg_peak_rms_val
        )
    else:
        prob_dict['neg_peak_rms_val'] = None
        rms_val = excl_rms_val

    prob_dict['calc_ext_exp_exceed'] = (
        float(norm.cdf(-next_ext_peak / rms_val)) * n_excl_meas
    )
    prob_dict['calc_ext_snr'] = float(next_ext_peak / rms_val)

    calc_int_exp_exceed = []
    calc_int_snr = []
    for peak in int_peak_val:
        calc_int_exp_exceed.append(
            float(norm.cdf(-peak / rms_val)) * n_incl_meas
        )
        calc_int_snr.append(float(peak / rms_val))
    prob_dict['calc_int_exp_exceed'] = calc_int_exp_exceed
    prob_dict['calc_int_snr'] = calc_int_snr

    return prob_dict


def summary(fits_file: str, threshold: float = 0.01, radius_buffer: float = 5.0, ext_threshold: float = None,\
            short_dict: bool = True, plot: bool = True, save_path: str = ''):
    """
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
                    floats
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
                    floats
                        The exclusion regions' maximum fluxes in Jy.
                        If there are multiple entries in this list,
                        they are the maxmimum fluxes as the exclusion region becomes increasingly small
                        as external peaks deemed significant are added to the inclusion region.
            list
                A list with:
                    tuples (float, float)
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
                    floats:
                        The radii in arcsec of inclusion zones.
            float
                The inclusion region's signal to noise ratio.
            list
                A list with:
                    floats
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
            dicts
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
                            floats
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
    """
    info = (_statistics_from_extreme_peaks(_statistics_from_rms_uncertainty(fits_file=fits_file, threshold=threshold, radius_buffer=radius_buffer,\
                                                                ext_threshold=ext_threshold)))

    center = info['field_center']

    header_data = fits.getheader(fits_file)
    pixel_scale = Angle(abs(header_data['CDELT1']), header_data['CUNIT1']).to_value('arcsec')

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

        try:
            title = fits_file[fits_file.rindex('/')+1:fits_file.index('.fits')]
        except:
            title = fits_file
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
                if ext_threshold is None:
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
            short_info['ext_exp_exceed'] = 'No significant external peak'

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
    """
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
    """

    #make sure reasonable input
    if not (threshold >= 0 and threshold <= 1):
        raise ValueError('Threshold must be between 0 and 1, inclusive.')

    summ = summary(fits_file=fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold, short_dict=True, plot=False)
    return (summ['int_exp_exceed'][0] < threshold and summ['calc_int_exp_exceed'][0] < threshold)
