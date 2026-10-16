from astropy.io import fits
import numpy as np
import emcee
import corner
import math
from astropy.coordinates import Angle
import astropy.units as units
import scipy.special as sp
from scipy.stats import norm
import warnings
import itertools
from uncertainties import ufloat
from uv_fit import *
from find_source import summary
import sigfig
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from casatools import componentlist
from casatasks import importfits, imfit, simobserve, uvmodelfit
import os
import re
import shutil

def compare_imfit_summary(
        fits_file: str | Path,
        clean: bool = True,
    ):
    """
    Perform image fitting with CASA's `imfit()` and with `summary()`.

    Parameters
    ----------
    fits_file : str | Path
        The path of the FITS file that contains the image. The FITS file is
        assumed to be in the same format as produced by CASA.
    clean : bool, optional
        Whether to clean up (shorten) the `imfit()` results.

    Returns
    -------
    tuple of 2 dict
        The first dictionary is the (shortened, if `clean` is `True`) CASA
        `imfit()` results dictionary and the second dicitonary is the
        (shortened, if `clean` is `True`) `summary()` results dictionary.

    Notes
    -----
    `imfit()` fits a Gaussian, while `summary()` fits a point source.
    """
    # Get result from CASA's imfit().
    imagename = os.path.splitext(fits_file)[0] + ".image"
    user_imagename = os.path.expanduser(imagename)

    if not os.path.exists(user_imagename):
        importfits(
            fitsimage=fits_file,
            imagename=imagename,
            overwrite=False,
        )
    else:
        print(f"Using existing CASA image: {imagename}.")

    imfit_result = imfit(
        imagename=imagename,
        append=False,
        logfile='',
        complist='',
        overwrite=True,
    )

    # Only keep peak value (Jy), peak error (Jy), and coordinates relative to
    # field center (arcsec) for each source for the clean version.
    if clean:
        with fits.open(fits_file) as hdul:
            header_data = hdul[0].header
        cdelt1 = header_data['CDELT1']
        cdelt2 = header_data['CDELT2']
        axis1_sign = cdelt1 / abs(cdelt1)
        axis2_sign = cdelt2 / abs(cdelt2)
        naxis1 = header_data['NAXIS1']
        naxis2 = header_data['NAXIS2']
        center = ((naxis1 - 1) / 2, (naxis2 - 1) / 2)

        cleaned_imfit = {}
        for component_num, source in imfit_result['results'].items():
            pix_per_arcsec = imfit_result['pixelsperarcsec']
            if component_num != 'nelements':
                imfit_peak_val = source['peak']['value']
                imfit_peak_err = source['peak']['error']
                imfit_pixelcoords = source['pixelcoords']
                imfit_relative_pix = (
                    float(
                        (imfit_pixelcoords[0] - center[0])
                        / pix_per_arcsec[0] * axis1_sign
                    ),
                    float(
                        (imfit_pixelcoords[1] - center[1])
                        / pix_per_arcsec[1] * axis2_sign
                    ),
                )

                cleaned_imfit[component_num] = {
                    'peak_value' : imfit_peak_val,
                    'peak_error' : imfit_peak_err,
                    'arcsec_coords' : imfit_relative_pix,
                }

        imfit_result = cleaned_imfit

    # Get result from summary().
    summary_result = summary(
        fits_file,
        plot=False,
    )

    # Only keep the information on the field center; internal and external peak
    # values (Jy) and coordinates relative to the field center (arcsec), and
    # their uncertainties.
    if clean:
        keys_to_keep = [
            'field_center',
            'int_peak_val',
            'int_peak_coord',
            'int_peak_uncert',
            'int_pos_uncert',
            'ext_peak_val',
            'ext_peak_coord',
            'ext_peak_uncert',
            'ext_pos_uncert',
        ]

        summary_result = {
            k: summary_result[k] for k in keys_to_keep
        }

    return (imfit_result, summary_result)


def generate_ms_simobserve(
    project_name: str,
    source_position: str,
    source_type : str,
    flux: float,
    major_axis: str | None = None,
    ratio: float | None = None,
    position_angle: str | None = None,
    flux_unit: str = 'Jy',
    field_center: str = 'J2000 18h00m00.0s -23d00m00.0s',
    antenna_list: str | Path = 'alma;0.5arcsec',
    incenter: str | int | float = '230.0GHz',
    compwidth: str | int | float = '2.0GHz',
    comp_nchan: int = 64,
    totaltime: str | int | float = '1200s',
    integration: str | int | float = '10s',
):
    """
    Generate a synthetic CASA Measurement Set with `simobserve()` and perform
    visibility fitting with `uvmodelfit()` and `uv_fit_no_image()`.

    Parameters
    ----------
    project_name : str
        The name of the simulated project.
    source_position : str
        The position of the source, in the format '[Frame] [RA] [Dec]', with
        the following styles accepted:
        '[Frame] HHhMMmSS.Ss +/-DDdMMmSS.Ss'
        '[Frame] HH:MM:SS.S +/-DD:MM:SS.S'
        '[Frame] DDD.DDdeg +/-DD.DDdeg'.
        Valid reference frames are J2000, B1950, and GALACTIC.
    source_type: str
        The type/shape of the source.
        The accepted values (case-sensitive) are 'point', 'Gaussian', and
        'disk'.
    flux : float
        The flux of the source, in units of `flux_unit`.
    major_axis : str | None, optional
        The major axis of the source, if the source is extended.
        The accepted format is a numerical value directly followed by a valid
        angular unit: 'arcsec' (arcseconds), 'arcmin' (arcminutes), 'deg'
        (degrees), 'rad' (radians).
    ratio : float | None, optional
        The ratio of the minor axis to the major axis of the source, if the
        source is extended.
    position_angle: str | None, optional
        The position angle of the source, if the source is not rotationally
        symmetric.
        The accepted format is a numerical value directly followed by a valid
        angular unit: 'deg' (degrees), 'rad' (radians).
    flux_unit: str, optional
        The unit of the source's flux.
        Accepted values are 'Jy' (Jansky), 'mJy' (milli-Jansky), 'uJy'
        (micro-Jansky), 'K' (Kelvin).
    field_center: str, optional
        The position of the field center, in the format '[Frame] [RA] [Dec]',
        with the following styles accepted:
        '[Frame] HHhMMmSS.Ss +/-DDdMMmSS.Ss'
        '[Frame] HH:MM:SS.S +/-DD:MM:SS.S'
        '[Frame] DDD.DDdeg +/-DD.DDdeg'.
        Valid reference frames are J2000, B1950, and GALACTIC.
    antenna_list : str | Path
        A string-parsable antenna list or a path to the antenna configuration
        file.
    incenter: str | int | float, optional
        The central sky frequency of the simulated observation track.
        The accepted format is a numerical value directly followed by a valid
        frequency unit: 'GHz' (Gigahertz), 'MHz' (Megahertz), 'kHz'
        (Kilohertz), 'Hz' (Hertz).
        Pure numbers are assumed to be in Hz.
    compwidth: str | int | float, optional
        The total, continuous bandwidth of the spectral window of the simulated
        observation track.
        The accepted format is a numerical value directly followed by a valid
        frequency unit: 'GHz' (Gigahertz), 'MHz' (Megahertz), 'kHz'
        (Kilohertz), 'Hz' (Hertz).
        Pure numbers are assumed to be in Hz.
    comp_nchan: int, optional
        The number of discrete spectral frequency channels inside the total
        bandwidth of the simulated observation track.
    totaltime: str | int | float, optional
        The total duration of the simulated obesrvation track.
        The accepted format is a numerical value directly followed by a valid
        time unit: 's' (seconds), 'min' (minutes), 'h' (hours), 'd' (days).
        Pure numbers are assumed to be in seconds.
    integration: str, optional
        The integration interval of a single simulated visibility measurement
        stamp.
        The accepted format is a numerical value directly followed by a valid
        time unit: 's' (seconds), 'min' (minutes), 'h' (hours), 'd' (days).
        Pure numbers are assumed to be in seconds.

    Notes
    -----
    The default values are set to match ALMA Band 6 parameters.
    """
    cl_name = f"{project_name}.cl"

    cl = componentlist()

    # Remove old component list, if needed.
    if os.path.exists(cl_name):
        if os.path.isdir(cl_name):
            shutil.rmtree(cl_name)
        else:
            os.remove(cl_name)

    # Remove old simulation directory, if needed.
    if os.path.exists(project_name) and os.path.isdir(project_name):
        shutil.rmtree(project_name)

    cl.done()

    parsed_maj_ax = re.match(r'^(\d+\.?\d*)([a-zA-Z]+)', major_axis)
    min_ax_val = float(parsed_maj_ax.group(1)) * ratio
    minor_axis = f'{min_ax_val}' + parsed_maj_ax.group(2)

    print(minor_axis)

    source_params = {
        'flux': flux,
        'fluxunit': flux_unit,
        'shape': source_type,
        'dir': source_position,
    }
    if source_type in ['Gaussian', 'disk']:
        source_params['majoraxis'] = major_axis
        source_params['minoraxis'] = minor_axis
        source_params['positionangle'] = position_angle

    cl.addcomponent(**source_params)

    # Save components to disk.
    cl.rename(f"{project_name}.cl")
    cl.done()

    print(f"Component list saved as: {project_name}.cl")

    # Generate the MS with simobserve().
    simobserve(
        project=project_name,
        complist=f"{project_name}.cl",
        compwidth=compwidth,
        comp_nchan=comp_nchan,
        obsmode='int',
        antennalist=antenna_list,
        totaltime=totaltime,
        integration=integration,
        indirection=field_center,
        incenter=incenter,
        thermalnoise='',
        graphics='none',
        verbose=True,
        overwrite=True,
    )


def compare_uvmodelfit_uv_fit_no_image(

):

    casa_src_type = source_type.upper()
    uvfit_src_type = source_type.lower()

    if casa_src_type not in ['P', 'G', 'D']:
        raise ValueError(
            "The source type 'source_type' must be one of 'P', 'p', 'G', 'g', "
            f"'D', or 'd'. Got {source_type}."
        )

    sourcepar = [
        flux_guess,
        ra_guess,
        dec_guess,
    ]

    if casa_src_type in ['G', 'D']:
        sourcepar += [
            width_guess,
            ratio_guess,
            pa_guess
        ]

    uvmodelfit(
        vis=str(ms_path),
        field="",
        spw="",
        comptype=casa_src_type,
        sourcepar=sourcepar,
        niter=niter,
    )

    # flux_guess,
    #         0.0, 0.0,
    #         major_guess,
    #         minor_guess,
    #         pa_guess,
    #     ],

    return str(ms_path)


P_PARAMS = ['peak', 'ra', 'dec']
C_PARAMS = ['peak', 'ra', 'dec', 'sigma']
G_PARAMS = ['peak', 'ra', 'dec', 'sigma', 'ratio', 'vis_theta']
D_PARAMS = ['peak', 'ra', 'dec', 'r', 'ratio', 'vis_theta']

SOURCE_TYPES = {
    'p': [3, p_p0, p_prior, p_model, P_PARAMS],
    'c': [4, c_p0, c_prior, c_model, C_PARAMS],
    'g': [6, g_p0, g_prior, g_model, G_PARAMS],
    'd': [6, d_p0, d_prior, d_model, D_PARAMS],
}

def generate_synthetic_info_vis(
    fits_file: str | Path,
    sources: list,
    radius_buffer : float,
    peaks: list,
    coords: list,
    noise: float,
    widths: list | None = None,
    ratios: list | None = None,
    thetas: list | None = None,
) -> tuple:
    """
    Using the frequencies or frequency bins, UV-coordinates, and weights from a
    FITS file, generate a synthetic visibility table and image information
    dictionary for a single- or multi- source model.

    Parameters
    ----------
    fits_file : str | Path
        The path to the FITS file that contains the visibility table as its
        index 1 element.
    sources : list
        List of source types to fit to simulate. Each source type should be one
        of 'p' (point), 'c' (circular gaussian), 'g' (gaussian), or 'd' (disk).
    radius_buffer : float
        The amount of buffer, in arcsec, to add to the beam FWHM to get the
        radius of the circle differentiating the 'internal' and 'external'
        regions of the field.
    peaks : list
        List of source intensities, in Jy, corresponding to each source in
        `sources`.
    coords : list
        List of coordinates as (`ra`, `dec`), in arcsec, corresponding to each
        source in `sources`.
    noise : float
        Noise of the map, in Jy.
    widths : list | None, optional
        List of widths, in arcsec, corresponding to each extended source in
        `sources`.
    ratios : list | None, optional
        List of ratios corresponding to each non-symmetrical source in
        `sources`.
    thetas : list | None, optional
        List of source angles corresponding to each non-symmetrical source in
        `sources`.

    Returns
        tuple of
            dict
                Dictionary with the following keys:
                `CDELT1` : float
                    The pixel size of the first axis, in the units of
                    `info['CUNIT1']`.
                `CUNIT1` : u.UnitBase
                    The units of the first axis.
                `NAXIS1` : int
                    The length of the first axis, in pixels.
                `BMAJ` : float
                    The beam major axis, in degrees.
                `BMIN` : float
                    The beam minor axis, in degrees.
                `int_peak_val` : list
                    A list of guesses for the internal peak intensities, in Jy.
                `int_peak_coord` : tuple of float
                    A list of guesses for the internal peak coordinates, in
                    relative arcseconds.
                `ext_peak_val` : float
                    A list of guesses for the external peak intensities, in Jy.
                `ext_peak_coord` : tuple float
                    A list of guesses for the external peak coordinates, in
                    relative arcseconds.
            list
                The visibility table. List elements are rows of 6 elements
                each: the 0th index corresponding to the frequency or frequency
                bin, the 1st index corresponding to the `u` coordinate, in
                wavelengths; the 2nd index corresponding to the `v` coordinate,
                in wavelengths; the 3rd index corresponding to the real
                component of the visibility, in Jy; the 4th index corresponding
                to the imaginary component of the visibility, in Jy; and the
                5th index corresponding to the visibility weight (the inverse
                of the uncertainty squared), in 1/Jy^2.

    Raises
    ------
    ValueError
        If less than one peak, coordinate, or source type is provided, if
        different numbers of peaks, coordinates, or source types are provided,
        if an invalid source type is provided, if a coordinate is provided in
        an invalid format, if an incorrect number of widths is provided, if an
        incorrect number of ratios is provided, or if an incorrect number of
        source angles is given.

    Notes
    -----
    Assumes the FITS file contains a table as its index 1 element and that this
    table has columns "Frequency" or "Frequency bin", "U" or "indexU", and "V"
    or "indexV".

    `radius_buffer` impacts which sources are deemed 'internal' or 'external'
    in the returned dictionary `info`.
    """
    num_sources = len(sources)
    if num_sources == 0:
        raise ValueError(
            "At least one peak, coordinate, and source type must be provided."
        )
    if num_sources != len(peaks):
        raise ValueError(
            "Length of sources and peaks must be the same."
        )
    if num_sources != len(coords):
        raise ValueError(
            "Length of sources and coords must be the same."
        )
    for i in range(num_sources):
        if sources[i] not in SOURCE_TYPES.keys():
            raise ValueError(
                f"Source type '{sources[i]}' is not recognized. Source type "
                "must be one of the following: 'p' (point), 'c' (circular "
                "gaussian), 'g' (gaussian), or 'd' (disk)."
            )
        if type(coords[i]) not in [tuple, list] or len(coords[i]) != 2:
            raise ValueError(
                f"Coordinate for source {i+1} must be a tuple or list of "
                "length 2 representing (ra, dec) in radians."
            )

    # Count how many widths, ratios, and source angles there should be
    # according to the given source types.
    num_widths = 0
    num_ratios = 0
    num_thetas = 0
    for source in sources:
        if source in ['c', 'd', 'g']:
            num_widths += 1
            if source in ['d', 'g']:
                num_ratios += 1
                num_thetas += 1

    if widths is not None:
        if len(widths) != num_widths:
            raise ValueError(
                f"{num_widths} sources require widths, but only {len(widths)} "
                "have been given."
            )
    if ratios is not None:
        if len(ratios) != num_ratios:
            raise ValueError(
                f"{num_ratios} sources require ratios, but only {len(ratios)} "
                "have been given."
            )
    if thetas is not None:
        if len(thetas) != num_thetas:
            raise ValueError(
                f"{num_thetas} sources require source angles, but only "
                "{len(thetas)} have been given."
            )

    fits_file = Path(fits_file)
    with fits.open(fits_file) as file:
        data = file[1].data
        cdelt1 = file[0].header['CDELT1']
        cunit1 = file[0].header['CUNIT1']
        naxis1 = file[0].header['NAXIS1']
        bmaj = file[0].header['BMAJ']
        bmin = file[0].header['BMIN']

        if file[1].header['TTYPE2'] == 'indexU':
            u_index = 'indexU'
            u_res = file[1].header['U_RES']
        else:
            u_index = 'U'
            u_res = 1
        if file[1].header['TTYPE3'] == 'indexV':
            v_index = 'indexV'
            v_res = file[1].header['V_RES']
        else:
            v_index = 'V'
            v_res = 1

    arcsec_bmaj = Angle(bmaj, cunit1).to(units.arcsec).value
    search_radius = arcsec_bmaj + radius_buffer

    pix_field_center = (0,0)

    int_peaks = []
    int_coords = []
    ext_peaks = []
    ext_coords = []

    # Divide the sources into 'internal' and 'external' sources based on their
    # coordinates.
    for i in range(num_sources):
        if np.sqrt(
            (coords[i][0] - pix_field_center[0]) ** 2
            + (coords[i][1] - pix_field_center[1]) ** 2
        ) <= search_radius:
            int_peaks.append(peaks[i])
            int_coords.append(coords[i])
        else:
            ext_peaks.append(peaks[i])
            ext_coords.append(coords[i])

    info = {}
    info['CDELT1'] = cdelt1
    info['CUNIT1'] = cunit1
    info['NAXIS1'] = naxis1
    info['BMAJ'] = bmaj
    info['BMIN'] = bmin
    info['int_peak_val'] = int_peaks
    info['int_peak_coord'] = int_coords
    info['ext_peak_val'] = ext_peaks
    info['ext_peak_coord'] = ext_coords

    vis_err = noise * np.sqrt(len(data)) * np.sqrt(2)
    weight = 1/(vis_err**2)

    vis = []
    clean_re = []
    clean_im = []

    # Sum the visibilities of each source.
    width_counter = 0
    g_counter = 0
    for i in range(num_sources):
        l0 = Angle(coords[i][0], units.arcsec).to(units.radian).value
        m0 = Angle(coords[i][1], units.arcsec).to(units.radian).value
        model_func = SOURCE_TYPES[sources[i]][3]
        source_info = [peaks[i], l0, m0]
        if sources[i] in ['c', 'd', 'g']:
            source_info.append(
                Angle(widths[width_counter], units.arcsec)
                .to(units.radian).value
            )
            width_counter += 1
            if sources[i] == 'g':
                source_info.append(ratios[g_counter])
                source_info.append(thetas[g_counter] * np.pi/180)
                g_counter += 1
        for j in range(len(data)):
            row = data[j]
            model = model_func(
                source_info,
                row[u_index] * u_res,
                row[v_index] * v_res,
                bmaj,
                np.pi * bmaj * bmin / (4 * np.log(2))
            )
            if i == 0:
                clean_re.append(model.real)
                clean_im.append(model.imag)
            else:
                clean_re[j] += model.real
                clean_im[j] += model.imag

    # Determine whether the given FITS file has a column named 'Frequency' or
    # a column named 'Frequency bin'.
    freq_name = 'Frequency'
    try:
        temp = data[0][freq_name]
    except KeyError:
        freq_name = 'Frequency bin'

    for i in range(len(data)):
        row = data[i]
        new_row = (
            row[freq_name],
            row[u_index] * u_res,
            row[v_index] * v_res,
            (np.random.normal(scale=vis_err) + clean_re[i]) * weight,
            (np.random.normal(scale=vis_err) + clean_im[i]) * weight,
            weight
        )
        vis.append(new_row)

    return info, vis


def sim_best_auto_detect(
    info: dict,
    vis: list,
    n_sources: int | None = None,
    clean_output: bool = True,
    corner_plot: bool = True,
    min_sep : float | None = None,
) -> list:
    """
    Fit visibility data from a FITS file to a given number of point sources or
    to different numbers of point sources, and summarize the results of the
    (best) fit in a list and/or corner plot.

    Paramters
    ---------
    info : dict
        Image domain information and some results from image domain fitting.
        Should include the following keys:
        `all_peaks` : npt.ArrayLike
            Array-like object with each element corresponding to a source
            detected in image fitting. Each element is an array-like object
            with the first index corresponding to the flux density, in Jy, of
            the peak detected in the image domain and the second index
            corresponding to the coordinates, each in arcseconds, of the
            detected peak.
        `n_peaks` : int
            The number of peaks detected in the image domain.
        `rad_pix` : float
            The length of one side of a pixel (assumed to be square), in
            radians.
        `rad_barea` : float
            Beam area, in radians^2.
        `naxis1` : int
            The number of pixels in the first axis of the image (assumed to be
            square).
        `rad_bmaj` : float
            Beam major axis, in radians.
        `rad_bmin` : float
            Beam minor axis, in radians.
    vis : npt.ArrayLike
        An array-like object with 6 indices, corresponding to the data for the
        frequency or frequency bin; the `u` coordinate of the visibility, in
        wavelengths; the `v` coordinate of the visibility, in wavelenghts; the
        real part of the visibility, in Jy; the imaginary part of the
        visibility, in Jy; and the weight (the inverse of the uncertainty
        squared) for each visibility measurement, in 1/Jy^2.
    n_sources : int | None, optional
        The number of point sources in the model that will be fit to the data.
        If none, models with different numbers of point sources will be fit to
        the data and the results of the best-fit model will be reported.
    clean_output : bool, optional
        Whether to return the list of fit results in a more human-parsable
        format. See Returns section for more details.
    corner_plot : bool, optional
        Whether to create a corner plot for each source.
    min_sep : float | None, optional
        The minimum separation, in radians, to differentiate two distinct
        sources.

    Returns
    -------
    list of dict
        A list of a dictionary with the following keys:
        `n_sources` : int
            The number of point sources for which the fit was performed.
        `result` : dict
            A dictionary with the following keys:
            `source_m` : dict
                A dictionary, for each source (m = 1 through m = `n_sources`),
                with the following keys:
                    `type` : str
                        The source type, as a single character code (a key of
                        SOURCE_TYPES).
                        By definition, 'p' for all the sources in
                        best_auto_detect().
                    `peak` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted peak estimate and its
                        uncertainty, respectively, in Jy. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sampled
                        parameters, in Jy, all rounded to 3 signficant figures.
                        Else, a tuple of the fitted peak estimate and its
                        uncertainty, respectively, in Jy.
                    `ra` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Right Ascension
                        estimate and its uncertainty, respectively, in
                        arcseconds relative to the field center. The
                        uncertainty is rounded to 3 significant figures and the
                        estimate is rounded to the appropriate number of digits
                        to match the uncertainty. The second tuple contains
                        the 2.5th, 16th, 50th, 84th, and 97.5th percentile of
                        sample parameters, in relative arcseconds, all rounded
                        to 3 significant figures.
                        Else, a tuple of the fitted Right Ascension estimate
                        and its uncertainty, respectively, in radians relative
                        to the field center.
                    `dec` : tuple of tuple | tuple of float
                        If `clean_output` is `True`, a tuple of 2 tuples. The
                        first tuple contains the fitted Declination estimate
                        and its uncertainty, respectively, in arcseconds
                        relative to the field center. The uncertainty is
                        rounded to 3 significant figures and the estimate is
                        rounded to the appropriate number of digits to match
                        the uncertainty. The second tuple contains the 2.5th,
                        16th, 50th, 84th, and 97.5th percentile of sample
                        parameters, in relative arcseconds, all rounded to 3
                        significant figures.
                        Else, a tuple of the fitted Declination estimate and
                        its uncertainty, respectively, in radians relative to
                        the field center.
                    `best` : dict
                        This key is only present if `clean_output` is `False`.
                        A dictionary, of the collection of sample parameters
                        that resulted in the lowest chi-squared value, with the
                        following keys:
                            `peak` : float
                                The peak flux density, in Jy.
                            `ra` : float
                                The right ascension, in radians relative to the
                                field center.
                            `dec` : float
                                The declination, in radians relative to the
                                field center.
        `bic` : float
            The Bayesian Information Criterion (BIC), estimated as
            chi^2 + k * ln(n), where chi^2 is the chi-squared value of the
            fit, k is the number of parameters, and n is the  number of data
            points to which the model is being fit.
            If `clean_output` is `True`, the BIC is rounded to the hundredths.

    Raises
    ------
    ValueError
        If all models attempted fail to converge.

    Notes
    -----
    The visibility table should contain as its elements  rows of 6 elements
    each: the 0th index corresponding to the frequency or frequency bin, the
    1st index corresponding to the `u` coordinate, in wavelengths; the 2nd
    index corresponding to the `v` coordinate, in wavelengths; the 3rd index
    corresponding to the real component of the visibility, in Jy; the 4th index
    corresponding to the imaginary component of the visibility, in Jy; and the
    5th index corresponding to the visibility weight (the inverse of the
    uncertainty squared), in 1/Jy^2.

    Regardless of `clean_output`, corner plots will be of the visibility
    parameters. This means that the RA and Dec will be in radians, and the
    position angle will be in radians and rotated by pi/2 relative to the
    position angle in the image domain.

    This function employs Markov chain Monte Carlo (MCMC) via the emcee package
    to sample the parameter space.

    This function operates under several assumptions:
    1.  Assume that the presence of more than 2 internal detected peaks
        suggests the source is extended, rather than the presence of more than
        2 separate sources in the interior region.
    2.  Assume that extended external sources are extremely rare, so do not
        apply a similar assumption as (1) to cases with more than 2 external
        sources.
    3.  Assume that the image analysis more often has false positives than
        false negatives. As a consequence, when `n_sources` is None, begin
        guesses for the number of point sources at the number of peaks detected
        in the image domain and then try models with fewer and fewer sources.
    """
    cdelt1 = info['CDELT1']
    cunit1 = info['CUNIT1']
    naxis1 = info['NAXIS1']
    bmaj = info['BMAJ']
    bmin = info['BMIN']

    rad_bmaj = Angle(bmaj, cunit1).to(units.radian).value
    rad_bmin = Angle(bmin, cunit1).to(units.radian).value
    rad_barea = np.pi * rad_bmaj * rad_bmin / (4 * np.log(2))
    rad_pix = float(Angle(cdelt1, cunit1).to(units.radian).value)

    int_peaks = info['int_peak_val']
    int_coords = info['int_peak_coord']
    ext_peaks = info['ext_peak_val']
    ext_coords = info['ext_peak_coord']

    if min_sep is None:
        min_sep = rad_bmaj / 10

    # Assume the presence of more than 2 internal detected peaks suggests the
    # source is extended, rather than the presence of more than 2 separate
    # sources in the interior region.
    if len(int_peaks) > 2:
        int_peaks = int_peaks[:1]
        int_coords = int_coords[:1]

    int_info = list(zip(int_peaks, int_coords))
    if type(ext_peaks) is list:
        ext_info = list(zip(ext_peaks, ext_coords))
    else:
        ext_info = []

    # Create a list of image domain detected peak information. The list
    # elements are tuples (peak, (ra, dec)).
    all_peaks = int_info + ext_info
    all_peaks.sort(reverse=True)  # Sort by peak flux density (high to low).
    n_peaks = len(all_peaks)

    new_info = {
        'all_peaks': all_peaks,
        'n_peaks': n_peaks,
        'rad_pix': rad_pix,
        'rad_barea': rad_barea,
        'naxis1': naxis1,
        'rad_bmaj': rad_bmaj,
        'rad_bmin': rad_bmin
    }

    results = []
    if n_sources is None:
        # Assuming that the image analysis more often has false positives than
        # false negatives, start at the number of peaks detected in the image
        # domain and try models with fewer and fewer sources.
        for i in range(n_peaks, 0, -1):
            temp = auto_detect(
                vis=vis,
                info=new_info,
                n_sources=i,
                clean_output=clean_output,
                corner_plot=corner_plot,
            )
            if temp:
                if results:
                    if results[0]['bic'] - temp[0]['bic'] > 10:
                        results = temp
                else:
                    results = temp
    else:
        results = auto_detect(
            vis=vis,
            info=new_info,
            n_sources=n_sources,
            clean_output=clean_output,
            corner_plot=corner_plot,
            min_sep=min_sep,
        )

    if results:
        return results
    else:
        raise ValueError("All attempts failed to converge.")


def score(z):
    k = int(abs(z)) # sigma bin
    numerator = 2 * (norm.cdf(k+1) - norm.cdf(k)) # probability of being in this sigma bin
    denominator = 2 * (norm.cdf(1) - norm.cdf(0)) # probability of being in 0 to 1 sigma bin, normalization factor so that score of 1 corresponds to being in 0 to 1 sigma bin
    return numerator / denominator


def average_points(fits_file, sources, peaks, coords, noise, widths=None, ratios=None, thetas=None, reps=100):
    all_points = []
    peak_pts = []
    ra_pts = []
    dec_pts = []
    pixel = 0.214255908880632
    image_edge = pixel * 194
    num_pixels = 194**2
    beam_maj = 1.96

    all_peaks = {}
    all_ras = {}
    all_decs = {}
    # all_sigmas = {}
    num_errors = 0
    for i in range(reps):
        # give some variation to parameters
        new_peaks = [peak * np.random.uniform(0.95, 1.05) for peak in peaks] # plus or minus 5% on snr
        new_coords = [(ra + np.random.uniform(-beam_maj/2, beam_maj/2), dec + np.random.uniform(-beam_maj/2, beam_maj/2)) for ra, dec in coords] # plus or minus half beam major fwhm on position

        info, vis = generate_synthetic_info_vis(fits_file, sources, new_peaks, new_coords, noise, widths, ratios, thetas)

        pts = 0
        result = auto_detect(info, vis, corner_plot=False)[0]

        # try:
        #     result = sim_auto_detect(info, vis, corner_plot=False)[0]
        # except:
        #     print(f"Error occurred while running simulation for rep {i}")
        #     num_errors += 1
        #     continue
        counter = 0
        # gaussians = 0
        for key, res in result['result'].items():
            if key not in all_peaks:
                all_peaks[key] = []
            if key not in all_ras:
                all_ras[key] = []
            if key not in all_decs:
                all_decs[key] = []
            src_type = res['type']
            peak = (float(res['peak'][0][0]), float(res['peak'][0][1]))
            all_peaks[key].append(peak[0]-new_peaks[counter]) # the delta
            ra = (float(res['ra'][0][0]), float(res['ra'][0][1]))
            all_ras[key].append(ra[0]-new_coords[counter][0]) # the delta
            dec = (float(res['dec'][0][0]), float(res['dec'][0][1]))
            all_decs[key].append(dec[0]-new_coords[counter][1]) # the delta
            # if src_type == 'c':
            #     if key not in all_sigmas:
            #         all_sigmas[key] = []
            #     sigma = (float(res['sigma'][0][0]), float(res['sigma'][0][1]))
            #     all_sigmas[key].append(sigma[0])

            # immediately gets -1 point if spurious detection
            significance_threshold = norm.ppf(-1/num_pixels, loc=0, scale=1) # threshold for being a once in image significant detection
            z_peak = (peak[0]-new_peaks[counter]) / noise if peak[1] > 0 else float('inf')
            if (abs(ra[0]-new_coords[counter][0])>5*(beam_maj/2) or abs(dec[0]-new_coords[counter][1])>5*(beam_maj/2)) and z_peak >= significance_threshold: # spurious detection if at least 5 sigma off in position and sigificant peak
                pts = -1
                peak_pts.append(-1/3)
                ra_pts.append(-1/3)
                dec_pts.append(-1/3)
                counter += 1
                continue

            # peak points
            pts += score(z_peak)
            peak_pts.append(score(z_peak))

            # ra points
            z_ra = (ra[0]-new_coords[counter][0]) / beam_maj if ra[1] > 0 else float('inf')
            pts += score(z_ra)
            ra_pts.append(score(z_ra))

            # dec points
            z_dec = (dec[0]-new_coords[counter][1]) / beam_maj if dec[1] > 0 else float('inf')
            pts += score(z_dec)
            dec_pts.append(score(z_dec))

            counter += 1
        all_points.append(pts)

    # histograms of deltas
    if all_peaks:
        peak_keys = list(all_peaks.keys())
        pfig, pax = plt.subplots(nrows=1, ncols=len(peak_keys))
        if len(peak_keys) == 1:
            pax = [pax]
        for i in range(len(peak_keys)):
            key = peak_keys[i]
            pax[i].hist(all_peaks[key], bins=20, color='b', edgecolor='k', weights=np.ones(len(all_peaks[key]))/len(all_peaks[key]))
            pax[i].set_title(f'Peak, {key}')
            pax[i].yaxis.set_major_formatter(PercentFormatter(xmax=1))
            pax[i].set_ylabel('Frequencies')
            pax[i].set_xlabel('Difference from \ntrue value [Jy]')
            pax[i].tick_params(axis='x', labelrotation=45)
            pfig.tight_layout(pad=2.0)
    if all_ras:
        ra_keys = list(all_ras.keys())
        rfig,rax = plt.subplots(nrows=1, ncols=len(ra_keys))
        if len(ra_keys) == 1:
            rax = [rax]
        for i in range(len(ra_keys)):
            key = ra_keys[i]
            rax[i].hist(all_ras[key], bins=20, color='g', edgecolor='k', weights=np.ones(len(all_ras[key]))/len(all_ras[key]))
            rax[i].set_title(f'RA, {key}')
            rax[i].yaxis.set_major_formatter(PercentFormatter(xmax=1))
            rax[i].set_ylabel('Frequencies')
            rax[i].set_xlabel('Difference from \ntrue value [arcsec]')
            rax[i].tick_params(axis='x', labelrotation=45)
            rfig.tight_layout(pad=2.0)
    if all_decs:
        dec_keys = list(all_decs.keys())
        dfig,dax = plt.subplots(nrows=1, ncols=len(dec_keys))
        if len(dec_keys) == 1:
            dax = [dax]
        for i in range(len(dec_keys)):
            key = dec_keys[i]
            dax[i].hist(all_decs[key], bins=20, color='r', edgecolor='k', weights=np.ones(len(all_decs[key]))/len(all_decs[key]))
            dax[i].set_title(f'Dec, {key}')
            dax[i].yaxis.set_major_formatter(PercentFormatter(xmax=1))
            dax[i].set_ylabel('Frequencies')
            dax[i].set_xlabel('Difference from \ntrue value [arcsec]')
            dax[i].tick_params(axis='x', labelrotation=45)
            dfig.tight_layout(pad=2.0)
    # if all_sigmas:
    #     sigma_keys = list(all_sigmas.keys())
    #     sfig,sax = plt.subplots(nrows=1, ncols=len(sigma_keys))
    #     if len(sigma_keys) == 1:
    #         sax = [sax]
    #     for i in range(len(sigma_keys)):
    #         key = sigma_keys[i]
    #         sax[i].hist(all_sigmas[key], bins=20, color='c', edgecolor='k')
    #         sax[i].set_title(f'Sigma, {key}')

    return {'average total points': round(float(np.mean(all_points)),2), 'std total points': round(float(np.std(all_points)),2),\
            'average peak points': round(float(np.mean(peak_pts)),2), 'std peak points': round(float(np.std(peak_pts)),2),\
            'average ra points': round(float(np.mean(ra_pts)),2), 'std ra points': round(float(np.std(ra_pts)),2),\
            'average dec points': round(float(np.mean(dec_pts)),2), 'std dec points': round(float(np.std(dec_pts)),2),\
            'number of errors': num_errors}


def average_by_snr(fits_file, sources, coords, noise, widths=None, ratios=None, thetas=None, reps_per_snr=50, snr_vals=[3,4,5,6,7,8,9,10,20,50,100,200,500,1000]):
    # single source only for now, can be extended to multiple sources in the future
    pixel = 0.214255908880632
    image_edge = pixel * 194
    num_pixels = 194**2
    beam_maj = 1.96
    peaks_by_snr = []
    peak_std_by_snr = []
    ras_by_snr = []
    ra_std_by_snr = []
    decs_by_snr = []
    dec_std_by_snr = []
    # sigmas_by_snr = []
    # sigma_std_by_snr = []
    # snr_vals_for_sigma = []
    points_by_snr = []
    point_std_by_snr = []

    for snr in snr_vals:
        peaks = [snr * noise]

        all_points = []
        all_peaks = {}
        all_ras = {}
        all_decs = {}
        # all_sigmas = {}
        num_errors = 0
        for i in range(reps_per_snr):
            # give some variation to parameters
            coords = [(ra + np.random.uniform(-beam_maj/2, beam_maj/2), dec + np.random.uniform(-beam_maj/2, beam_maj/2)) for ra, dec in coords] # plus or minus half beam major fwhm on position
            info, vis = generate_synthetic_info_vis(fits_file, sources, peaks, coords, noise, widths, ratios, thetas)

            pts = 0
            try:
                result = sim_auto_detect(info, vis, corner_plot=False)[0]
            except:
                print(f"Error occurred while running simulation for rep {i}")
                num_errors += 1
                continue
            counter = 0
            # gaussians = 0
            for key, res in result['result'].items():
                if key not in all_peaks:
                    all_peaks[key] = []
                if key not in all_ras:
                    all_ras[key] = []
                if key not in all_decs:
                    all_decs[key] = []
                src_type = res['type']
                peak = (float(res['peak'][0][0]), float(res['peak'][0][1]))
                all_peaks[key].append(peak[0]-peaks[counter]) # the delta
                ra = (float(res['ra'][0][0]), float(res['ra'][0][1]))
                all_ras[key].append(ra[0]-coords[counter][0]) # the delta
                dec = (float(res['dec'][0][0]), float(res['dec'][0][1]))
                all_decs[key].append(dec[0]-coords[counter][1]) # the delta
                # if src_type == 'c':
                #     if key not in all_sigmas:
                #         all_sigmas[key] = []
                #     sigma = (float(res['sigma'][0][0]), float(res['sigma'][0][1]))
                #     all_sigmas[key].append(sigma[0])

                # immediately gets -1 point if spurious detection
                significance_threshold = norm.ppf(-1/num_pixels, loc=0, scale=1) # threshold for being a once in image significant detection
                z_peak = (peak[0]-peaks[counter]) / noise if peak[1] > 0 else float('inf')
                if (abs(ra[0]-coords[counter][0])>5*(beam_maj/2) or abs(dec[0]-coords[counter][1])>5*(beam_maj/2)) and z_peak >= significance_threshold: # spurious detection if at least 5 sigma off in position and sigificant peak
                    pts = -1
                    counter += 1
                    continue

                # peak points
                pts += score(z_peak)

                # ra points
                z_ra = (ra[0]-coords[counter][0]) / beam_maj if ra[1] > 0 else float('inf')
                pts += score(z_ra)

                # dec points
                z_dec = (dec[0]-coords[counter][1]) / beam_maj if dec[1] > 0 else float('inf')
                pts += score(z_dec)

                counter += 1
            all_points.append(pts)

        if all_peaks:
            peaks_by_snr.append(np.mean(list(all_peaks.values())))
            peak_std_by_snr.append(np.std(list(all_peaks.values())))
        if all_ras:
            ras_by_snr.append(np.mean(list(all_ras.values())))
            ra_std_by_snr.append(np.std(list(all_ras.values())))
        if all_decs:
            decs_by_snr.append(np.mean(list(all_decs.values())))
            dec_std_by_snr.append(np.std(list(all_decs.values())))
        # if all_sigmas:
        #     sigmas_by_snr.append(np.mean(list(all_sigmas.values())))
        #     sigma_std_by_snr.append(np.std(list(all_sigmas.values())))
        #     snr_vals_for_sigma.append(snr)
        points_by_snr.append(round(float(np.mean(all_points)),2))
        point_std_by_snr.append(round(float(np.std(all_points)),2))

    if peaks_by_snr:
        pfig, pax = plt.subplots()
        snr_measured = [peak/noise for peak in peaks_by_snr]
        snr_std = [std/noise for std in peak_std_by_snr]
        pax.errorbar(snr_vals, snr_measured, yerr=snr_std, fmt='o', color='b', ecolor='b', capsize=5)
        pax.set_xscale('log')
        pax.set_title('Difference in SNR vs True SNR')
        pax.set_ylabel('Difference in SNR')
        pax.set_xlabel('True SNR')
    if ras_by_snr:
        rfig, rax = plt.subplots()
        rax.errorbar(snr_vals, ras_by_snr, yerr=ra_std_by_snr, fmt='o', color='g', ecolor='g', capsize=5)
        rax.set_xscale('log')
        rax.set_title('Difference in RA vs SNR')
        rax.set_ylabel('Difference in RA')
        rax.set_xlabel('True SNR')
    if decs_by_snr:
        dfig, dax = plt.subplots()
        dax.errorbar(snr_vals, decs_by_snr, yerr=dec_std_by_snr, fmt='o', color='r', ecolor='r', capsize=5)
        dax.set_xscale('log')
        dax.set_title('Difference in Dec vs SNR')
        dax.set_ylabel('Difference in Dec')
        dax.set_xlabel('True SNR')
    # if sigmas_by_snr:
    #     sfig, sax = plt.subplots()
    #     sax.errorbar(snr_vals_for_sigma, sigmas_by_snr, yerr=sigma_std_by_snr, fmt='o', color='c', ecolor='c', capsize=5)
    #     sax.set_xscale('log')
    #     sax.set_title('Average Sigma vs SNR')
    if points_by_snr:
        ptsfig, ptsax = plt.subplots()
        ptsax.errorbar(snr_vals, points_by_snr, yerr=point_std_by_snr, fmt='o', color='k', ecolor='k', capsize=5)
        ptsax.set_xscale('log')
        ptsax.set_title('Average Points vs SNR')
        ptsax.set_ylabel('Average Points')
        ptsax.set_xlabel('True SNR')

def average_by_width(fits_file, sources, coords, peaks, noise, width_vals=[0.1,0.2,0.5,1,1.5,2,3,4,5], ratios=None, thetas=None, reps_per_width=50):
    # single source only for now, can be extended to multiple sources in the future
    pixel = 0.214255908880632
    image_edge = pixel * 194
    beam_maj = 1.96
    peaks_by_width = []
    peak_std_by_width = []
    ras_by_width = []
    ra_std_by_width = []
    decs_by_width = []
    dec_std_by_width = []
    sigmas_by_width = []
    sigma_std_by_width = []
    width_vals_for_sigma = []
    points_by_width = []
    point_std_by_width = []

    for w in width_vals:
        widths = [w]
        info, vis = generate_synthetic_info_vis(fits_file, sources, peaks, coords, noise, widths, ratios, thetas)

        all_points = []
        all_peaks = {}
        all_ras = {}
        all_decs = {}
        all_sigmas = {}
        num_errors = 0
        for i in range(reps_per_width):
            pts = 0
            result = sim_auto_detect(info, vis, corner_plot=False)[0]
            try:
                result = sim_auto_detect(info, vis, corner_plot=False)[0]
            except:
                print(f"Error occurred while running simulation for rep {i}")
                num_errors += 1
                continue
            counter = 0
            gaussians = 0
            for key, res in result['result'].items():
                if key not in all_peaks:
                    all_peaks[key] = []
                if key not in all_ras:
                    all_ras[key] = []
                if key not in all_decs:
                    all_decs[key] = []
                src_type = res['type']
                peak = (float(res['peak'][0][0]), float(res['peak'][0][1]))
                all_peaks[key].append(peak[0])
                ra = (float(res['ra'][0][0]), float(res['ra'][0][1]))
                all_ras[key].append(ra[0])
                dec = (float(res['dec'][0][0]), float(res['dec'][0][1]))
                all_decs[key].append(dec[0])
                if src_type == 'c':
                    if key not in all_sigmas:
                        all_sigmas[key] = []
                    sigma = (float(res['sigma'][0][0]), float(res['sigma'][0][1]))
                    all_sigmas[key].append(sigma[0])
                if peak[0] - 3*peak[1] > noise and abs(ra[0]) <= image_edge and abs(dec[0]) <= image_edge: # significant detection and within image bounds
                    pts += 1
                if np.sqrt((ra[0]-coords[counter][0])**2 + (dec[0]-coords[counter][1])**2) < beam_maj and abs(peak[0]-peaks[counter]) < 3*noise:
                    if src_type == 'c':
                        if widths is not None:
                            if gaussians <= len(widths):
                                if abs(sigma[0] - widths[gaussians]) < 3*noise:
                                    pts += 1
                                gaussians += 1
                    else:
                        pts += 1
                counter += 1
            all_points.append(pts)

        if all_peaks:
            peaks_by_width.append(np.mean(list(all_peaks.values())))
            peak_std_by_width.append(np.std(list(all_peaks.values())))
        if all_ras:
            ras_by_width.append(np.mean(list(all_ras.values())))
            ra_std_by_width.append(np.std(list(all_ras.values())))
        if all_decs:
            decs_by_width.append(np.mean(list(all_decs.values())))
            dec_std_by_width.append(np.std(list(all_decs.values())))
        if all_sigmas:
            sigmas_by_width.append(np.mean(list(all_sigmas.values())))
            sigma_std_by_width.append(np.std(list(all_sigmas.values())))
            width_vals_for_sigma.append(w)
        points_by_width.append(round(float(np.mean(all_points)),2))
        point_std_by_width.append(round(float(np.std(all_points)),2))

    if peaks_by_width:
        pfig, pax = plt.subplots()
        pax.errorbar(width_vals, peaks_by_width, yerr=peak_std_by_width, fmt='o', color='b', ecolor='b', capsize=5)
        pax.set_title('Measured Peaks vs True Width')
    if ras_by_width:
        rfig, rax = plt.subplots()
        rax.errorbar(width_vals, ras_by_width, yerr=ra_std_by_width, fmt='o', color='g', ecolor='g', capsize=5)
        rax.set_title('Average RA vs Width')
    if decs_by_width:
        dfig, dax = plt.subplots()
        dax.errorbar(width_vals, decs_by_width, yerr=dec_std_by_width, fmt='o', color='r', ecolor='r', capsize=5)
        dax.set_title('Average Dec vs Width')
    if sigmas_by_width:
        sfig, sax = plt.subplots()
        sax.errorbar(width_vals_for_sigma, sigmas_by_width, yerr=sigma_std_by_width, fmt='o', color='c', ecolor='c', capsize=5)
        sax.set_title('Average Sigma vs Width')
    if points_by_width:
        ptsfig, ptsax = plt.subplots()
        ptsax.errorbar(width_vals, points_by_width, yerr=point_std_by_width, fmt='o', color='k', ecolor='k', capsize=5)
        ptsax.set_title('Average Points vs Width')
