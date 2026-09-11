from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
import glob
import pandas as pd
import os
import math
import sqlite3
import io
import re
import warnings
from PIL import Image
from pathlib import Path
from find_source import summary, _fits_data_index


def _interpolation_kernel(s: float):
    """Evaluate the Keys cubic convolution interpolation kernel.

    Parameters
    ----------
    s : float
        Dimensionless distance from the interpolation node.

    Returns
    -------
    float
        Value of the interpolation kernel at `s`.

    References
    ----------
    .. [1] R. G. Keys, "Cubic Convolution Interpolation for Digital Image
       Processing," IEEE Transactions on Acoustics, Speech, and Signal
       Processing, 29(6), 1153-1160, 1981.
    """
    dist = abs(s)
    if dist < 1:
        return 3 / 2 * dist**3 - 5 / 2 * dist**2 + 1
    if dist < 2:
        return -1 / 2 * dist**3 + 5 / 2 * dist**2 - 4 * dist + 2
    return 0


def _interpolation_function(
    x: float,
    node_x: list,
    node_val: list,
) -> float:
    """Interpolate values using Keys cubic convolution interpolation.

    Parameters
    ----------
    x : float
        Position at which to evaluate the interpolation.
    node_x : list
        Positions of the interpolation nodes. The nodes must be uniformly
        spaced and contain at least three elements.
    node_val : list
        Values associated with `node_x`. Must have the same length as
        `node_x`.

    Returns
    -------
    float
        Interpolated value at `x`.

    Raises
    ------
    ValueError
        If fewer than three elements are provided in `node_x`; `node_x` and
        `node_val` have different lengths; or if `node_x` is not uniformly
        spaced.

    Notes
    -----
    Third-order boundary conditions derived by Keys [1] are used to determine
    the coefficients associated with the two virtual nodes immediately outside
    the interpolation domain.

    References
    ----------
    .. [1] R. G. Keys, "Cubic Convolution Interpolation for Digital Image
       Processing," IEEE Transactions on Acoustics, Speech, and Signal
       Processing, 29(6), 1153-1160, 1981.
    """
    num_nodes = len(node_x)
    N = num_nodes - 1

    if len(node_val) != num_nodes:
        raise ValueError(
            f"The number of nodes given ({num_nodes}) does not match the "
            f"number of node values given ({len(node_val)})."
        )
    if num_nodes < 3:
        raise ValueError(
            f"At least 3 interpolation nodes are required. Got {num_nodes}."
        )
    h = node_x[1] - node_x[0]
    if not np.allclose(np.diff(node_x), h):
        raise ValueError("Nodes are not uniformly spaced.")

    # Apply the third-order boundary conditions derived by Keys (1981).
    node_neg1 = node_x[0] - h
    node_Nplus1 = node_x[N] + h
    c_neg1 = node_val[2] - 3 * node_val[1] + 3 * node_val[0]
    c_Nplus1 = 3*node_val[N] - 3 * node_val[N-1] + 3 * node_val[N-2]

    interpolated_val = c_neg1 * _interpolation_kernel((x - node_neg1) / h)
    for k in range(num_nodes):
        s = (x - node_x[k]) / h
        interpolated_val += node_val[k] * _interpolation_kernel(s)
    interpolated_val += c_Nplus1 * _interpolation_kernel((x - node_Nplus1) / h)

    return float(interpolated_val)


def thumbnail(
    fits_file: str | Path,
    peak_coord: tuple,
    radius_buffer: float,
    pts_bw_nodes: int = 4,
) -> bytes:
    """Create a Keys cubic convolution interpolated image thumbnail.

    A region centered on the specified source position is extracted from the
    FITS image and interpolated using Keys cubic convolution interpolation.
    The region extends approximately one restoring-beam major axis plus
    `radius_buffer` from the source position.

    Parameters
    ----------
    fits_file : str | Path
        The path of the FITS file that contains the image.
    peak_coord : tuple
        A tuple `(x, y)` containing the x- and y-coordinates of the source, in
        arcseconds relative to the field center.
    radius_buffer : float
        Additional radius, in arcsec, added to the beam major-axis FWHM to
        determine the extent of the image region.
    pts_bw_nodes : int, optional
        Number of interpolated points generated between each pair of
        adjacent image pixels. Must be non-negative, by default 4.

    Returns
    -------
    bytes
        PNG image bytes containing the interpolated thumbnail.

    Raises
    ------
    ValueError
        If `pts_bw_nodes` is negative, if FITS image data is not a 3D array
        containing a 2D image, or if the selected image region is empty.

    References
    ----------
    .. [1] R. G. Keys, "Cubic Convolution Interpolation for Digital Image
       Processing," IEEE Transactions on Acoustics, Speech, and Signal
       Processing, 29(6), 1153-1160, 1981.
    """
    if pts_bw_nodes < 0:
        raise ValueError(
            "pts_bw_nodes must be non-negative. "
            f"Got {pts_bw_nodes}."
        )

    fits_file = Path(fits_file)
    hdu_index = _fits_data_index(fits_file)

    # Extract image data array and header from FITS file.
    with fits.open(fits_file) as hdulist:
        image_hdu = hdulist[hdu_index]
        data = image_hdu.data
        header_data = image_hdu.header

    if data.ndim != 3 or data.shape[0] != 1:
        raise ValueError(
            "FITS image data must be a 3D array containing a 2D image."
        )
    data_array = data[0]

    min_flux = np.min(data_array)
    max_flux = np.max(data_array)

    beam_maj = Angle(
        header_data['BMAJ'],
        header_data['CUNIT1']
    ).to_value('arcsec')
    pixel_scale = Angle(
        abs(header_data['CDELT1']),
        header_data['CUNIT1']
    ).to_value('arcsec')
    x_dim = header_data['NAXIS1']
    y_dim = header_data['NAXIS2']

    center = ((x_dim - 1) / 2, (y_dim - 1) / 2)

    # Convert source offsets from arcsec to absolute pixel coordinates.
    abs_x = round((peak_coord[0] / pixel_scale) + center[0])
    abs_y = round((peak_coord[1] / pixel_scale) + center[1])

    # Convert the search radius from arcsec to pixels, rounded up.
    delta = math.ceil((radius_buffer + beam_maj) / pixel_scale)

    # Restrict the image to a box centered on the source, clipped to the
    # original image boundaries.
    y_start = max(abs_y - delta, 0)
    y_stop = min(abs_y + delta, y_dim)
    x_start = max(abs_x - delta, 0)
    x_stop = min(abs_x + delta, x_dim)

    new_data = data_array[y_start:y_stop, x_start:x_stop]

    y_length, x_length = new_data.shape
    if y_length == 0 or x_length == 0:
        raise ValueError(
            "Attempts to obtain a smaller image centered on the source "
            "resulted in an empty data array."
        )
    node_x = np.arange(0, x_length).tolist()
    node_y = np.arange(0, y_length).tolist()

    interpolated_data = []

    # Apply Keys cubic convolution interpolation in the x-direction.
    pts_spacing = 1 / (pts_bw_nodes + 1)
    for row_num in range(y_length):
        temp = new_data[row_num].tolist()
        for i in range(x_length - 1):
            temp2 = []
            for j in range(1, pts_bw_nodes + 1):
                x = i + j * pts_spacing
                temp2.append(
                    _interpolation_function(
                        x,
                        node_x=node_x,
                        node_val=new_data[row_num]
                    )
                )
            # Insert the interpolated points while preserving the original
            # node values at their corresponding positions.
            temp = temp[:i - x_length + 1] + temp2 + temp[i - x_length + 1:]
        interpolated_data.append(temp)

    # Apply Keys cubic convolution interpolation in the y-direction.
    new_x_length = len(interpolated_data[0])
    temp = []
    for i in range((y_length - 1) * pts_bw_nodes):
        temp.append([])
    for col_num in range(new_x_length):
        temp2 = [row[col_num] for row in interpolated_data]
        for i in range(y_length - 1):
            for j in range(1, pts_bw_nodes + 1):
                y = i + j * pts_spacing
                temp[i * pts_bw_nodes + j - 1].append(
                    _interpolation_function(
                        y,
                        node_x=node_y,
                        node_val=temp2
                    )
                )
    for i in range(y_length - 1):
        interpolated_data = (
            interpolated_data[:i - y_length + 1]
            + temp[:pts_bw_nodes]
            + interpolated_data[i-y_length+1:]
        )
        temp = temp[pts_bw_nodes:]

    fig, ax = plt.subplots()
    ax.axis("off")
    ax.set_title(
        "Arcsec offset from center: {},\n pixel scale: {} arcsec".format(
            (round(peak_coord[0], 2), round(peak_coord[1], 2)),
            round(pixel_scale, 2)
        )
    )
    img = ax.imshow(interpolated_data, vmin=min_flux, vmax=max_flux)
    fig.colorbar(img)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = buffer.read()
    plt.close(fig)

    return plot_data


def _format_coordinates(ra: Angle, dec: Angle) -> tuple[str, str]:
    """Format sky coordinates as right ascension and declination strings.

    Parameters
    ----------
    ra : astropy.coordinates.Angle
        Right Ascension to format.
    dec : astropy.coordinates.Angle
        Declination to format.

    Returns
    -------
    tuple of str
        Formatted Right Ascension and Declination strings. Seconds are rounded
        to two decimal places.
    """
    hms_ra = ra.hms
    dms_dec = dec.dms

    str_ra = (
        f'{int(hms_ra.h)}h{abs(int(hms_ra.m))}'
        f'm{round(abs(hms_ra.s), 2)}s'
    )
    str_dec = (
        f'{int(dms_dec.d)}d{abs(int(dms_dec.m))}m'
        f'{round(abs(dms_dec.s), 2)}s'
    )

    return str_ra, str_dec


def make_catalog(
    fits_file: str | Path,
    threshold: float = 0.01,
    radius_buffer: float = 5.0,
    ext_threshold: float | None = None,
) -> dict | None:
    """
    Create a catalog of significant point sources detected in a FITS image.

    Parameters
    ----------
    fits_file : str | Path
        The path of the FITS file that contains the image.
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
        Dictionary with keys of the form `Source1`, `Source2`, etc., labeling
        the significant internal and external sources found in the image. Each
        key is associated with a dictionary with the following keys:
        `FieldName` : str
            The name of the target object of the observation.
        `ObsDateTime` : str
            The date and time of the observation, in the format M-d-yy h:m:s.
        `Stationary` : bool
            Whether the source can be approximated as stationary.
        `FileName` : str
            The name of the FITS file that contains the image.
        `BeamMajAxis_arcsec` : float
            The restoring beam major axis, in arcsec, rounded to 3 decimal
            places.
        `BeamMinAxis_arcsec` : float
            The restoring beam minor axis, in arcsec, rounded to 3 decimal
            places.
        `BeamPosAngle_deg` : float
            The restoring beam position angle, in degrees, rounded to 3 decimal
            places.
        `Freq_GHz` : float | str
            The frequency at which the image data was recorded, in GHz, rounded
            to 3 decimal places, or `Not found` if the frequency cannot be
            determined.
        `FluxUncert_mJy` : float
            The uncertainty in flux density measurements, in mJy, rounded to 3
            decimal places.
        `Flux_mJy` : float
            The flux density of the source, in mJy, rounded to 3 decimal
            places.
        `RAUncert_arcsec` : float
            The uncertainty in Right Ascension of the source, in arcsec,
            rounded to 3 decimal places.
        `DecUncert_arcsec` : float
            The uncertainty in the Declination of the source, in arcsec,
            rounded to 3 decimal places.
        `RA` : str
            The Right Ascension of the source, in the format {h}h{m}m{s}s,
            where the seconds are rounded to 2 decimal places.
        `Dec` : str
            The Declination of the source, in the format {d}d{m}m{s}s, where
            the seconds are rounded to 2 decimal places.
        `Internal` : bool
            Whether the detected point source is in the initial search region.
        `Image` : bytes
            PNG image bytes containing the interpolated thumbnail.
    None
        If no significant sources are found.

    Raises
    ------
    ValueError
        If `CTYPE1` and `CTYPE2` do not describe one Right Ascension axis and
        one Declination axis, or if the `CUNIT1` and `CUNIT2` are not equal.

    Notes
    -----
    The restoring beam major and minor axes are assumed to use the angular
    units specified by `CUNIT1` and `CUNIT2`, respectively. The Right Ascension
    and Declination axes are required to have the same units. The first two
    axes are assumed to describe Right Ascension and Declination.

    `FluxUncert_mJy` is taken from the most conservative RMS value in
    `summary()`.

    `RAUncert_arcsec` and `DecUncert_arcsec` are estimated from the restoring
    beam dimensions and source SNR, with the major and minor contributions
    projected onto the Right Ascension and Declination axes according to the
    restoring beam position angle.
    """
    fits_file = Path(fits_file)

    summ = summary(
        fits_file=fits_file,
        radius_buffer=radius_buffer,
        ext_threshold=ext_threshold,
        silence_dict=False,
        plot=False
    )

    header_data = fits.getheader(fits_file)
    name = header_data['OBJECT']
    obs_date_time = header_data['DATE-OBS']
    bmaj = header_data['BMAJ']
    bmin = header_data['BMIN']
    bpa = header_data['BPA']
    ctype1 = header_data['CTYPE1'].upper()
    crval1 = header_data['CRVAL1']
    cunit1 = header_data['CUNIT1']
    ctype2 = header_data['CTYPE2'].upper()
    crval2 = header_data['CRVAL2']
    cunit2 = header_data['CUNIT2']
    ctype3 = header_data['CTYPE3'].upper()
    crval3 = header_data['CRVAL3']
    cunit3 = header_data['CUNIT3'].lower()

    if cunit1 != cunit2:
        raise ValueError("Axes have different units.")

    # Support frequency information stored either directly in the image header
    # or in a channel table extension.
    freq = 'Not found'
    if ctype3 == 'FREQ':
        if cunit3 == 'ghz':
            freq = round(crval3, 3)
        elif cunit3 == 'hz':
            freq = round(crval3 / 1e9, 3)  # Convert to GHz.
    elif ctype3 == 'CHANNUM':
        with fits.open(fits_file) as hdul:
            freq_col = hdul[1].columns[1]
            freq_unit = freq_col.unit.lower()
            if freq_col.name == 'Freq':
                if freq_unit == 'hz':
                    freq = round(hdul[1].data[0][1] / 1e9, 3)  # Convert to GHz.
                elif freq_unit == 'ghz':
                    freq = round(hdul[1].data[0][1], 3)

    # Interpret the beam axes using the corresponding celestial-axis units.
    beam_maj_axis = Angle(bmaj, cunit1)
    beam_min_axis = Angle(bmin, cunit2)
    bpa_rad = math.radians(bpa)

    # Solar System bodies require moving-source treatment and are not treated as
    # stationary targets.
    moving_objects = [
        'venus',
        'mars',
        'jupiter',
        'uranus',
        'neptune',
        'io',
        'europa',
        'ganymede',
        'callisto',
        'titan',
        'ceres',
        'vesta',
        'pallas',
        'juno'
    ]

    name_lower = name.lower()
    stationary = not any(
        re.search(rf"\b{re.escape(obj)}\b", name_lower)
        for obj in moving_objects
    )

    interesting_sources = {}
    field_info = {
        'FieldName': name,
        'ObsDateTime': obs_date_time,
        'FileName': fits_file.name,
        'Stationary': stationary,
        'BeamMajAxis_arcsec': round(
            float(beam_maj_axis.to(u.arcsec).value), 3
        ),
        'BeamMinAxis_arcsec': round(
            float(beam_min_axis.to(u.arcsec).value), 3
        ),
        'BeamPosAngle_deg': round(bpa, 3),
        'Freq_GHz': freq
    }

    field_info['FluxUncert_mJy'] = round(summ['conservative_rms'] * 1e3, 3)

    n_int_sources = len(summ['int_peak_val'])

    # A string value indicates that no significant external peaks were found.
    if isinstance(summ['ext_peak_val'], str):
        n_ext_sources = 0
    else:
        n_ext_sources = len(summ['ext_peak_val'])

    ra_index = 0
    dec_index = 1

    if 'RA' in ctype1:
        ra = crval1
    elif 'RA' in ctype2:
        ra = crval2
        ra_index = 1
    else:
        raise ValueError("No RA in image.")

    if 'DEC' in ctype1:
        dec = crval1
        dec_index = 0
    elif 'DEC' in ctype2:
        dec = crval2
    else:
        raise ValueError("No dec in image.")

    center = SkyCoord(ra, dec, unit=cunit1)

    def create_source_info(
        peak_value,
        peak_coord,
        internal,
    ):
        """Create catalog metadata and measurements for a detected source.

        Parameters
        ----------
        peak_value : float
            Peak flux density of the detected source.
        peak_coord : tuple
            Source coordinates as offsets from the image center, in arcsec.
        internal : bool
            Whether the source was detected in the initial internal search
            region.

        Returns
        -------
        dict
            Catalog information for the detected source, including its flux,
            estimated positional uncertainties, sky coordinates, and
            interpolated image thumbnail.
        """
        info = field_info.copy()

        info['Flux_mJy'] = round(peak_value * 1000, 3)

        # Estimate positional uncertainties from the restoring beam and source
        # SNR.
        snr = peak_value / summ['conservative_rms']
        bmin_uncert = float(beam_maj_axis.to(u.arcsec).value / snr)
        bmaj_uncert = float(beam_min_axis.to(u.arcsec).value / snr)

        info['RAUncert_arcsec'] = round(
            bmin_uncert * abs(math.sin(bpa_rad))
            + bmaj_uncert * abs(math.cos(bpa_rad)),
            3
        )
        info['DecUncert_arcsec'] = round(
            bmaj_uncert * abs(math.sin(bpa_rad))
            + bmin_uncert * abs(math.cos(bpa_rad)),
            3
        )

        ra_offset = peak_coord[ra_index] * u.arcsec
        dec_offset = peak_coord[dec_index] * u.arcsec
        coord = center.spherical_offsets_by(ra_offset, dec_offset)

        info['RA'], info['Dec'] = _format_coordinates(coord.ra, coord.dec)

        info['Internal'] = internal

        info['Image'] = thumbnail(
            fits_file=fits_file,
            peak_coord=peak_coord,
            radius_buffer=radius_buffer,
            pts_bw_nodes=4,
        )

        return info

    pt_source_count = 1
    # Create a catalog dictionary for each significant internal source.
    for i in range(n_int_sources):
        if (
            summ['int_exp_exceed'][i] < threshold
            and summ['calc_int_exp_exceed'][i] < threshold
        ):
            key = f'Source{pt_source_count}'
            interesting_sources[key] = create_source_info(
                summ['int_peak_val'][i],
                summ['int_peak_coord'][i],
                internal=True,
            )
            pt_source_count += 1

    # Create a catalog dictionary for each external source.
    for i in range(n_ext_sources):
        key = f'Source{pt_source_count}'
        interesting_sources[key] = create_source_info(
            summ['ext_peak_val'][i],
            summ['ext_peak_coord'][i],
            internal=False,
        )
        pt_source_count += 1

    if not interesting_sources:
        return None

    return interesting_sources


def combine_catalogs(
    catalog_1: dict,
    catalog_2: dict,
) -> dict:
    """
    Combine two catalogs returned by `make_catalog()`.

    Entries from `catalog_2` are added to `catalog_1` with their source
    numbers shifted to follow the entries already present in `catalog_1`.
    `catalog_1` is modified in place.

    Parameters
    ----------
    catalog_1 : dict
        Catalog to which the entries from `catalog_2` are added. This
        dictionary is modified in place.
    catalog_2 : dict
        Catalog whose entries are added to `catalog_1`.

    Returns
    -------
    dict
        The combined catalog. This is the same dictionary object as
        `catalog_1`.

    Notes
    -----
    Catalog keys are expected to have the form `SourceN`, where `N` is an
    integer source number.
    """
    shift = len(catalog_1)
    for key, value in catalog_2.items():
        new_number = int(key.replace('Source', ''))
        new_key = f'Source{new_number + shift}'
        catalog_1[new_key] = value
    return catalog_1


def low_level_table(
    folder: str,
    db_path: str = '../sources.db',
) -> None:
    """Create a SQLite table containing information about significant point
    sources detected in images in a folder.

    Parameters
    ----------
    folder : str
        Path to the folder containing FITS images.
    db_path : str | Path, optional
        Path to the SQLite database, by default '../sources.db'.

    Raises
    ------
    ValueError
        If an error occurs when attempting to add information to the
        `low_level` table.

    Warns
    -----
    UserWarning
        If the observation ID cannot be determined from the folder name or old
        data cannot be removed from the database.

    Notes
    -----
    Existing `low_level` records for the observation are removed before the
    newly generated catalog entries are inserted. If the observation ID
    cannot be determined, existing records are not removed.
    """
    db_path = Path(db_path)
    big_catalog = None

    # Retrieve numerical SMA observation ID from folder name.
    try:
        str_obs_id = folder.replace('/mnt/COMPASS9/sma/quality/', '')
        obs_id = str_obs_id.replace('/', '')
        obs_id = int(obs_id)
    except ValueError:
        obs_id = 'Unknown'
        warnings.warn(
            f"Error with obsID. Old/outdated data may not be deleted."
        )

    if os.path.exists(db_path) and obs_id != 'Unknown':
        # Remove existing entries for this observation before inserting new
        # data.
        try:
            with sqlite3.connect(db_path) as con:
                con.execute(
                    "DELETE FROM low_level WHERE ObsID = ?",
                    (obs_id,),
                )
        except sqlite3.Error as e:
            warnings.warn(
                f'Error removing old/outdated data from table "low_level" '
                f"at {db_path}: {e}."
            )

    # Combine source catalogs from all FITS files in the folder.
    for file in glob.glob(os.path.join(folder, '*.fits')):
        try:
            catalog = make_catalog(file)
            if catalog is not None:
                for value in catalog.values():
                    value['ObsID'] = obs_id
                    value['SourceID'] = 'Unknown'
                if big_catalog is None:
                    big_catalog = catalog
                else:
                    big_catalog = combine_catalogs(big_catalog, catalog)
        except Exception as e:
            warnings.warn(f"Unable to create catalog for {file}: {e}.")

    if big_catalog is not None:
        df = pd.DataFrame.from_dict(big_catalog)
        df = df.T

        # Normalize timestamps whose rounded seconds field is 60 so they can
        # be parsed by datetime.
        date_times = df['ObsDateTime'].tolist()
        df.drop(columns='ObsDateTime', inplace=True)
        for i in range(len(date_times)):
            dt = date_times[i]
            m_end = dt.rindex(':')
            s_start = m_end + 1
            if dt[s_start:] == '60':
                dt = dt[:s_start] + '0'
                fmt = '%m-%d-%y %H:%M'
                date_times[i] = (
                    datetime.strptime(dt[:m_end], fmt) + timedelta(minutes=1)
                ).strftime('%m-%d-%y %H:%M:%S')
        df['ObsDateTime'] = date_times

        # Append the combined catalog to the low_level table.
        dtype = {
            'FieldName': 'TEXT',
            'FileName': 'TEXT',
            'Stationary': 'INTEGER',
            'BeamMajAxis_arcsec': 'REAL',
            'BeamMinAxis_arcsec': 'REAL',
            'BeamPosAngle_deg': 'REAL',
            'Freq_GHz': 'REAL',
            'FluxUncert_mJy': 'REAL',
            'Flux_mJy': 'REAL',
            'RAUncert_arcsec': 'REAL',
            'DecUncert_arcsec': 'REAL',
            'RA': 'TEXT',
            'Dec': 'TEXT',
            'Internal': 'INTEGER',
            'Image': 'BLOB',
            'ObsID': 'TEXT',
            'SourceID': 'TEXT',
            'ObsDateTime': 'TEXT',
        }
        try:
            with sqlite3.connect(db_path) as con:
                df.to_sql(
                    "low_level",
                    con=con,
                    if_exists='append',
                    index=False,
                    dtype=dtype,
                )
        except Exception as e:
            raise ValueError(
                f'Error adding to table "low_level" at {db_path}.'
            ) from e


def high_level_table(
    db_path: str | Path = '../sources.db',
    ambiguity_threshold: float = 0.7,
):
    """Create or update source-level tables in a SQLite database.

    Parameters
    ----------
    db_path : str | Path, optional
        Path to the SQLite database, by default '../sources.db'.
    ambiguity_threshold : float, optional
        Minimum proportion of observations that must be consistent with a
        common average position for two source groups to be recorded as
        ambiguous ties, by default 0.7.

    Returns
    -------
    None
        This function updates the `low_level` and `high_level` tables in
        `db_path` and does not return a value.

    Raises
    ------
    FileNotFoundError
        If `db_path` does not exist.
    ValueError
        If the `low_level` table is empty or if the database tables cannot be
        updated.

    Warns
    -----
    UserWarning
        If the `low_level` table cannot be read.

    Notes
    -----
    The `high_level` table represents unique sources by combining detections
    from multiple observations in the `low_level` table. The existing
    `low_level` and `high_level` tables are replaced with the updated tables.

    Sources are initially matched using their coordinates and restoring-beam
    major-axis sizes, followed by a refinement procedure that tests whether
    observations associated with different preliminary source IDs are
    consistent with a common average position.

    Two preliminary source groups are considered to represent the same source
    when all of their combined observations lie within half of the geometric
    mean beam FWHM of their mean position. If more than `ambiguity_threshold`
    of the observations satisfy this criterion, but not all observations do,
    the source groups are recorded as ambiguous ties.

    The default threshold of 70% is heuristic and was selected empirically.
    """
    db_path = Path(db_path)

    unique_sources = None

    if db_path.exists():
        # Get all rows from low level and high level tables, if they exist.
        with sqlite3.connect(db_path) as con:
            low_df = pd.read_sql_query("SELECT * FROM low_level;", con)
            if low_df.empty:
                raise ValueError('Table "low_level" is empty')
        try:
            with sqlite3.connect(db_path) as con:
                unique_sources = pd.read_sql_query(
                    "SELECT * FROM high_level;", con
                ).to_dict(orient='list')
        # Do not raise error if high_level table does not yet exist.
        except pd.errors.DatabaseError:
            pass
    else:
        raise FileNotFoundError(f"Path {db_path} not found.")

    # Treat sources as coarsely matched when their separation is no greater
    # than the geometric mean of their beam major-axis FWHM values.
    for row in range(len(low_df)):
        # Skip rows that already have a source assignment.
        if low_df['SourceID'].iloc[row] == 'Unknown':
            # Source coordinate comparison only makes sense for approximately
            # stationary sources.
            if low_df['Stationary'].iloc[row]:
                if unique_sources is not None:
                    ra = low_df['RA'].iloc[row]
                    dec = low_df['Dec'].iloc[row]
                    coord1 = SkyCoord(ra, dec)
                    fwhm = low_df['BeamMajAxis_arcsec'].iloc[row]
                    source_ids = unique_sources['SourceID']
                    matched = False
                    for i, source_id in enumerate(source_ids):
                        coord2 = SkyCoord(
                            unique_sources['RA'][i],
                            unique_sources['Dec'][i]
                        )
                        sep = coord1.separation(coord2)
                        fwhm2_val = float(unique_sources['FWHM_arcsec'][i])
                        max_sep = (fwhm * fwhm2_val)**(1 / 2) * u.arcsec
                        matched = (sep <= max_sep)
                        if matched:
                            low_df.loc[row, 'SourceID'] = source_id
                            break
                    # Assign lowest available source ID number to unmatched
                    # source.
                    if not matched:
                        num = 1
                        id_nums = [
                            int(source_id.replace('id', ''))
                            for source_id in unique_sources['SourceID']
                        ]
                        while num in id_nums:
                            num += 1
                        next_id = f'id{num:04d}'
                        source_ids.append(next_id)
                        unique_sources['RA'].append(ra)
                        unique_sources['Dec'].append(dec)
                        unique_sources['FWHM_arcsec'].append(fwhm)
                        low_df.loc[row, 'SourceID'] = next_id
                        unique_sources['AmbiguousTies'].append('Unknown')
                # If no unique source has already been determined, this source
                # is automatically the first unique source.
                else:
                    ra = low_df['RA'].iloc[row]
                    dec = low_df['Dec'].iloc[row]
                    fwhm = low_df['BeamMajAxis_arcsec'].iloc[row]
                    unique_sources = {
                        'SourceID': ['id0001'],
                        'RA': [ra],
                        'Dec': [dec],
                        'FWHM_arcsec': [fwhm],
                        'AmbiguousTies': ['Unknown']
                    }
                    low_df.loc[row, 'SourceID'] = 'id0001'
            else:
                low_df.loc[row, 'SourceID'] = 'Not Stationary'

    # Refine coarse source groups by testing whether their combined
    # observations are consistently represented by a common average position.
    new_sources = {
        key: value.copy()
        for key, value in unique_sources.items()
    }
    refined = []
    to_skip = []

    def _average_coordinates(ra, dec):
        """Calculate the mean sky position using Cartesian unit vectors.

        Parameters
        ----------
        ra : sequence of astropy.coordinates.Angle
            Right Ascension values.
        dec : sequence of astropy.coordinates.Angle
            Declination values.

        Returns
        -------
        tuple of astropy.coordinates.Angle
            Mean Right Ascension and Declination.

        Notes
        -----
        Averaging Cartesian unit vectors avoids problems associated with the
        wrap-around of Right Ascension at 0/360 degrees.
        """
        coords = SkyCoord(ra=ra, dec=dec)
        xyz = coords.cartesian.xyz

        mean_xyz = xyz.mean(axis=1)

        mean_coord = SkyCoord(
            x=mean_xyz[0],
            y=mean_xyz[1],
            z=mean_xyz[2],
            representation_type='cartesian',
        )

        return mean_coord.ra, mean_coord.dec

    for i, source_id in enumerate(unique_sources['SourceID']):
        temp_df = low_df[(low_df['SourceID']) == source_id]
        ra_list = [Angle(ra, u.deg) for ra in temp_df['RA']]
        dec_list = [Angle(dec, u.deg) for dec in temp_df['Dec']]
        fwhm_list = [
            Angle(fwhm, u.arcsec) for fwhm in temp_df['BeamMajAxis_arcsec']
        ]
        if len(unique_sources['SourceID']) > 1 and i not in to_skip:
            # Compare ith to every source (which has not been marked as a
            # source to skip) that comes after it in unique_sources.
            for j in range(i + 1, len(unique_sources['SourceID'])):
                if j not in to_skip:
                    temp_df2 = low_df[
                        (low_df['SourceID']) == unique_sources['SourceID'][j]
                    ]
                    ra_list2 = [Angle(ra, u.deg) for ra in temp_df2['RA']]
                    dec_list2 = [Angle(dec, u.deg) for dec in temp_df2['Dec']]
                    fwhm_list2 = [
                        Angle(fwhm, u.arcsec)
                        for fwhm in temp_df2['BeamMajAxis_arcsec']
                    ]
                    new_ra_list = ra_list + ra_list2
                    new_dec_list = dec_list + dec_list2
                    new_fwhm_list = fwhm_list + fwhm_list2
                    num_pts = len(new_ra_list)
                    avg_ra, avg_dec = _average_coordinates(
                        new_ra_list,
                        new_dec_list,
                    )
                    # Use the geometric mean of the beam FWHMs as the
                    # characteristic matching scale.
                    geo_avg_fwhm = math.prod(new_fwhm_list) ** (1/num_pts)
                    avg_pt = SkyCoord(avg_ra, avg_dec)
                    temp = 0
                    for pt in range(num_pts):
                        sep = avg_pt.separation(
                            SkyCoord(new_ra_list[pt], new_dec_list[pt])
                        )
                        if sep > geo_avg_fwhm / 2:
                            temp += 1
                    proportion = (num_pts - temp) / (num_pts)
                    # If the average point is a good representative for all
                    # points, then all these sources will be considered the
                    # same source.
                    if proportion == 1:
                        refined.append(new_sources['SourceID'][i])
                        new_sources['RA'][i], new_sources['Dec'][i] = (
                            _format_coordinates(avg_ra, avg_dec)
                        )
                        new_sources['FWHM_arcsec'][i] = round(
                            geo_avg_fwhm.value, 3
                        )
                        # Remove merged source IDs from existing ambiguity
                        # records.
                        for k in range(len(unique_sources['SourceID'])):
                            unique_sources['AmbiguousTies'][k] = (
                                unique_sources['AmbiguousTies'][k].replace(
                                    unique_sources['SourceID'][j], ''
                                )
                            )
                            unique_sources['AmbiguousTies'][k] = (
                                unique_sources['AmbiguousTies'][k].replace(
                                    '__', '_'
                                )
                            )
                            ambiguous_ties = unique_sources['AmbiguousTies'][k]
                            # Handle formatting if the first source in
                            # AmbiguousTies was removed.
                            if ambiguous_ties.startswith('_'):
                                ambiguous_ties = ambiguous_ties[1:]
                            # Handle formatting if the last source in
                            # AmbiguousTies was removed.
                            if ambiguous_ties.endswith('_'):
                                ambiguous_ties = ambiguous_ties[:-1]
                            unique_sources['AmbiguousTies'][k] = ambiguous_ties

                        # Update low_df.
                        indices = low_df.index[
                            low_df['SourceID'] == unique_sources['SourceID'][j]
                        ]
                        low_df.loc[indices, 'SourceID'] = source_id
                        # Do not repeat this analysis with the jth source since
                        # we have just determined that the ith and jth source
                        # are the same source.
                        to_skip.append(j)
                    # If more than 70% but less than 100% of observations are
                    # represented by the average position, record the source
                    # pair as an ambiguous match.
                    elif proportion > ambiguity_threshold:
                        if (
                            new_sources['AmbiguousTies'][i] == 'Unknown'
                            or new_sources['AmbiguousTies'][i] == 'None found'
                        ):
                            new_sources['AmbiguousTies'][i] = (
                                unique_sources['SourceID'][j]
                            )
                        elif (
                            unique_sources['SourceID'][j]
                            not in new_sources['AmbiguousTies'][i]
                        ):
                            new_sources['AmbiguousTies'][i] += (
                                '_{}'.format(unique_sources['SourceID'][j])
                            )
                        if (
                            new_sources['AmbiguousTies'][j] == 'Unknown'
                            or new_sources['AmbiguousTies'][j] == 'None found'
                        ):
                            new_sources['AmbiguousTies'][j] = source_id
                        elif (
                            source_id not in new_sources['AmbiguousTies'][j]
                        ):
                            new_sources['AmbiguousTies'][j] += (
                                '_{}'.format(source_id)
                            )
                    if new_sources['AmbiguousTies'][i] == 'Unknown':
                        new_sources['AmbiguousTies'][i] = 'None found'
                    if new_sources['AmbiguousTies'][j] == 'Unknown':
                        new_sources['AmbiguousTies'][j] = 'None found'
    to_skip.sort(reverse=True)
    for k in to_skip:
        del new_sources['SourceID'][k]
        del new_sources['RA'][k]
        del new_sources['Dec'][k]
        del new_sources['FWHM_arcsec'][k]
        del new_sources['AmbiguousTies'][k]

    # Get averages for sources only matched with coarse matching
    for i, source_id in enumerate(new_sources['SourceID']):
        if source_id not in refined:
            temp_df = low_df[
                (low_df['SourceID']) == source_id
            ]
            ra_list = [Angle(ra, u.deg) for ra in temp_df['RA']]
            dec_list = [Angle(dec, u.deg) for dec in temp_df['Dec']]
            fwhm_list = [
                Angle(fwhm, u.arcsec) for fwhm in temp_df['BeamMajAxis_arcsec']
            ]
            num_pts = len(ra_list)
            avg_ra, avg_dec = _average_coordinates(ra_list, dec_list)
            geo_avg_fwhm = math.prod(fwhm_list) ** (1/num_pts)
            new_sources['RA'][i], new_sources['Dec'][i] = _format_coordinates(
                avg_ra,
                avg_dec,
            )
            new_sources['FWHM_arcsec'][i] = round(geo_avg_fwhm.value, 3)

    df = pd.DataFrame.from_dict(new_sources)

    # Write into low and high level tables.
    try:
        with sqlite3.connect(db_path) as con:
            df.to_sql(
                "high_level",
                con=con,
                if_exists='replace',
                index=False,
            )
            low_df.to_sql(
                "low_level",
                con=con,
                if_exists='replace',
                index=False,
            )
    except Exception as e:
        raise ValueError(
            f'Error adding to table(s) at {db_path}: {e}'
        ) from e


def light_curve(
    source_id: str,
    db_path: str | Path = '../sources.db',
    plot: bool = True,
    table: bool = True,
    save_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """
    Retrieve and optionally plot light curve data for a source.

    Parameters
    ----------
    source_id : str
        ID of the source as recorded in the SQLite source database.
    db_path : str | Path, optional
        Path to the SQLite source database, by default '../sources.db'.
    plot : bool, optional
        Whether to plot the source light curve, by default True.
    table : bool, optional
        Whether to return the light curve data as a pandas DataFrame,
        by default True.
    save_path : str | Path | None, optional
        Directory in which to save the light curve plot. If None, the plot
        is not saved, by default None.

    Returns
    -------
    pd.DataFrame or None
        Processed light curve data if `table` is True and there are
        measurements for this source; otherwise, None.

    Raises
    ------
    FileNotFoundError
        If `db_path` does not exist.
    ValueError
        If the `low_level` table is empty.
    OSError
        If the plot cannot be saved to `save_path`.

    Warns
    -----
    UserWarning
        If there are no measurements for the specified source.

    Notes
    -----
    For plotting, observations are grouped into approximate wavelength bands
    using frequency ranges that roughly correspond to the wavelength groupings
    commonly used for SMA light curves:

    - 241.77--282.82 GHz: approximately 1.1--1.2 mm.
    - 208.19--237.93 GHz: approximately 1.3--1.4 mm.
    - 333.10--356.90 GHz: approximately 870 µm.

    These ranges are used here to classify observations for plotting and are
    not intended to define official SMA band boundaries. Observations outside
    these ranges, or with unavailable frequency information, are classified as
    `Other/not found`.
    """
    db_path = Path(db_path)

    if db_path.exists():
        with sqlite3.connect(db_path) as con:
            low_df = pd.read_sql_query("SELECT * FROM low_level;", con)
            if low_df.empty:
                raise ValueError('Table "low_level" is empty.')
    else:
        raise FileNotFoundError(f"Path {db_path} not found.")

    source_df = low_df[low_df['SourceID'] == source_id]
    if source_df.empty:
        warnings.warn(f"There are no measurements for source {source_id}.")
        return

    fmt_str = '%m-%d-%y %H:%M:%S'
    mjd_list = [
        float(
            Time(
                datetime.strptime(dt, fmt_str),
                format='datetime',
                scale='utc'
            ).mjd
        )
        for dt in source_df['ObsDateTime']
    ]

    if plot:
        fig, ax = plt.subplots()

        fluxes = source_df['Flux_mJy'].to_list()
        flux_errs = source_df['FluxUncert_mJy'].to_list()
        flux_unit = 'mJy'
        if max(fluxes) > 1000:
            flux_unit = 'Jy'
            fluxes = [flux / 1000 for flux in fluxes]
            flux_errs = [err / 1000 for err in flux_errs]
        freqs = source_df['Freq_GHz'].tolist()

        band_defs = {
            '~1.1-1.2 mm': (241.77, 282.82),
            '~1.3-1.4 mm': (208.19, 237.93),
            '~870 µm': (333.10, 356.90),
        }

        # Frequency ranges roughly follow wavelength groupings commonly used
        # for SMA light curves.
        bands = {band_name: [] for band_name in band_defs}
        other_band = 'Other/not found'
        bands[other_band] = []

        for date_time, flux, flux_err, freq in zip(
            mjd_list,
            fluxes,
            flux_errs,
            freqs,
        ):
            category = other_band

            if freq != 'Not found':
                float_freq = float(freq)

                for band_name, (lower_freq, upper_freq) in band_defs.items():
                    if lower_freq < float_freq < upper_freq:
                        category = band_name
                        break

            bands[category].append((date_time, flux, flux_err))

        band_styles = {
            '~1.1-1.2 mm': {'color': 'g'},
            '~1.3-1.4 mm': {'color': 'r'},
            '~870 µm': {'color': 'b'},
            'Other/not found': {'color': 'k'},
        }

        for band_name, measurements in bands.items():
            if not measurements:
                continue

            band_date_times, band_fluxes, band_flux_errs = zip(*measurements)

            ax.errorbar(
                band_date_times,
                band_fluxes,
                yerr=band_flux_errs,
                fmt='x',
                capsize=3,
                markersize=2,
                capthick=0.5,
                elinewidth=0.5,
                label=band_name,
                **band_styles[band_name],
            )

        ax.set_title(f'Source {source_id[2:]}')
        ax.set_xlabel('Modified Julian Date')
        ax.set_ylabel(f'Flux [{flux_unit}]')
        ax.legend()
        ax.set_ylim(bottom=0)

        if save_path is not None:
            full_path = Path(save_path) / f'{source_id}.jpg'
            fig.savefig(full_path)

        plt.close(fig)

    if table:
        columns = [
            'ObsDateTime',
            'ObsID',
            'Flux_mJy',
            'FluxUncert_mJy',
            'Freq_GHz',
        ]
        cal_df = source_df[columns].copy()
        cal_df['SNR'] = (
            cal_df['Flux_mJy'] / cal_df['FluxUncert_mJy']
        ).round(2)
        cal_df['MJD'] = mjd_list
        return cal_df


def clause_helper(
    column_name: str,
    parameter: object | None,
    other_type: type,
):
    """
    Construct a SQL condition for a column and parameter.

    Parameters
    ----------
    column_name : str
        Name of the database column to which the condition applies.
    parameter : object | None
        Value used to construct the condition. A single value or a list of
        values is accepted. If `None` or an empty/whitespace-only string is
        supplied, no condition is constructed.
    other_type : type
        Expected type of `parameter` and, when `parameter` is a list, of
        each element in the list.

    Returns
    -------
    tuple[str, list]
        SQL condition fragment and the corresponding parameter values.
        An empty condition and parameter list are returned when `parameter`
        is `None` or is an empty string or list.

    Raises
    ------
    TypeError
        If `parameter` is neither `None`, a list, nor an instance of
        `other_type`, or if an element of a list has an incorrect type.
    """
    if parameter is None:
        return '', []

    if isinstance(parameter, list):
        if not parameter:
            return '', []

        for value in parameter:
            if not isinstance(value, other_type):
                raise TypeError(
                    "In order to write a condition for "
                    f"{column_name}, if input is a list, its elements "
                    f"must be of type {other_type}."
                )

        condition = ' OR '.join(
            f'{column_name} = ?' for _ in parameter
        )
        return f'({condition})', parameter

    if not isinstance(parameter, other_type):
        raise TypeError(
            f"In order to write a condition for {column_name}, input must "
            f"be None, of type list, or of type {other_type}."
        )

    if isinstance(parameter, str) and not parameter.strip():
        return '', []

    return f'({column_name} = ?)', [parameter]


def search_low_level(
    db_path: str | Path = '../sources.db',
    field_name: str | list[str] | None = None,
    stationary: bool = True,
    lower_freq: float | None = None,
    upper_freq: float | None = None,
    lower_flux: float | None = None,
    upper_flux: float | None = None,
    coord: tuple[str, str] | None = None,
    sep_lower: str | None = None,
    sep_upper: str | None = None,
    internal: bool | None = None,
    obs_id: str | list[str] | None = None,
    source_id: str | list[str] | None = None,
    obs_dt_lower: str | None = None,
    obs_dt_upper: str | None = None,
    show_thumbnails: bool = True,
) -> pd.DataFrame:
    """
    Search the low-level source database using the supplied criteria.

    Parameters
    ----------
    db_path : str | Path, optional
        Path to the SQLite database containing the `low_level` table.
    field_name : str | list[str] | None, optional
        Field name used to restrict the search. If a list of field names is
        supplied, records matching any of the field names are returned.
    stationary : bool | None, optional
        If `True`, return stationary sources. If `False`, return
        non-stationary sources. If `None`, do not filter on stationarity.
    lower_freq : float | None, optional
        Lower frequency bound in GHz. Sources with unavailable frequencies
        are excluded when a frequency bound is supplied.
    upper_freq : float | None, optional
        Upper frequency bound in GHz. Sources with unavailable frequencies
        are excluded when a frequency bound is supplied.
    lower_flux : float | None, optional
        Lower flux-density bound in mJy.
    upper_flux : float | None, optional
        Upper flux-density bound in mJy.
    coord : tuple[str, str] | None, optional
        Right ascension and declination used to restrict the search, given as
        `(ra, dec)`. Both values must be parseable by
        `astropy.coordinates.Angle`.
    sep_lower : str | None, optional
        Lower angular-separation bound. The value must be parseable by
        `astropy.coordinates.Angle`.
    sep_upper : str | None, optional
        Upper angular-separation bound. The value must be parseable by
        `astropy.coordinates.Angle`.
    internal : bool | None, optional
        If `True`, return internal sources. If `False`, return non-internal
        sources. If `None`, do not filter on this property.
    obs_id : str | list[str] | None, optional
        Observation ID used to restrict the search. A list of observation
        IDs may also be supplied.
    source_id : str | list[str] | None, optional
        Source ID used to restrict the search. A list of source IDs may also
        be supplied.
    obs_dt_lower : str | None, optional
        Lower observation date/time bound in the format
        `%m-%d-%y %H:%M:%S`.
    obs_dt_upper : str | None, optional
        Upper observation date/time bound in the format
        `%m-%d-%y %H:%M:%S`.
    show_thumbnails : bool, optional
        If `True`, display the images stored in the `Image` column of the
        returned table.

    Returns
    -------
    pd.DataFrame
        Table containing the database records that satisfy the supplied
        search criteria.

    Raises
    ------
    OSError
        If `db_path` does not exist.
    TypeError
        If a frequency or flux-density bound has an invalid type, or if a
        value supplied to one of the list-based search parameters has an
        invalid type.
    ValueError
        If a supplied coordinate, separation, or observation date/time
        cannot be interpreted as required, or if a lower bound is greater
        than its corresponding upper bound.

    Notes
    -----
    Frequency, coordinate, and observation date/time filtering is performed
    after querying the database.

    Observation date/time values must use the format
    `%m-%d-%y %H:%M:%S`.
    """
    db_path = Path(db_path)

    # Validate numeric bounds before querying the database.
    for value, name in (
        (lower_freq, 'frequency lower'),
        (upper_freq, 'frequency upper'),
        (lower_flux, 'flux lower'),
        (upper_flux, 'flux upper'),
    ):
        if (
            value is not None
            and not (
                isinstance(value, int)
                or isinstance(value, float)
            )
        ):
            raise TypeError(
                f"Inputted {name} bound must be None, of type int, "
                "or of type float."
            )

    if lower_freq is not None and upper_freq is not None:
        if lower_freq > upper_freq:
            raise ValueError(
                f"Inputted frequency lower bound {lower_freq} is greater "
                f"than inputted frequency upper bound {upper_freq}."
            )

    if lower_flux is not None and upper_flux is not None:
        if lower_flux > upper_flux:
            raise ValueError(
                f"Inputted flux lower bound {lower_flux} is greater "
                f"than inputted flux upper bound {upper_flux}."
            )

    conditions = []
    parameters = []

    condition, values = clause_helper(
        column_name='FieldName',
        parameter=field_name,
        other_type=str,
    )
    if condition:
        conditions.append(condition)
        parameters.extend(values)

    if stationary is not None:
        conditions.append('(Stationary = ?)')
        parameters.append(stationary)

    if lower_flux is not None and upper_flux is not None:
        conditions.append('(Flux_mJy BETWEEN ? AND ?)')
        parameters.extend([lower_flux, upper_flux])
    elif lower_flux is not None:
        conditions.append('(Flux_mJy >= ?)')
        parameters.append(lower_flux)
    elif upper_flux is not None:
        conditions.append('(Flux_mJy <= ?)')
        parameters.append(upper_flux)

    if internal is not None:
        conditions.append('(Internal = ?)')
        parameters.append(internal)

    condition, values = clause_helper(
        column_name='ObsID',
        parameter=obs_id,
        other_type=str,
    )
    if condition:
        conditions.append(condition)
        parameters.extend(values)

    condition, values = clause_helper(
        column_name='SourceID',
        parameter=source_id,
        other_type=str,
    )
    if condition:
        conditions.append(condition)
        parameters.extend(values)

    where_clause = ''
    if conditions:
        where_clause = ' WHERE ' + ' AND '.join(conditions)

    query = f'SELECT * FROM low_level{where_clause}'

    if db_path.exists():
        with sqlite3.connect(db_path) as con:
            result_df = pd.read_sql_query(
                query,
                con,
                params=parameters,
            )
    else:
        raise OSError(f'Path {db_path} not found')

    freq = pd.to_numeric(result_df['Freq_GHz'], errors='coerce')

    if lower_freq is not None and upper_freq is not None:
        result_df = result_df[
            freq.between(lower_freq, upper_freq)
        ]
    elif lower_freq is not None:
        result_df = result_df[freq >= lower_freq]
    elif upper_freq is not None:
        result_df = result_df[freq <= upper_freq]

    lower_ang = None
    upper_ang = None

    if sep_lower is not None:
        lower_ang = Angle(sep_lower)

    if sep_upper is not None:
        upper_ang = Angle(sep_upper)

    if lower_ang is not None and upper_ang is not None:
        if lower_ang > upper_ang:
            raise ValueError(
                f"Inputted separation lower bound {sep_lower} is greater "
                f"than inputted separation upper bound {sep_upper}."
            )

    if coord is not None:
        if len(coord) != 2:
            raise ValueError(
                f"`coord` must contain an RA and a Dec. Got {coord}."
            )

        if coord[0] is None or coord[1] is None:
            raise ValueError(
                "If `coord` is provided as a tuple, neither entries may be "
                "`None`."
            )

        search_coord = SkyCoord(
            ra=Angle(coord[0]),
            dec=Angle(coord[1]),
        )

        keep = []

        for ra, dec in zip(result_df['RA'], result_df['Dec']):
            result_coord = SkyCoord(ra=ra, dec=dec)
            separation = search_coord.separation(result_coord)

            if lower_ang is not None and separation <= lower_ang:
                keep.append(False)
            elif upper_ang is not None and separation >= upper_ang:
                keep.append(False)
            else:
                keep.append(True)

        result_df = result_df[keep]

    lower_dt = None
    upper_dt = None
    fmt = '%m-%d-%y %H:%M:%S'

    if obs_dt_lower is not None:
        try:
            lower_dt = datetime.strptime(obs_dt_lower, fmt)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Error converting inputted observation date and time lower "
                "bound to datetime object. Please check the input format and "
                f"ensure it matches {fmt}."
            ) from e

    if obs_dt_upper is not None:
        try:
            upper_dt = datetime.strptime(obs_dt_upper, fmt)
        except (TypeError, ValueError) as e:
            raise ValueError(
                "Error converting inputted observation date and time upper "
                "bound to datetime object. Please check the input format and "
                f"ensure it matches {fmt}."
            ) from e

    if lower_dt is not None and upper_dt is not None:
        if lower_dt > upper_dt:
            raise ValueError(
                "Inputted observation date and time lower bound "
                f"{obs_dt_lower} is later than inputted observation date and "
                f"time upper bound {obs_dt_upper}."
            )

    obs_datetime = pd.to_datetime(
        result_df['ObsDateTime'],
        format=fmt,
    )

    if lower_dt is not None and upper_dt is not None:
        result_df = result_df[
            (obs_datetime >= lower_dt) &
            (obs_datetime <= upper_dt)
        ]
    elif lower_dt is not None:
        result_df = result_df[obs_datetime >= lower_dt]
    elif upper_dt is not None:
        result_df = result_df[obs_datetime <= upper_dt]

    if result_df.empty:
        print('Search returned an empty table.')

    result_df.reset_index(drop=True, inplace=True)

    if show_thumbnails:
        image_data = result_df['Image']

        for thumbnail in image_data:
            image_buffer = io.BytesIO(thumbnail)

            with Image.open(image_buffer) as image:
                image.show()

    return result_df


def search_high_level(
    db_path: str | Path = '../sources.db',
    source_id: str | list | None = None,
    coord: tuple | None = None,
    sep_lower: str | None = None,
    sep_upper: str | None = None,
    ambiguous_ties: bool | str | list | None = None,
    ambig_exact: bool = False,
):
    """
    Search the high-level source database using the supplied criteria.

    Parameters
    ----------
    db_path : str | Path, optional
        Path to the SQLite database containing the `high_level` table.
    source_id : str | list | None, optional
        Source ID used to restrict the search. A list of source IDs may also
        be supplied.
    coord : tuple of str | None, optional
        Right Ascension and declination used to restrict the search, given as
        `(ra, dec)`. Both values must be parseable by
        `astropy.coordinates.Angle`, and neither value may be `None`.
    sep_lower : str | None, optional
        Lower angular-separation bound. The value must be parseable by
        `astropy.coordinates.Angle`.
    sep_upper : str | None, optional
        Upper angular-separation bound. The value must be parseable by
        `astropy.coordinates.Angle`.
    ambiguous_ties : bool | str | list | None, optional
        Restrict the search according to ambiguous source ties. If a boolean
        is supplied, `True` selects sources with ambiguous ties and `False`
        selects sources without ambiguous ties. A string may be supplied to
        search for a single specific ambiguous source ID, while a list of
        strings may be supplied to search for multiple specific ambiguous
        source IDs.
    ambig_exact : bool, optional
        If `True`, require the supplied ambiguous source IDs to match exactly.
        If `False`, search for ambiguous source IDs contained within the
        `AmbiguousTies` field.

    Returns
    -------
    pd.DataFrame
        Table containing the database records that satisfy the supplied
        search criteria.

    Raises
    ------
    OSError
        If `db_path` does not exist.
    ValueError
        If `coord` does not contain exactly two values, if either value in
        `coord` is `None`, or if a lower separation bound is greater than the
        corresponding upper separation bound.

    Notes
    -----
    Coordinate filtering is performed using the angular separation between
    the supplied coordinate and the coordinates in the database.

    If no coordinate is supplied, no coordinate filtering is performed.

    Multiple ambiguous source IDs are stored in the `AmbiguousTies` column as
    underscore-separated values. When `ambig_exact` is `False`, matching is
    performed against the individual IDs contained in this field.
    """
    db_path = Path(db_path)

    conditions = []
    parameters = []

    condition, values = clause_helper(
        column_name='SourceID',
        parameter=source_id,
        other_type=str,
    )
    if condition:
        conditions.append(condition)
        parameters.extend(values)

    where_clause = ''
    if conditions:
        where_clause = " WHERE " + " AND ".join(conditions)

    if db_path.exists():
        with sqlite3.connect(db_path) as con:
            result_df = pd.read_sql_query(
                f"SELECT * FROM high_level{where_clause}",
                con,
                params=parameters,
            )
    else:
        raise OSError(f"Path {db_path} not found.")

    lower_ang = None
    upper_ang = None

    if sep_lower is not None:
        lower_ang = Angle(sep_lower)


    if sep_upper is not None:
        upper_ang = Angle(sep_upper)

    if lower_ang is not None and upper_ang is not None:
        if lower_ang > upper_ang:
            raise ValueError(
                f"Inputted separation lower bound {sep_lower} is greater than "
                f"inputted separation upper bound {sep_upper}."
            )

    if coord is not None:
        if len(coord) != 2:
            raise ValueError(
                f"`coord` must contain an RA and a Dec. Got {coord}."
            )

        if coord[0] is None or coord[1] is None:
            raise ValueError(
                "If `coord` is provided as a tuple, neither entries may be "
                "`None`."
            )

        search_coord = SkyCoord(
            ra=Angle(coord[0]),
            dec=Angle(coord[1]),
        )

        keep = []

        for ra, dec in zip(result_df['RA'], result_df['Dec']):
            result_coord = SkyCoord(ra=ra, dec=dec)
            separation = search_coord.separation(result_coord)

            if lower_ang is not None and separation <= lower_ang:
                keep.append(False)
            elif upper_ang is not None and separation >= upper_ang:
                keep.append(False)
            else:
                keep.append(True)

        result_df = result_df[keep]

    if ambiguous_ties is not None:
        if isinstance(ambiguous_ties, bool):
            if ambiguous_ties:
                result_df = result_df[
                    result_df['AmbiguousTies'] != 'None found'
                ]

            elif not ambiguous_ties:
                result_df = result_df[
                    result_df['AmbiguousTies'] == 'None found'
                ]

        # Require ambiguous ties to exactly match provided source IDs.
        elif ambig_exact:
            if isinstance(ambiguous_ties, list):
                if ambiguous_ties:
                    try:
                        ambiguous_ties = [
                            ele.strip() for ele in ambiguous_ties
                        ]
                    except AttributeError:
                        if isinstance(ambiguous_ties, str):
                            raise AttributeError
                        else:
                            raise TypeError(
                                "In order to search by ambiguous ties, if "
                                "input is a list, its elements must be of "
                                "type str."
                            )

                    keep = []

                    for row in range(len(result_df)):
                        temp = result_df['AmbiguousTies'].iloc[row]
                        for ele in ambiguous_ties:
                            if ele not in temp:
                                keep.append(False)
                                break
                            temp = temp.replace(ele, '')

                        if temp.replace('_', ''):
                            keep.append(False)
                        else:
                            keep.append(True)

                    result_df = result_df[keep]

            elif isinstance(ambiguous_ties, str):
                keep = []

                ambiguous_ties = ambiguous_ties.strip()
                if ambiguous_ties:
                    for row in range(len(result_df)):
                        temp = result_df['AmbiguousTies'].iloc[row]
                        if ambiguous_ties != temp:
                            keep.append(False)
                        else:
                            keep.append(True)

                result_df = result_df[keep]

            else:
                raise TypeError(
                    "In order to search by ambiguous ties, input must be "
                    "None, of type list, or of type str."
                )

        # Only require provided source IDs to be present in the ambiguous ties.
        elif not ambig_exact:
            if isinstance(ambiguous_ties, list):
                if ambiguous_ties:
                    try:
                        ambiguous_ties = [
                            ele.strip() for ele in ambiguous_ties
                        ]
                    except AttributeError:
                        if isinstance(ambiguous_ties, str):
                            raise AttributeError
                        else:
                            raise TypeError(
                                "In order to search by ambiguous ties, if "
                                "input is a list, its elements must be of "
                                "type str."
                            )

                    for row in range(len(result_df)):
                        mismatch_found = False
                        temp = result_df['AmbiguousTies'].iloc[row]
                        for ele in ambiguous_ties:
                            if ele not in temp:
                                keep.append(False)
                                mismatch_found = True
                                break
                        if not mismatch_found:
                            keep.append(True)

                    result_df = result_df[keep]

            elif isinstance(ambiguous_ties, str):
                ambiguous_ties = ambiguous_ties.strip()
                if ambiguous_ties:
                    for row in range(len(result_df)):
                        temp = result_df['AmbiguousTies'].iloc[row]
                        if ambiguous_ties not in temp:
                            keep.append(False)
                        else:
                            keep.append(True)

                    result_df = result_df[keep]

    if result_df.empty:
        print("Search returned an empty table.")

    result_df.reset_index(drop=True, inplace=True)

    return result_df
