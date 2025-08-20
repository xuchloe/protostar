from astropy.io import fits
from astropy.coordinates import Angle, SkyCoord
from astropy.time import Time
import astropy.units as u
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import glob
import pandas as pd
import os
import math
import sqlite3
from find_source import summary


def make_catalog(fits_file: str, threshold: float = 0.01, radius_buffer: float = 5.0, ext_threshold: float = None):
    '''
    Summarizes information on any significant point sources detected in an image.

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

    summ = summary(fits_file=fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold, short_dict=True, plot=False)

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
    ctype3 = header_data['CTYPE3']
    crval3 = header_data['CRVAL3']
    cunit3 = header_data['CUNIT3']

    freq = 'Not found'
    if ctype3 == 'FREQ':
        if cunit3 == 'GHz':
            freq = crval3
        elif cunit2 == 'Hz':
            freq = crval3 / 1e9 # into GHz
        freq = round(freq, 3)
    elif ctype3 == 'CHANNUM':
        hdul = fits.open(fits_file)
        try:
            freq_col = hdul[1].columns[1]
            if freq_col.name == 'Freq':
                if freq_col.unit == 'Hz':
                    freq = hdul[1].data[0][1] / 1e9 # into GHz
                elif freq_col.unit == 'GHz':
                    freq = hdul[1].data[0][1]
            freq = round(freq, 3)
        except:
            pass

    #assume beam axes in same units as CUNIT1 and CUNIT2 and BPA in degrees
    beam_maj_axis = Angle(bmaj, cunit1)
    beam_min_axis = Angle(bmin, cunit1)
    bpa_rad = math.radians(bpa)

    moving_objects = ['venus', 'mars', 'jupiter', 'uranus', 'neptune', 'io', 'europa', 'ganymede', 'callisto', 'titan',\
               'ceres', 'vesta', 'pallas', 'juno']

    stationary = True
    if name.lower() in moving_objects:
        stationary = False
    else:
        for obj in moving_objects:
            if obj in name.lower():
                stationary = False
                break

    interesting_sources = {}
    field_info = {'FieldName': name, 'ObsDateTime': obs_date_time, 'FileName': fits_file[fits_file.rindex('/')+1:],\
                   'Stationary': stationary,\
                   'BeamMajAxis_arcsec': round(float(beam_maj_axis.to(u.arcsec)/u.arcsec), 3),\
                   'BeamMinAxis_arcsec': round(float(beam_min_axis.to(u.arcsec)/u.arcsec), 3),\
                   'BeamPosAngle_deg': round(bpa, 3),\
                   'Freq_GHz': freq}

    field_info['FluxUncert_mJy'] = round(summ['conservative_rms'] * 1e3, 3)

    n_int_sources = len(summ['int_peak_val'])
    if type(summ['ext_peak_val']) == str:
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

    for i in range(n_int_sources):
        if (summ['int_prob'][i] < threshold and summ['calc_int_prob'][i] < threshold):
            info = field_info.copy()
            info['Flux_mJy'] = round(summ['int_peak_val'][i] * 1000, 3)

            snr = summ['int_peak_val'][i] / summ['conservative_rms']
            b_min_uncert = float((beam_maj_axis.to(u.arcsec) / u.arcsec) / snr)
            b_maj_uncert = float((beam_min_axis.to(u.arcsec) / u.arcsec) / snr)
            info['RAUncert_arcsec'] = round(b_min_uncert*abs(math.sin(bpa)) + b_maj_uncert*abs(math.cos(bpa)), 3)
            info['DecUncert_arcsec'] = round(b_maj_uncert*abs(math.sin(bpa)) + b_min_uncert*abs(math.cos(bpa)), 3)

            ra_offset = summ['int_peak_coord'][i][ra_index] * u.arcsec
            dec_offset = summ['int_peak_coord'][i][dec_index] * u.arcsec
            coord = center.spherical_offsets_by(ra_offset, dec_offset)

            ra_tuple = coord.ra.hms
            dec_tuple = coord.dec.dms

            # rounding the arcseconds to 2 past the decimal
            ra_str = f'{int(ra_tuple.h)}h{abs(int(ra_tuple.m))}m{abs(round(float(ra_tuple.s), 2))}s'
            dec_str = f'{int(dec_tuple.d)}d{abs(int(dec_tuple.m))}m{abs(round(float(dec_tuple.s), 2))}s'

            info['RA'] = ra_str
            info['Dec'] = dec_str
            info['Internal'] = True

            key = f'Source{pt_source_count}'
            interesting_sources[key] = info
            pt_source_count +=1

    for i in range(n_ext_sources):
        info = field_info.copy()
        info['Flux_mJy'] = round(summ[f'ext_peak_val'][i] * 1000, 3)

        snr = summ['ext_peak_val'][i] / summ['conservative_rms']
        b_min_uncert = float(bmaj / snr)
        b_maj_uncert = float(bmin / snr)
        info['RAUncert_arcsec'] = round(b_min_uncert*abs(math.sin(bpa)) + b_maj_uncert*abs(math.cos(bpa)), 3)
        info['DecUncert_arcsec'] = round(b_maj_uncert*abs(math.sin(bpa)) + b_min_uncert*abs(math.cos(bpa)), 3)

        ra_offset = summ['ext_peak_coord'][i][ra_index] * u.arcsec
        dec_offset = summ['ext_peak_coord'][i][dec_index] * u.arcsec
        coord = center.spherical_offsets_by(ra_offset, dec_offset)

        ra_tuple = coord.ra.hms
        dec_tuple = coord.dec.dms

        # rounding the arcseconds to 2 past the decimal
        ra_str = f'{int(ra_tuple.h)}h{abs(int(ra_tuple.m))}m{abs(round(float(ra_tuple.s), 2))}s'
        dec_str = f'{int(dec_tuple.d)}d{abs(int(dec_tuple.m))}m{abs(round(float(dec_tuple.s), 2))}s'

        info['RA'] = ra_str
        info['Dec'] = dec_str
        info['Internal'] = False

        key = f'Source{pt_source_count}'
        interesting_sources[key] = info
        pt_source_count +=1

    if interesting_sources == {}:
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
        new_number = int(key.replace('Source', ''))
        new_key = f'Source{new_number + shift}'
        catalog_1[new_key] = value
    return catalog_1


def low_level_table(folder: str, db_path: str = '../sources.db'):

    str_obs_id = 'Unknown'
    big_catalog = None

    try:
        str_obs_id = folder.replace('/mnt/COMPASS9/sma/quality/', '')
        obs_id = str_obs_id.replace('/', '')
        obs_id = int(obs_id) #will throw Exception if obs_id isn't just numbers
    except Exception as e:
        obs_id = 'Unknown'
        print(f'Error with obsID: {e}. WARNING: Old/outdated data may not be deleted.')

    if os.path.exists(db_path):
        # get all rows from existing low level table, if it exists
        con1_established = False
        con1_closed = False
        old_data_cleared = False
        try:
            con1 = sqlite3.connect(db_path)
            con1_established = True
            cur1 = con1.cursor()
            cur1.execute("DELETE FROM low_level WHERE ObsID='{}'".format(obs_id))
            con1.commit()
            old_data_cleared = True
            con1.close()
            con1_closed = True
        except Exception as e:
            if con1_established and not con1_closed:
                con1.close()
                if not old_data_cleared and not isinstance(e, sqlite3.OperationalError):
                    print(f'Error removing old/outdated data from table "low_level" at {db_path}: {e}')

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
            print(f'Error for {file}: {e}')

    if big_catalog is not None:
        df = pd.DataFrame.from_dict(big_catalog)
        df = df.T

        # fixing rounding error where 60 appears in the seconds
        date_times = df['ObsDateTime'].tolist()
        df.drop(columns='ObsDateTime', inplace=True)
        for i in range(len(date_times)):
            dt = date_times[i]
            m_end = dt.rindex(':')
            s_start = m_end + 1
            if dt[s_start:] == '60':
                dt = dt[:s_start] + '0'
                fmt = '%m-%d-%y %H:%M'
                date_times[i] = (datetime.strptime(dt[:m_end], fmt) + timedelta(minutes=1)).strftime('%m-%d-%y %H:%M:%S')
        df['ObsDateTime'] = date_times

        # write into low level table
        con2_established = False
        con2_closed = False
        try:
            con2 = sqlite3.connect(db_path)
            con2_established = True
            df.to_sql("low_level", con=con2, if_exists='append', index=False)
            con2.close()
            con2_closed = True
        except Exception as e:
            if con2_established and not con2_closed:
                con2.close()
            print(f'Error adding to table "low_level" at {db_path}: {e}')


def high_level_table(db_path: str = '../sources.db'):

    unique_sources = None

    if os.path.exists(db_path):
        # get all rows from low level and high level tables, if they exist
        con1_established = False
        con1_closed = False
        try:
            con1 = sqlite3.connect(db_path)
            con1_established = True
            low_df = pd.read_sql_query("SELECT * FROM low_level;", con1)
            if low_df.empty:
                raise ValueError('Table "low_level" is empty')
            unique_sources = pd.read_sql_query("SELECT * FROM high_level;", con1).to_dict(orient='list')
            con1.close()
            con1_closed = True
        except Exception as e:
            if con1_established and not con1_closed:
                con1.close()
            if not isinstance(e, pd.errors.DatabaseError):
                print(f'Error reading from database at {db_path}: {e}')
    else:
        raise OSError(f'Path {db_path} not found')

    #coarse matching
    for row in range(len(low_df)):
        if low_df['SourceID'].iloc[row] == 'Unknown': #check to make sure we didn't already do coarse matching
            if low_df['Stationary'].iloc[row]:
                if unique_sources is not None:
                    ra = low_df['RA'].iloc[row]
                    dec = low_df['Dec'].iloc[row]
                    coord1 = SkyCoord(ra, dec)
                    fwhm = low_df['BeamMajAxis_arcsec'].iloc[row]
                    source_ids = unique_sources['SourceID']
                    matched  = False
                    while not matched:
                        for i in range(len(source_ids)): #compare with each unique source
                            coord2 = SkyCoord(unique_sources['RA'][i], unique_sources['Dec'][i])
                            sep = coord1.separation(coord2)
                            fwhm2_val = float(unique_sources['FWHM_arcsec'][i])
                            max_sep = (fwhm * fwhm2_val)**(1/2) * u.arcsec
                            matched = (sep <= max_sep)
                            if matched:
                                low_df.loc[row, 'SourceID'] = source_ids[i]
                                break
                        break
                    if not matched:
                        num = 1
                        id_nums = [int(source_id.replace('id', '')) for source_id in unique_sources['SourceID']]
                        while num in id_nums:
                            num += 1
                        next_number = '0' * (4 - len(str(num))) + str(num)
                        next_id = f'id{next_number}'
                        source_ids.append(next_id)
                        unique_sources['RA'].append(ra)
                        unique_sources['Dec'].append(dec)
                        unique_sources['FWHM_arcsec'].append(fwhm)
                        low_df.loc[row, 'SourceID'] = next_id
                        unique_sources['AmbiguousTies'].append('Unknown')
                else:
                    ra = low_df['RA'].iloc[row]
                    dec = low_df['Dec'].iloc[row]
                    fwhm = low_df['BeamMajAxis_arcsec'].iloc[row]
                    unique_sources = {'SourceID': ['id0001'], 'RA': [ra], 'Dec': [dec], 'FWHM_arcsec': [fwhm], 'AmbiguousTies': ['Unknown']}
                    low_df.loc[row, 'SourceID'] = 'id0001'
            else:
                low_df.loc[row, 'SourceID'] = 'Not Stationary'

    #further refining matches
    new_sources = unique_sources.copy()
    refined = []
    to_skip = []
    for i in range(len(unique_sources['SourceID'])):
        temp_df = low_df[(low_df['SourceID']) == unique_sources['SourceID'][i]]
        ra_list = [Angle(ra, u.deg) for ra in temp_df['RA']]
        dec_list = [Angle(dec, u.deg) for dec in temp_df['Dec']]
        fwhm_list = [Angle(fwhm, u.arcsec) for fwhm in temp_df['BeamMajAxis_arcsec']]
        if len(unique_sources['SourceID']) > 1 and i not in to_skip:
            for j in range(i + 1, len(unique_sources['SourceID'])):
                if j not in to_skip:
                    temp_df2 = low_df[(low_df['SourceID']) == unique_sources['SourceID'][j]]
                    ra_list2 = [Angle(ra, u.deg) for ra in temp_df2['RA']]
                    dec_list2 = [Angle(dec, u.deg) for dec in temp_df2['Dec']]
                    fwhm_list2 = [Angle(fwhm, u.arcsec) for fwhm in temp_df2['BeamMajAxis_arcsec']]
                    new_ra_list = ra_list + ra_list2
                    new_dec_list = dec_list + dec_list2
                    new_fwhm_list = fwhm_list + fwhm_list2
                    num_pts = len(new_ra_list)
                    avg_ra = sum(new_ra_list) / num_pts
                    avg_dec = sum(new_dec_list) / num_pts
                    geo_avg_fwhm = math.prod(new_fwhm_list) ** (1/num_pts)
                    avg_pt = SkyCoord(avg_ra, avg_dec)
                    temp = 0
                    for pt in range(num_pts):
                        sep = avg_pt.separation(SkyCoord(new_ra_list[pt], new_dec_list[pt]))
                        if sep > geo_avg_fwhm / 2:
                            temp += 1
                    proportion = (num_pts - temp) / (num_pts)
                    if proportion == 1: #average point is a good representative for all points, same source
                        refined.append(new_sources['SourceID'][i])
                        #match found, update averages
                        hms_ra = avg_ra.hms
                        dms_dec = avg_dec.dms
                        str_ra = f'{int(hms_ra.h)}h{abs(int(hms_ra.m))}m{round(abs(hms_ra.s), 2)}s'
                        str_dec = f'{int(dms_dec.d)}d{abs(int(dms_dec.m))}m{round(abs(dms_dec.s), 2)}s'
                        new_sources['RA'][i] = str_ra
                        new_sources['Dec'][i] = str_dec
                        new_sources['FWHM_arcsec'][i] = round(geo_avg_fwhm.value, 3)
                        #get rid of "replaced" source in AmbiguousTies
                        for k in range(len(unique_sources['SourceID'])):
                            unique_sources['AmbiguousTies'][k] = unique_sources['AmbiguousTies'][k].replace(unique_sources['SourceID'][j], '')
                            unique_sources['AmbiguousTies'][k] = unique_sources['AmbiguousTies'][k].replace('__', '_')
                            if unique_sources['AmbiguousTies'][k][0] == '_':
                                unique_sources['AmbiguousTies'][k] = unique_sources['AmbiguousTies'][k][1:]
                            if unique_sources['AmbiguousTies'][k][-1] == '_':
                                unique_sources['AmbiguousTies'][k] = unique_sources['AmbiguousTies'][k][:-1]
                        #update low_df
                        indices = low_df.index[low_df['SourceID'] == unique_sources['SourceID'][j]]
                        low_df.loc[indices, 'SourceID'] = unique_sources['SourceID'][i]
                        to_skip.append(j)
                    elif proportion > 0.7: #average point is a good representative for over 70% but less than 100% of points, ambiguous
                        if new_sources['AmbiguousTies'][i] == 'Unknown' or new_sources['AmbiguousTies'][i] == 'None found':
                            new_sources['AmbiguousTies'][i] = unique_sources['SourceID'][j]
                        elif unique_sources['SourceID'][j] not in new_sources['AmbiguousTies'][i]:
                            new_sources['AmbiguousTies'][i] += '_{}'.format(unique_sources['SourceID'][j])
                        if new_sources['AmbiguousTies'][j] == 'Unknown' or new_sources['AmbiguousTies'][j] == 'None found':
                            new_sources['AmbiguousTies'][j] = unique_sources['SourceID'][i]
                        elif unique_sources['SourceID'][i] not in new_sources['AmbiguousTies'][j]:
                            new_sources['AmbiguousTies'][j] += '_{}'.format(unique_sources['SourceID'][i])
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

    #get averages for sources only matched with coarse matching
    for i in range(len(new_sources['SourceID'])):
        if new_sources['SourceID'][i] not in refined:
            temp_df = low_df[(low_df['SourceID']) == new_sources['SourceID'][i]]
            ra_list = [Angle(ra, u.deg) for ra in temp_df['RA']]
            dec_list = [Angle(dec, u.deg) for dec in temp_df['Dec']]
            fwhm_list = [Angle(fwhm, u.arcsec) for fwhm in temp_df['BeamMajAxis_arcsec']]
            num_pts = len(ra_list)
            avg_ra = sum(ra_list) / num_pts
            hms_ra = avg_ra.hms
            str_ra = f'{int(hms_ra.h)}h{abs(int(hms_ra.m))}m{round(abs(hms_ra.s), 2)}s'
            avg_dec = sum(dec_list) / num_pts
            dms_dec = avg_dec.dms
            str_dec = f'{int(dms_dec.d)}d{abs(int(dms_dec.m))}m{round(abs(dms_dec.s), 2)}s'
            geo_avg_fwhm = math.prod(fwhm_list) ** (1/num_pts)
            new_sources['RA'][i] = str_ra
            new_sources['Dec'][i] = str_dec
            new_sources['FWHM_arcsec'][i] = round(geo_avg_fwhm.value, 3)

    df = pd.DataFrame.from_dict(new_sources)

    # write into low and high level tables
    con2_established = False
    con2_closed = False
    try:
        con2 = sqlite3.connect(db_path)
        con2_established = True
        df.to_sql("high_level", con=con2, if_exists='replace', index=False)
        low_df.to_sql("low_level", con=con2, if_exists='replace', index=False)
        con2.close()
        con2_closed = True
    except Exception as e:
        if con2_established and not con2_closed:
            con2.close()
        print(f'Error adding to table(s) at {db_path}: {e}')


def light_curve(source_id: str, db_path: str = '../sources.db',\
                plot: bool = True, table: bool = True, save_path: str = ''):

    if os.path.exists(db_path):
        # get all rows from low level and high level tables, if they exist
        con1_established = False
        con1_closed = False
        try:
            con1 = sqlite3.connect(db_path)
            con1_established = True
            low_df = pd.read_sql_query("SELECT * FROM low_level;", con1)
            if low_df.empty:
                raise ValueError('Table "low_level" is empty')
            high_df = pd.read_sql_query("SELECT * FROM high_level;", con1)
            if high_df.empty:
                raise ValueError('Table "high_level" is empty')
            con1.close()
            con1_closed = True
        except Exception as e:
            if con1_established and not con1_closed:
                con1.close()
            print(f'Error reading from database at {db_path}: {e}')
    else:
        raise OSError(f'Path {db_path} not found')

    source_df = low_df[low_df['SourceID'] == source_id]

    if plot:
        fluxes = source_df['Flux_mJy'].to_list()
        flux_errs = source_df['FluxUncert_mJy'].to_list()
        flux_unit = 'mJy'
        if max(fluxes) > 1000:
            flux_unit = 'Jy'
            for i in range(len(fluxes)):
                fluxes[i] /= 1000
                flux_errs[i] /= 1000
        date_times = source_df['ObsDateTime'].tolist()
        fmt_str = '%m-%d-%y %H:%M:%S'
        date_times = [Time(datetime.strptime(dt, fmt_str), format='datetime', scale='utc').mjd for dt in date_times]

        freqs = source_df['Freq_GHz'].tolist()
        other = []
        small_milli = [] # 1.1-1.2mm
        large_milli = [] # 1.3-1.4mm
        micro = [] # 870µm
        for i in range(len(freqs)):
            if freqs[i] == 'Not found':
                other.append(i)
                pass
            else:
                try:
                    float_freq = float(freqs[i])
                    if float_freq > 241.77 and float_freq < 282.82: # 1.24-1.06mm
                        small_milli.append(i)
                    elif float_freq > 208.19 and float_freq < 237.93: # 1.44-1.26mm
                        large_milli.append(i)
                    elif float_freq > 333.10 and float_freq < 356.90: # 900-840µm
                        micro.append(i)
                    else:
                        other.append(i)
                except Exception as e:
                    print(f'Error while getting the frequencies for source {source_id}: {e}')
        other_dt = [date_times[a] for a in other]
        other_flx = [fluxes[a] for a in other]
        other_flx_err = [flux_errs[a] for a in other]
        sm_milli_dt = [date_times[b] for b in small_milli]
        sm_milli_flx = [fluxes[b] for b in small_milli]
        sm_milli_flx_err = [flux_errs[b] for b in small_milli]
        lg_milli_dt = [date_times[c] for c in large_milli]
        lg_milli_flx = [fluxes[c] for c in large_milli]
        lg_milli_flx_err = [flux_errs[c] for c in large_milli]
        micro_dt = [date_times[d] for d in micro]
        micro_flx = [fluxes[d] for d in micro]
        micro_flx_err = [flux_errs[d] for d in micro]

        plt.errorbar(sm_milli_dt, sm_milli_flx, yerr=sm_milli_flx_err, color='g', fmt='x', capsize=3, markersize=2,\
                    capthick=0.5, elinewidth=0.5, label='~1.1-1.2mm')
        plt.errorbar(lg_milli_dt, lg_milli_flx, yerr=lg_milli_flx_err, color='r', fmt='x', capsize=3, markersize=2,\
                    capthick=0.5, elinewidth=0.5, label='~1.3-1.4mm')
        plt.errorbar(micro_dt, micro_flx, yerr=micro_flx_err, color='b', fmt='x', capsize=3, markersize=2,\
                    capthick=0.5, elinewidth=0.5, label='~870µm')
        plt.errorbar(other_dt, other_flx, yerr=other_flx_err, color='k', fmt='x', capsize=3, markersize=2,\
                    capthick=0.5, elinewidth=0.5, label='Other/not found')

        plt.title(f'Source {source_id[2:]}')
        plt.xlabel('Modified Julian Date')
        plt.ylabel(f'Flux [{flux_unit}]')
        plt.legend()
        plt.ylim(bottom=0)

        if save_path != '':
            try:
                if save_path[-1] != '/':
                    save_path = save_path + '/'
                plt.savefig(f'{save_path}{source_id}.jpg')
            except:
                print('Error saving figure. Double check path entered.')

    if table:
        cal_df = source_df.copy()
        for col in cal_df.columns:
            if col not in ['ObsDateTime', 'ObsID', 'Flux_mJy', 'FluxUncert_mJy', 'Freq_GHz']:
                cal_df.drop(columns=col, inplace=True)
        snr_list = [round(float(cal_df['Flux_mJy'].to_list()[i] / cal_df['FluxUncert_mJy'].to_list()[i]), 2) for i in range(len(cal_df))]
        cal_df['SNR'] = snr_list
        fmt_str = '%m-%d-%y %H:%M:%S'
        mjd_list = [float(Time(datetime.strptime(dt, fmt_str), format='datetime', scale='utc').mjd) for dt in cal_df['ObsDateTime']]
        cal_df['MJD'] = mjd_list
        return cal_df


def clause_helper(column_name: str, parameter, other_type):
    phrase = ''
    if parameter is not None:
        if type(parameter) == list:
            if parameter:
                for i in range(len(parameter)):
                    if type(parameter[i]) != other_type:
                        raise TypeError(f'In order to write a condition for {column_name}, if input is a list, its elements must be of {other_type}.')
                    if i == 0:
                        phrase += ' ({} = "{}"'.format(column_name, parameter[i])
                    else:
                        phrase += ' OR {} = "{}"'.format(column_name, parameter[i])
            phrase += ')'
        elif type(parameter) == other_type:
            if type(parameter) == str:
                if not parameter.strip():
                    return ''
            phrase += f' ({column_name} = "{parameter}")'
        else:
            raise TypeError(f'In order to write a condition for {column_name}, input must be None, of type list, or of {other_type}.')
        if phrase:
            phrase += ' AND'
    return phrase


def search_low_level(db_path: str = '../sources.db', field_name = None, stationary = True, lower_freq = None, upper_freq = None,\
                    lower_flux = None, upper_flux = None, ra = None, dec = None, sep_lower = None, sep_upper = None,\
                    internal = None, obs_id = None, source_id = None, obs_dt_lower = None, obs_dt_upper = None):

    where_clause = 'WHERE'
    where_clause += clause_helper(column_name='FieldName', parameter=field_name, other_type=str)
    if stationary is not None:
        where_clause += f' (Stationary = {stationary}) AND'
    if not (lower_flux is None or type(lower_flux) == float or type(lower_flux) == int):
        raise TypeError('Inputted lower bound for flux must be None, of type float, or of type int.')
    if not (upper_flux is None or type(upper_flux) == float or type(upper_flux) == int):
        raise TypeError('Inputted upper bound for flux must be None, of type float, or of type int.')
    if lower_flux is not None and upper_flux is not None:
        where_clause += f' (Flux_mJy BETWEEN {lower_flux} AND {upper_flux}) AND'
    elif lower_flux is not None:
        where_clause += f' (Flux_mJy >= {lower_flux})'
    elif upper_flux is not None:
        where_clause += f' (Flux_mJy <= {upper_flux})'
    if internal is not None:
        where_clause += f' (Internal = {internal}) AND'
    where_clause += clause_helper(column_name='ObsID', parameter=obs_id, other_type=str)
    where_clause += clause_helper(column_name='SourceID', parameter=source_id, other_type=str)

    if where_clause[-5:] == 'WHERE':
        where_clause = ''
    if where_clause[-4:] == ' AND':
        where_clause = where_clause[:-4]
    where_clause += ';'

    if os.path.exists(db_path):
        # connect and query
        con_established = False
        con_closed = False
        try:
            con = sqlite3.connect(db_path)
            con_established = True
            result_df = pd.read_sql_query(f'SELECT * FROM low_level {where_clause}', con)
            con.close()
            con_closed = True
        except Exception as e:
            if con_established and not con_closed:
                con.close()
            print(f'Error querying database at {db_path} : {e}')
    else:
        raise OSError(f'Path {db_path} not found')

    to_drop = []
    if not (lower_freq is None or type(lower_freq) == int or type(lower_freq) == float):
        raise TypeError('Inputted frequency lower bound must be None, of type int, or of type float.')
    if not (upper_freq is None or type(upper_freq) == int or type(upper_freq) == float):
        raise TypeError('Inputted frequency upper bound must be None, of type int, or of type float.')
    if lower_freq is not None and upper_freq is not None:
        for row in range(len(result_df)):
            if result_df['Freq_GHz'].iloc[row] == 'Not found':
                to_drop.append(row)
            elif not (float(result_df['Freq_GHz'].iloc[row]) <= upper_freq and float(result_df['Freq_GHz'].iloc[row]) >= lower_freq):
                to_drop.append(row)
        result_df.drop(to_drop, inplace=True)
    elif lower_freq is not None:
        for row in range(len(result_df)):
            if result_df['Freq_GHz'].iloc[row] == 'Not found':
                to_drop.append(row)
            elif not (float(result_df['Freq_GHz'].iloc[row]) >= lower_freq):
                to_drop.append(row)
        result_df.drop(to_drop, inplace=True)
    elif upper_freq is not None:
        for row in range(len(result_df)):
            if result_df['Freq_GHz'].iloc[row] == 'Not found':
                to_drop.append(row)
            elif not (float(result_df['Freq_GHz'].iloc[row]) <= upper_freq):
                to_drop.append(row)
        result_df.drop(to_drop, inplace=True)
    if to_drop and not result_df.empty:
        result_df.reset_index(inplace=True)

    # handling ra, dec stuff
    coord = None
    ra_ang = None
    dec_ang = None
    lower_ang = None
    upper_ang = None
    if sep_lower is not None:
        try:
            lower_ang = Angle(sep_lower)
            if lower_ang == 0:
                lower_ang = None
        except Exception as e:
            print(f'Error converting separation lower bound input to Angle: {e}')
    if sep_upper is not None:
        try:
            upper_ang = Angle(sep_upper)
            if upper_ang == 0:
                upper_ang = None
        except Exception as e:
            print(f'Error converting separation upper bound input to Angle: {e}')
    if lower_ang is not None and upper_ang is not None:
        if lower_ang > upper_ang:
            raise ValueError(f'Inputted separation lower bound {sep_lower} is greater than inputted separation upper bound {sep_upper}.')
    if ra is not None and dec is not None:
        try:
            coord = SkyCoord(ra, dec)
        except Exception as e:
            print(f'Error converting Right Ascension and Declination inputs to SkyCoord object: {e}')
    elif ra is not None:
        try:
            ra_ang = Angle(ra)
        except Exception as e:
            print(f'Error converting Right Ascension input to Angle: {e}')
    elif dec is not None:
        try:
            dec_ang = Angle(dec)
        except Exception as e:
            print(f'Error converting Declination input to Angle: {e}')

    to_drop = []
    if lower_ang is not None and upper_ang is not None:
        if coord is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_dec = result_df['Dec'].iloc[row]
                temp_coord = SkyCoord(temp_ra, temp_dec)
                sep = coord.separation(temp_coord)
                if not (sep <= upper_ang and sep >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif ra_ang is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_ang = Angle(temp_ra)
                if not (abs(ra_ang - temp_ang) <= upper_ang and abs(ra_ang - temp_ang) >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif dec_ang is not None:
            for row in range(len(result_df)):
                temp_dec = result_df['Dec'].iloc[row]
                temp_ang = Angle(temp_dec)
                if not (abs(dec_ang - temp_ang) <= upper_ang and abs(dec_ang - temp_ang) >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
    elif lower_ang is not None:
        if coord is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_dec = result_df['Dec'].iloc[row]
                temp_coord = SkyCoord(temp_ra, temp_dec)
                sep = coord.separation(temp_coord)
                if not (sep >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif ra_ang is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_ang = Angle(temp_ra)
                if not (abs(ra_ang - temp_ang) >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif dec_ang is not None:
            for row in range(len(result_df)):
                temp_dec = result_df['Dec'].iloc[row]
                temp_ang = Angle(temp_dec)
                if not (abs(dec_ang - temp_ang) >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
    elif upper_ang is not None:
        if coord is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_dec = result_df['Dec'].iloc[row]
                temp_coord = SkyCoord(temp_ra, temp_dec)
                sep = coord.separation(temp_coord)
                if not (sep <= upper_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif ra_ang is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_ang = Angle(temp_ra)
                if not (abs(ra_ang - temp_ang) <= upper_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif dec_ang is not None:
            for row in range(len(result_df)):
                temp_dec = result_df['Dec'].iloc[row]
                temp_ang = Angle(temp_dec)
                if not (abs(dec_ang - temp_ang) <= upper_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
    else:
        if coord is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_dec = result_df['Dec'].iloc[row]
                temp_coord = SkyCoord(temp_ra, temp_dec)
                if not (temp_coord == coord):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif ra_ang is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_ang = Angle(temp_ra)
                if not (temp_ang == ra_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif dec_ang is not None:
            for row in range(len(result_df)):
                temp_dec = result_df['Dec'].iloc[row]
                temp_ang = Angle(temp_dec)
                if not (temp_ang == dec_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
    if to_drop and not result_df.empty:
        result_df.reset_index(inplace=True)

    to_drop = []
    # handling observation date time stuff
    lower_dt = None
    upper_dt = None
    fmt = '%m-%d-%y %H:%M:%S'
    if obs_dt_lower is not None:
        try:
            lower_dt = datetime.strptime(obs_dt_lower, fmt)
        except Exception as e:
            print(f'Error converting inputted observation date and time lower bound to datetime object: {e}. Please check the input format and ensure it matches {fmt}.')
    if obs_dt_upper is not None:
        try:
            upper_dt = datetime.strptime(obs_dt_upper, fmt)
        except Exception as e:
            print(f'Error converting inputted observation date and time upper bound input to datetime object: {e}. Please check the input format and ensure it matcheds {fmt}.')

    if lower_dt is not None and upper_dt is not None:
        if lower_dt > upper_dt:
            raise ValueError(f'Inputted observation date and time lower bound {obs_dt_lower} is later than inputted observation date and time upper bound {obs_dt_upper}.')
        for row in range(len(result_df)):
            temp_dt = datetime.strptime(result_df['ObsDateTime'].iloc[row], fmt)
            if not (temp_dt <= upper_dt and temp_dt >= lower_dt):
                to_drop.append(row)
        result_df.drop(to_drop, inplace=True)
    elif lower_dt is not None:
        for row in range(len(result_df)):
            temp_dt = datetime.strptime(result_df['ObsDateTime'].iloc[row], fmt)
            if not (temp_dt >= lower_dt):
                to_drop.append(row)
        result_df.drop(to_drop, inplace=True)
    elif upper_dt is not None:
        for row in range(len(result_df)):
            temp_dt = datetime.strptime(result_df['ObsDateTime'].iloc[row], fmt)
            if not (temp_dt <= upper_dt):
                to_drop.append(row)
        result_df.drop(to_drop, inplace=True)
    if to_drop and not result_df.empty:
        result_df.reset_index(inplace=True)

    if result_df.empty:
        print('Search returned an empty table.')
    else:
        if 'level_0' in result_df:
            result_df.drop(columns='level_0', inplace=True)
        if 'index' in result_df:
            result_df.drop(columns='index', inplace=True)
    return result_df


def search_high_level(db_path: str = '../sources.db', source_id = None, ra = None, dec = None, sep_lower = None, sep_upper = None,\
                      ambiguous_ties = None, ambig_exact: bool = False):
    where_clause = 'WHERE'
    where_clause += clause_helper(column_name='SourceID', parameter=source_id, other_type=str)

    if where_clause[-5:] == 'WHERE':
        where_clause = ''
    if where_clause[-4:] == ' AND':
        where_clause = where_clause[:-4]
    where_clause += ';'

    if os.path.exists(db_path):
        # connect and query
        con_established = False
        con_closed = False
        try:
            con = sqlite3.connect(db_path)
            con_established = True
            result_df = pd.read_sql_query(f'SELECT * FROM high_level {where_clause}', con)
            con.close()
            con_closed = True
        except Exception as e:
            if con_established and not con_closed:
                con.close()
            print(f'Error querying database at {db_path} : {e}')
    else:
        raise OSError(f'Path {db_path} not found')

    # handling ra, dec stuff
    coord = None
    ra_ang = None
    dec_ang = None
    lower_ang = None
    upper_ang = None
    if sep_lower is not None:
        try:
            lower_ang = Angle(sep_lower)
            if lower_ang == 0:
                lower_ang = None
        except Exception as e:
            print(f'Error converting separation lower bound input to Angle: {e}')
    if sep_upper is not None:
        try:
            upper_ang = Angle(sep_upper)
            if upper_ang == 0:
                upper_ang = None
        except Exception as e:
            print(f'Error converting separation upper bound input to Angle: {e}')
    if lower_ang is not None and upper_ang is not None:
        if lower_ang > upper_ang:
            raise ValueError(f'Inputted separation lower bound {sep_lower} is greater than inputted separation upper bound {sep_upper}.')
    if ra is not None and dec is not None:
        try:
            coord = SkyCoord(ra, dec)
        except Exception as e:
            print(f'Error converting Right Ascension and Declination inputs to SkyCoord object: {e}')
    elif ra is not None:
        try:
            ra_ang = Angle(ra)
        except Exception as e:
            print(f'Error converting Right Ascension input to Angle: {e}')
    elif dec is not None:
        try:
            dec_ang = Angle(dec)
        except Exception as e:
            print(f'Error converting Declination input to Angle: {e}')

    to_drop = []
    if lower_ang is not None and upper_ang is not None:
        if coord is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_dec = result_df['Dec'].iloc[row]
                temp_coord = SkyCoord(temp_ra, temp_dec)
                sep = coord.separation(temp_coord)
                if not (sep <= upper_ang and sep >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif ra_ang is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_ang = Angle(temp_ra)
                if not (abs(ra_ang - temp_ang) <= upper_ang and abs(ra_ang - temp_ang) >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif dec_ang is not None:
            for row in range(len(result_df)):
                temp_dec = result_df['Dec'].iloc[row]
                temp_ang = Angle(temp_dec)
                if not (abs(dec_ang - temp_ang) <= upper_ang and abs(dec_ang - temp_ang) >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
    elif lower_ang is not None:
        if coord is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_dec = result_df['Dec'].iloc[row]
                temp_coord = SkyCoord(temp_ra, temp_dec)
                sep = coord.separation(temp_coord)
                if not (sep >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif ra_ang is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_ang = Angle(temp_ra)
                if not (abs(ra_ang - temp_ang) >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif dec_ang is not None:
            for row in range(len(result_df)):
                temp_dec = result_df['Dec'].iloc[row]
                temp_ang = Angle(temp_dec)
                if not (abs(dec_ang - temp_ang) >= lower_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
    elif upper_ang is not None:
        if coord is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_dec = result_df['Dec'].iloc[row]
                temp_coord = SkyCoord(temp_ra, temp_dec)
                sep = coord.separation(temp_coord)
                if not (sep <= upper_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif ra_ang is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_ang = Angle(temp_ra)
                if not (abs(ra_ang - temp_ang) <= upper_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif dec_ang is not None:
            for row in range(len(result_df)):
                temp_dec = result_df['Dec'].iloc[row]
                temp_ang = Angle(temp_dec)
                if not (abs(dec_ang - temp_ang) <= upper_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
    else:
        if coord is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_dec = result_df['Dec'].iloc[row]
                temp_coord = SkyCoord(temp_ra, temp_dec)
                if not (temp_coord == coord):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif ra_ang is not None:
            for row in range(len(result_df)):
                temp_ra = result_df['RA'].iloc[row]
                temp_ang = Angle(temp_ra)
                if not (temp_ang == ra_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
        elif dec_ang is not None:
            for row in range(len(result_df)):
                temp_dec = result_df['Dec'].iloc[row]
                temp_ang = Angle(temp_dec)
                if not (temp_ang == dec_ang):
                    to_drop.append(row)
            result_df.drop(to_drop, inplace=True)
    if to_drop and not result_df.empty:
        result_df.reset_index(inplace=True)

    to_drop = []
    if ambiguous_ties is not None:
        if type(ambiguous_ties) == bool:
            if ambiguous_ties:
                for row in range(len(result_df)):
                    if result_df['AmbiguousTies'].iloc[row] == 'None found':
                        to_drop.append(row)
                result_df.drop(to_drop, inplace=True)
            elif not ambiguous_ties:
                for row in range(len(result_df)):
                    if result_df['AmbiguousTies'].iloc[row] != 'None found':
                        to_drop.append(row)
                result_df.drop(to_drop, inplace=True)
        elif ambig_exact:
            if type(ambiguous_ties) == list:
                if ambiguous_ties:
                    try:
                        ambiguous_ties = [ele.strip() for ele in ambiguous_ties]
                    except AttributeError:
                        if type(ambiguous_ties) == str:
                            raise AttributeError
                        else:
                            raise TypeError('In order to search by ambiguous ties, if input is a list, its elements must be of type str.')
                    for row in range(len(result_df)):
                        temp = result_df['AmbiguousTies'].iloc[row]
                        for ele in ambiguous_ties:
                            if ele not in temp:
                                to_drop.append(row)
                            temp = temp.replace(ele, '')
                        if temp.replace('_', ''): # this means there are source IDs in temp that are not in ambiguous_ties
                            if row not in to_drop:
                                to_drop.append(row)
                    result_df.drop(to_drop, inplace=True)
            elif type(ambiguous_ties) == str:
                ambiguous_ties = ambiguous_ties.strip()
                if ambiguous_ties:
                    for row in range(len(result_df)):
                        temp = result_df['AmbiguousTies'].iloc[row]
                        if ambiguous_ties != temp:
                            to_drop.append(row)
                    result_df.drop(to_drop, inplace=True)
            else:
                raise TypeError('In order to search by ambiguous ties, input must be None, of type list, or of type str.')
        elif not ambig_exact:
            if type(ambiguous_ties) == list:
                if ambiguous_ties:
                    try:
                        ambiguous_ties = [ele.strip() for ele in ambiguous_ties]
                    except AttributeError:
                        if type(ambiguous_ties) == str:
                            raise AttributeError
                        else:
                            raise TypeError('In order to search by ambiguous ties, if input is a list, its elements must be of type str.')
                    for row in range(len(result_df)):
                        temp = result_df['AmbiguousTies'].iloc[row]
                        for ele in ambiguous_ties:
                            if ele in temp:
                                continue
                            to_drop.append(row)
                    result_df.drop(to_drop, inplace=True)
            elif type(ambiguous_ties) == str:
                ambiguous_ties = ambiguous_ties.strip()
                if ambiguous_ties:
                    for row in range(len(result_df)):
                        temp = result_df['AmbiguousTies'].iloc[row]
                        if ambiguous_ties not in temp:
                            to_drop.append(row)
                    result_df.drop(to_drop, inplace=True)
    if to_drop and not result_df.empty:
        result_df.reset_index(inplace=True)

    if result_df.empty:
            print('Search returned an empty table.')
    else:
        if 'level_0' in result_df:
            result_df.drop(columns='level_0', inplace=True)
        if 'index' in result_df:
            result_df.drop(columns='index', inplace=True)
    return result_df
