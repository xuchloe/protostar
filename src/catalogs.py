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
    field_info = {'Field Name': name, 'Obs Date Time': obs_date_time, 'File Name': fits_file[fits_file.rindex('/')+1:],\
                   'Stationary': stationary,\
                   'Beam Maj Axis (arcsec)': round(float(beam_maj_axis.to(u.arcsec)/u.arcsec), 3),\
                   'Beam Min Axis (arcsec)': round(float(beam_min_axis.to(u.arcsec)/u.arcsec), 3),\
                   'Beam Pos Angle (deg)': round(bpa, 3),\
                   'Freq (GHz)': freq}

    field_info['Flux Uncert (mJy)'] = round(summ['conservative_rms'] * 1e3, 3)

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
            info['Flux (mJy)'] = round(summ['int_peak_val'][i] * 1000, 3)

            snr = summ['int_peak_val'][i] / summ['conservative_rms']
            b_min_uncert = float(bmaj / snr)
            b_maj_uncert = float(bmin / snr)
            info['RA Uncert (arcsec)'] = round(b_min_uncert*abs(math.sin(bpa)) + b_maj_uncert*abs(math.cos(bpa)), 3)
            info['Dec Uncert (arcsec)'] = round(b_maj_uncert*abs(math.sin(bpa)) + b_min_uncert*abs(math.cos(bpa)), 3)

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

            key = f'Source {pt_source_count}'
            interesting_sources[key] = info
            pt_source_count +=1

    for i in range(n_ext_sources):
        info = field_info.copy()
        info['Flux (mJy)'] = round(summ[f'ext_peak_val'][i] * 1000, 3)

        snr = summ['ext_peak_val'][i] / summ['conservative_rms']
        b_min_uncert = float(bmaj / snr)
        b_maj_uncert = float(bmin / snr)
        info['RA Uncert (arcsec)'] = round(b_min_uncert*abs(math.sin(bpa)) + b_maj_uncert*abs(math.cos(bpa)), 3)
        info['Dec Uncert (arcsec)'] = round(b_maj_uncert*abs(math.sin(bpa)) + b_min_uncert*abs(math.cos(bpa)), 3)

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

        key = f'Source {pt_source_count}'
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
        new_number = int(key.replace('Source ', ''))
        new_key = f'Source {new_number + shift}'
        catalog_1[new_key] = value
    return catalog_1


def low_level_csv(folder, csv_path = './low_level.csv'):

    master_catalog = None
    old_df = None
    str_obs_id = 'Unknown'

    try:
        old_df = pd.read_csv(csv_path)
    except:
        pass

    try:
        str_obs_id = folder.replace('/mnt/COMPASS9/sma/quality/', '')
        obs_id = str_obs_id.replace('/', '')
        obs_id = int(obs_id) #will throw Exception if obs_id isn't just numbers
        if old_df is not None:
            old_df = old_df[(old_df['Obs ID']) != obs_id] #removing old or outdated entries
    except Exception as e:
        obs_id = 'Unknown'
        print(f'Error with obsID: {e}. WARNING: Old/outdated data may not be deleted.')

    if old_df is not None:
        master_catalog = (old_df.T).to_dict()

    for file in glob.glob(os.path.join(folder, '*.fits')):
        try:
            catalog = make_catalog(file)
            if catalog is not None:
                for value in catalog.values():
                    value['Obs ID'] = obs_id
                    value['Source ID'] = 'Unknown'
                if master_catalog is None:
                    master_catalog = catalog
                elif catalog is not None:
                    master_catalog = combine_catalogs(master_catalog, catalog)
        except Exception as e:
            print(f'Error for {file}: {e}')

    df = pd.DataFrame.from_dict(master_catalog)
    df = df.T

    if master_catalog is not None:
        # fixing rounding error where 60 appears in the seconds
        date_times = df['Obs Date Time'].tolist()
        df.drop(columns='Obs Date Time', inplace=True)
        for i in range(len(date_times)):
            dt = date_times[i]
            m_end = dt.rindex(':')
            s_start = m_end + 1
            if dt[s_start:] == '60':
                dt = dt[:s_start] + '0'
                fmt = '%m-%d-%y %H:%M'
                date_times[i] = (datetime.strptime(dt[:m_end], fmt) + timedelta(minutes=1)).strftime('%m-%d-%y %H:%M:%S')
        df['Obs Date Time'] = date_times

    df.to_csv(csv_path, mode='w', header=True, index=False)


def high_level_csv(low_level_path = './low_level.csv', high_level_path = './high_level.csv'):

    low_df = pd.read_csv(low_level_path)
    unique_sources = None

    try:
        unique_sources = pd.read_csv(high_level_path).to_dict(orient='list')
    except:
        pass

    #coarse matching
    for row in range(len(low_df)):
        if low_df['Source ID'].iloc[row] == 'Unknown': #check to make sure we didn't already do coarse matching
            if low_df['Stationary'].iloc[row]:
                if unique_sources is not None:
                    ra = low_df['RA'].iloc[row]
                    dec = low_df['Dec'].iloc[row]
                    coord1 = SkyCoord(ra, dec)
                    fwhm = low_df['Beam Maj Axis (arcsec)'].iloc[row]
                    source_ids = unique_sources['Source ID']
                    matched  = False
                    while not matched:
                        for i in range(len(source_ids)): #compare with each unique source
                            coord2 = SkyCoord(unique_sources['RA'][i], unique_sources['Dec'][i])
                            sep = coord1.separation(coord2)
                            fwhm2_val = float(unique_sources['FWHM (arcsec)'][i])
                            max_sep = (fwhm * fwhm2_val)**(1/2) * u.arcsec
                            matched = (sep <= max_sep)
                            if matched:
                                low_df.loc[row, 'Source ID'] = source_ids[i]
                                break
                        break
                    if not matched:
                        num = 1
                        id_nums = [int(source_id.replace('id', '')) for source_id in unique_sources['Source ID']]
                        while num in id_nums:
                            num += 1
                        next_number = '0' * (4 - len(str(num))) + str(num)
                        next_id = f'id{next_number}'
                        source_ids.append(next_id)
                        unique_sources['RA'].append(ra)
                        unique_sources['Dec'].append(dec)
                        unique_sources['FWHM (arcsec)'].append(fwhm)
                        low_df.loc[row, 'Source ID'] = next_id
                        unique_sources['Ambiguous Ties'].append('Unknown')
                else:
                    ra = low_df['RA'].iloc[row]
                    dec = low_df['Dec'].iloc[row]
                    fwhm = low_df['Beam Maj Axis (arcsec)'].iloc[row]
                    unique_sources = {'Source ID': ['id0001'], 'RA': [ra], 'Dec': [dec], 'FWHM (arcsec)': [fwhm], 'Ambiguous Ties': ['Unknown']}
                    low_df.loc[row, 'Source ID'] = 'id0001'
            else:
                low_df.loc[row, 'Source ID'] = 'Not Stationary'

    #further refining matches
    new_sources = unique_sources.copy()
    refined = []
    to_skip = []
    for i in range(len(unique_sources['Source ID'])):
        temp_df = low_df[(low_df['Source ID']) == unique_sources['Source ID'][i]]
        ra_list = [Angle(ra, u.deg) for ra in temp_df['RA']]
        dec_list = [Angle(dec, u.deg) for dec in temp_df['Dec']]
        fwhm_list = [Angle(fwhm, u.arcsec) for fwhm in temp_df['Beam Maj Axis (arcsec)']]
        if len(unique_sources['Source ID']) > 1 and i not in to_skip:
            for j in range(i + 1, len(unique_sources['Source ID'])):
                if j not in to_skip:
                    temp_df2 = low_df[(low_df['Source ID']) == unique_sources['Source ID'][j]]
                    ra_list2 = [Angle(ra, u.deg) for ra in temp_df2['RA']]
                    dec_list2 = [Angle(dec, u.deg) for dec in temp_df2['Dec']]
                    fwhm_list2 = [Angle(fwhm, u.arcsec) for fwhm in temp_df2['Beam Maj Axis (arcsec)']]
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
                        refined.append(new_sources['Source ID'][i])
                        #match found, update averages
                        hms_ra = avg_ra.hms
                        str_ra = f'{hms_ra.h}h{hms_ra.m}m{round(hms_ra.s, 2)}s'
                        new_sources['RA'][i] = str_ra
                        new_sources['Dec'][i] = avg_dec
                        new_sources['FWHM (arcsec)'][i] = round(geo_avg_fwhm.value, 3)
                        #get rid of "replaced" source in Ambiguous Ties
                        for k in range(len(unique_sources['Source ID'])):
                            unique_sources['Ambiguous Ties'][k] = unique_sources['Ambiguous Ties'][k].replace(unique_sources['Source ID'][j], '')
                            unique_sources['Ambiguous Ties'][k] = unique_sources['Ambiguous Ties'][k].replace('__', '_')
                            if unique_sources['Ambiguous Ties'][k][0] == '_':
                                unique_sources['Ambiguous Ties'][k] = unique_sources['Ambiguous Ties'][k][1:]
                            if unique_sources['Ambiguous Ties'][k][-1] == '_':
                                unique_sources['Ambiguous Ties'][k] = unique_sources['Ambiguous Ties'][k][:-1]
                        #update low_df
                        indices = low_df.index[low_df['Source ID'] == unique_sources['Source ID'][j]]
                        low_df.loc[indices, 'Source ID'] = unique_sources['Source ID'][i]
                        to_skip.append(j)
                    elif proportion > 0.7: #average point is a good representative for over 70% but less than 100% of points, ambiguous
                        if new_sources['Ambiguous Ties'][i] == 'Unknown' or new_sources['Ambiguous Ties'][i] == 'None found':
                            new_sources['Ambiguous Ties'][i] = unique_sources['Source ID'][j]
                        elif unique_sources['Source ID'][j] not in new_sources['Ambiguous Ties'][i]:
                            new_sources['Ambiguous Ties'][i] += '_{}'.format(unique_sources['Source ID'][j])
                        if new_sources['Ambiguous Ties'][j] == 'Unknown' or new_sources['Ambiguous Ties'][j] == 'None found':
                            new_sources['Ambiguous Ties'][j] = unique_sources['Source ID'][i]
                        elif unique_sources['Source ID'][i] not in new_sources['Ambiguous Ties'][j]:
                            new_sources['Ambiguous Ties'][j] += '_{}'.format(unique_sources['Source ID'][i])
                    if new_sources['Ambiguous Ties'][i] == 'Unknown':
                        new_sources['Ambiguous Ties'][i] = 'None found'
                    if new_sources['Ambiguous Ties'][j] == 'Unknown':
                        new_sources['Ambiguous Ties'][j] = 'None found'
    to_skip.sort(reverse=True)
    for k in to_skip:
        del new_sources['Source ID'][k]
        del new_sources['RA'][k]
        del new_sources['Dec'][k]
        del new_sources['FWHM (arcsec)'][k]
        del new_sources['Ambiguous Ties'][k]

    #get averages for sources only matched with coarse matching
    for i in range(len(new_sources['Source ID'])):
        if new_sources['Source ID'][i] not in refined:
            temp_df = low_df[(low_df['Source ID']) == new_sources['Source ID'][i]]
            ra_list = [Angle(ra, u.deg) for ra in temp_df['RA']]
            dec_list = [Angle(dec, u.deg) for dec in temp_df['Dec']]
            fwhm_list = [Angle(fwhm, u.arcsec) for fwhm in temp_df['Beam Maj Axis (arcsec)']]
            num_pts = len(ra_list)
            avg_ra = sum(ra_list) / num_pts
            hms_ra = avg_ra.hms
            str_ra = f'{hms_ra.h}h{hms_ra.m}m{round(hms_ra.s, 2)}s'
            avg_dec = sum(dec_list) / num_pts
            geo_avg_fwhm = math.prod(fwhm_list) ** (1/num_pts)
            new_sources['RA'][i] = str_ra
            new_sources['Dec'][i] = avg_dec
            new_sources['FWHM (arcsec)'][i] = round(geo_avg_fwhm.value, 3)

    df = pd.DataFrame.from_dict(new_sources)
    df.to_csv(high_level_path, mode='w', header=True, index=False)
    low_df.to_csv(low_level_path, mode='w', header=True, index=False)


def light_curve(low_path: str = './low_level.csv', high_path: str = './high_level.csv', unique_ids: list = None,\
                plot: bool = True, table: bool = True, save_path: str = ''):

    low_df = pd.read_csv(low_path)
    high_df = pd.read_csv(high_path)
    if unique_ids == None:
        unique_ids = high_df['Source ID'].tolist()

    if plot:
        for source in unique_ids:
            plt.subplots()
            source_df = low_df[low_df['Source ID'] == source]
            fluxes = source_df['Flux (mJy)'].to_list()
            flux_errs = source_df['Flux Uncert (mJy)'].to_list()
            flux_unit = 'mJy'
            if max(fluxes) > 1000:
                flux_unit = 'Jy'
                for i in range(len(fluxes)):
                    fluxes[i] /= 1000
                    flux_errs[i] /= 1000
            date_times = source_df['Obs Date Time'].tolist()
            fmt_str = '%m-%d-%y %H:%M:%S'
            date_times = [Time(datetime.strptime(dt, fmt_str), format='datetime', scale='utc').mjd for dt in date_times]

            freqs = source_df['Freq (GHz)'].tolist()
            other = []
            small_milli = [] # 1.1mm
            large_milli = [] # 1.3mm
            micro = [] # 870µm
            for i in range(len(freqs)):
                if freqs[i] == 'Not found':
                    other.append(i)
                    pass
                else:
                    try:
                        float_freq = float(freqs[i])
                        if float_freq > 260.69 and float_freq < 285.52: # 1.15-1.05mm
                            small_milli.append(i)
                        elif float_freq > 222.07 and float_freq < 239.83: # 1.35-1.25mm
                            large_milli.append(i)
                        elif float_freq > 340.67 and float_freq < 348.60: # 880-860µm
                            micro.append(i)
                        else:
                            other.append(i)
                    except Exception as e:
                        print(f'Error while getting the frequencies for source {source}: {e}')
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
                        capthick=0.5, elinewidth=0.5, label='1.1mm')
            plt.errorbar(lg_milli_dt, lg_milli_flx, yerr=lg_milli_flx_err, color='r', fmt='x', capsize=3, markersize=2,\
                        capthick=0.5, elinewidth=0.5, label='1.3mm')
            plt.errorbar(micro_dt, micro_flx, yerr=micro_flx_err, color='b', fmt='x', capsize=3, markersize=2,\
                        capthick=0.5, elinewidth=0.5, label='870µm')
            plt.errorbar(other_dt, other_flx, yerr=other_flx_err, color='k', fmt='x', capsize=3, markersize=2,\
                        capthick=0.5, elinewidth=0.5, label='Other/not found')

            plt.title(f'Source {source[2:]}')
            plt.xlabel('Modified Julian Date')
            plt.ylabel(f'Flux [{flux_unit}]')
            plt.legend()
            plt.ylim(bottom=0)

            if save_path != '':
                try:
                    if save_path[-1] != '/':
                        save_path = save_path + '/'
                    plt.savefig(f'{save_path}{source}.jpg')
                except:
                    print('Error saving figure. Double check path entered.')

    if table:
        for j in range(len(unique_ids)):
            source = unique_ids[j]
            source_df = low_df[low_df['Source ID'] == source]
            dat_name = f'./{source}_flux_history.dat'
            with open(dat_name, 'w') as new_file:
                new_file.write('#{}, RA: {}, Dec:{}\n'.format(source, high_df.loc[j, 'RA'], low_df.loc[j, 'Dec']))
            cal_df = source_df.copy()
            for col in cal_df.columns:
                if col not in ['Obs Date Time', 'Obs ID', 'Flux (mJy)', 'Flux Uncert (mJy)', 'Freq (GHz)']:
                    cal_df.drop(columns=col, inplace=True)
            snr_list = [round(float(cal_df['Flux (mJy)'].to_list()[i] / cal_df['Flux Uncert (mJy)'].to_list()[i]), 2) for i in range(len(cal_df))]
            cal_df['SNR'] = snr_list
            fmt_str = '%m-%d-%y %H:%M:%S'
            mjd_list = [float(Time(datetime.strptime(dt, fmt_str), format='datetime', scale='utc').mjd) for dt in cal_df['Obs Date Time']]
            cal_df['MJD'] = mjd_list
            cal_df.to_csv(dat_name, sep='\t', index=False, mode='a')
