from astropy.io import fits
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import glob
import pandas as pd
import json
import os
import math
from find_source import summary
from catalogs import make_catalog, combine_catalogs


def start_html(html_path):
    '''
    Starts source_info.html, in which source information can be stored.
    '''

    with open(html_path, 'w') as html_file:
        start = '''
        <!DOCTYPE html>
        <html>
        <style>
        img.field {
        width: 40%;
        height: 40%
        }
        img.bp {
        width: 20%;
        height: 20%
        }
        img.gain {
        width: 45%;
        height: 45%
        }
        .centered-large-text {
        text-align: center;
        font-size: 36px;
        }
        </style>
        <body>
        '''
        html_file.write(start)
        html_file.close()


def obs_info_to_html(json_file: str, html_path: str):
    '''
    Appends observation information table to source_info.html using information from a .json file.

    Parameters
    ----------
    json_file : str
        The path of the .json file that contains the observation information.
    '''

    with open(html_path, 'a') as html_file:
        try:
            with open(json_file, 'r') as file:
                obs_dict = json.load(file)

            #cleaning up obs_dict
            for key, value in obs_dict.items():
                if type(value) == list:
                    string = ', '.join(value)
                    obs_dict[key] = [string]
            obs_id = obs_dict.pop('obsID')
            base_name = obs_dict.pop('basename')

            df = pd.DataFrame(obs_dict)
            df_transposed = df.T

            html_table = df_transposed.to_html()

            html_file.write(f'<p class=\'centered-large-text\'>Source Information for {base_name} (ObsID {obs_id}) </p>')
            html_file.write(html_table)
        except:
            html_file.write('<p> Error generating observation information table. </p>')


def ap_eff_to_html(html_path, matlab: str):

    try:
        data = loadmat(matlab)
        ap_eff_array = data['apEffCorr']

        n_ants = len(ap_eff_array)
        panda_dict = {}

        for ant in range(n_ants):
            ant_eff = {}
            ant_eff['RxA LSB'] = float(ap_eff_array[ant][0])
            ant_eff['RxA USB'] = float(ap_eff_array[ant][1])
            ant_eff['RxB LSB'] = float(ap_eff_array[ant][2])
            ant_eff['RxB USB'] = float(ap_eff_array[ant][3])
            panda_dict[f'Ant {ant+1}'] = ant_eff

        df = pd.DataFrame.from_dict(panda_dict)
        df_transposed = df.T
        html_table = df_transposed.to_html()

        with open(html_path, 'a') as html_file:
            html_file.write(html_table)
    except:
        print('Error with aperture efficiency data.')


def calibration_plots(html_path, matlab: str):

    plt.rcdefaults()
    plt.rcParams['figure.dpi'] = 60
    plt.rcParams['font.size'] = 8


    data = loadmat(matlab)
    gt = data['gainTime']
    gws = data['gainWinSoln']
    gcs = data['gainChanSoln']
    gain_type = data['gainType']

    n_times = len(gt)
    n_ants = len(gws[0])
    n_spws = len(gws[0][0])
    n_chans = len(gcs[0][0][0])

    utc_midpts = []
    for t in range(len(gt)):
        midpt = 0.5 * (gt[t][0].real + gt[t][0].imag)
        utc_midpts.append((midpt%1)*24)

    colors = ['blue','r','y','purple','orange','g','m','c']

    chan_bit = 7
    if all(bit == 0 for bit in (gain_type & (2**chan_bit))):
        chan_bit = 0
    spw_bit = 6
    if all(bit == 0 for bit in (gain_type & (2**spw_bit))):
        spw_bit = 1

    #plotting bandpass gain solutions for amplitude and phase
    fig, ax = plt.subplots(nrows=n_ants, ncols=1, sharex=True, figsize=(3,8))
    fig2, ax2 = plt.subplots(nrows=n_ants, ncols=1, sharex=True, figsize=(3,8))

    max_amp = 0

    for time in range(n_times):
        if (gain_type & (2**chan_bit))[time] != 0:
            for ant in range(n_ants):

                #shifting for cosmetics
                pos = ax[ant].get_position()
                pos.x0 += 0.05
                pos.x1 += 0.05
                ax[ant].set_position(pos)
                pos2 = ax2[ant].get_position()
                pos2.x0 += 0.06
                pos2.x1 += 0.06
                ax2[ant].set_position(pos2)

                #no x axis ticks
                ax[ant].xaxis.set_tick_params(labelbottom=False)
                ax2[ant].xaxis.set_tick_params(labelbottom=False)


                for spw in range(n_spws):
                    amp_to_plot = [abs(a) for a in gcs.copy()[time][ant][spw]]
                    pha_to_plot = [np.angle(p, deg=True) for p in gcs.copy()[time][ant][spw]]
                    if max(amp_to_plot) > max_amp:
                        max_amp = max(amp_to_plot)

                    x_axis = np.arange(spw * n_chans + 1, (1 + spw) * n_chans + 1)

                    ax[ant].scatter(x_axis, amp_to_plot, c=colors[spw], s=20, marker='x', linewidths=1.5)
                    ax2[ant].scatter(x_axis, pha_to_plot, c=colors[spw], s=20, marker='x', linewidths=1.5)

                    ax[ant].yaxis.set_label_position('right')
                    ax2[ant].yaxis.set_label_position('right')
                    ax[ant].set_ylabel(f'Ant{ant+1}')
                    ax2[ant].set_ylabel(f'Ant{ant+1}')

    plt.setp(ax, yticks=np.arange(0, max_amp+1, 0.5))
    plt.setp(ax2, yticks=[-180,-120,-60,0,60,120,180])
    fig.suptitle('Bandpass gain solutions for amplitude', y=0.92)
    fig2.suptitle('Bandpass gain solutions for phase', y=0.92)
    fig.supxlabel('Full antenna bandwidth', y=0.07)
    fig2.supxlabel('Full antenna bandwidth', y=0.07)
    fig.supylabel('Gain amplitude')
    fig2.supylabel('Gain phase')

    html_folder = os.path.dirname(html_path)

    fig.savefig(os.path.join(html_folder, 'bp_amp.jpg'))
    fig2.savefig(os.path.join(html_folder, 'bp_pha.jpg'))

    plt.close()

    #plotting gain solutions for amplitude and phase
    n_rows = math.ceil(n_ants / 2)
    n_cols = 2

    fig, ax = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, figsize=(5.7,4))
    fig2, ax2 = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, figsize=(5.7,4))

    max_amp, min_time, max_time = 0, float('inf'), 0

    for spw in range(n_spws):
        for ant in range(n_ants):
            amp_to_plot, pha_to_plot = [], []
            times = []

            if ant < n_rows:
                row, col = ant, 0

                #shifting for cosmetics
                pos = ax[row, col].get_position()
                pos.x0 -= 0.005
                pos.x1 -= 0.005
                ax[row, col].set_position(pos)
                pos2 = ax2[row, col].get_position()
                pos2.x0 -= 0.005
                pos2.x1 -= 0.005
                ax2[row, col].set_position(pos2)
            else:
                row, col = ant % n_rows, 1

            for time in range(n_times):
                if gain_type[time] & (2**6) != 0:
                    amp_val = abs((gws.copy())[time][ant][spw])
                    pha_val = np.angle((gws.copy())[time][ant][spw], deg=True)
                    amp_to_plot.append(amp_val)
                    pha_to_plot.append(pha_val)

                    if amp_val > max_amp:
                        max_amp = amp_val

                    t = utc_midpts[time]
                    if t < min_time:
                        min_time = t
                    if t > max_time:
                        max_time = t

                    times.append(t)

            ax[row, col].scatter(times, amp_to_plot, c=colors[spw], s=4, marker='D')
            ax2[row, col].scatter(times, pha_to_plot, c=colors[spw], s=4, marker='D')

            ax[row, col].yaxis.set_label_position('right')
            ax2[row, col].yaxis.set_label_position('right')
            ax[row, col].set_ylabel(f'Ant{ant+1}')
            ax2[row, col].set_ylabel(f'Ant{ant+1}')
            amp_to_plot, pha_to_plot = [], []

    plt.setp(ax, xticks=np.arange(min_time//1, math.ceil(max_time), 1), yticks=np.arange(0, max_amp+1, 0.5))
    plt.setp(ax2, xticks=np.arange(min_time//1, math.ceil(max_time), 1), yticks=[-180,-120,-60,0,60,120,180])
    fig.suptitle('Gain solutions for amplitude')
    fig2.suptitle('Gain solutions for phase')
    fig.supxlabel('UT hours')
    fig2.supxlabel('UT hours')
    fig.supylabel('Gain amplitude')
    fig2.supylabel('Gain phase')

    fig.savefig(os.path.join(html_folder, 'g_amp.jpg'))
    fig2.savefig(os.path.join(html_folder, 'g_pha.jpg'))

    plt.close()


def fig_to_html(html_path: str, fits_file: str, radius_buffer: float = 5.0, ext_threshold: float = None):
    '''
    Appends source figures to source_info.html.

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
    '''

    with open(html_path, 'a') as html_file:
        try:
            summary(fits_file=fits_file, radius_buffer=radius_buffer, ext_threshold=ext_threshold,\
                    short_dict=False, plot=True, save_path=os.path.dirname(html_path))

            #getting full path
            file = fits_file
            while '/' in file:
                file = file[file.index('/')+1:]
            file = file.replace('.fits', '')
            if ext_threshold == None:
                ext_threshold = 'default'
            file += f'_rb{radius_buffer}_et{ext_threshold}'
            full_path = f'./{file}.jpg'

            html_figure = f'''
            <img class=\'field\' src=\'{full_path}\'>
            <br>
            '''

            html_file.write(html_figure)
        except:
            html_file.write(f'<p> Error generating figure for {fits_file}. </p>')


def catalog_to_html(catalog: dict, html_path):
    '''
    Appends source information table to source_info.html.

    Parameters
    ----------
    catalog : dict
        A catalog in the format returned by make_catalog().
    '''

    df = pd.DataFrame.from_dict(catalog)
    df_transposed = df.T
    html_table = df_transposed.to_html()

    with open(html_path, 'a') as html_file:
        html_file.write(html_table)


def end_html(html_path: str):
    '''
    Ends source_info.html, in which source information can be stored.
    '''

    with open(html_path, 'a') as html_file:

        end = '''
        </body>
        </html>
        '''

        html_file.write(end)


def full_html_and_txt(folder: str, threshold: float = 0.01, radius_buffer: float = 5.0, ext_threshold: float = None):
    '''
    From a folder of FITS files, creates source_info.html with observation information table, source figures, and source information table
    and creates interesting_field.txt with names of objects with any (possibly) interesting detections.

    Parameters
    ----------
    folder : str
        The path of the folder containing the FITS files to be analyzed.
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
    '''

    html_path = os.path.join(folder, 'index.html')
    matlab_file = os.path.join(folder, 'gains.mat')

    start_html(html_path)

    json_file = os.path.join(folder, 'polaris.json')

    obs_info_to_html(json_file, html_path)

    ap_eff_to_html(html_path, matlab_file)

    try:
        calibration_plots(html_path, matlab_file)

        with open(html_path, 'a') as html_file:
            html_gain_info = f'''
            <img class=\'bp\' src=\'./bp_amp.jpg'\'>
            <img class=\'bp\' src=\'./bp_pha.jpg'\'>
            <br>
            <img class=\'gain\' src=\'./g_amp.jpg'\'>
            <img class=\'gain\' src=\'./g_pha.jpg'\'>
            <br>
            '''
            html_file.write(html_gain_info)
    except:
        print('Error with gain calibration information.')

    final_catalog = {}
    with open(json_file, 'r') as file:
        obs_dict = json.load(file)

    sci_targs = [targ.lower() for targ in obs_dict['sciTargs']]
    pol_cals = [cal.lower() for cal in obs_dict['polCals']]
    with open(os.path.join(folder, 'interesting_fields.txt'), 'w') as txt:
        for file in glob.glob(os.path.join(folder, '*.fits')):
            obj = fits.getheader(file)['OBJECT']
            if obj.lower() not in pol_cals:
                fig_to_html(html_path, file, radius_buffer=radius_buffer, ext_threshold=ext_threshold)
            if obj.lower() in sci_targs:
                catalog = make_catalog(file, threshold=threshold, radius_buffer=radius_buffer, ext_threshold=ext_threshold)

                #add field name to .txt file if it is a science target with a significant detection in the initial inclusion region
                if catalog != None:
                    for key, value in catalog.items():
                        if value['Internal'] == True:
                            txt.write(f'{obj}\n')
                    final_catalog = combine_catalogs(final_catalog, catalog)

    catalog_to_html(final_catalog, html_path)
    end_html(html_path)

    plt.close('all')
