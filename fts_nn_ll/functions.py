'''
Mathematical functions and data processing utilities 
also full spectrum Voigt profile fitting - fit_spec()
'''

import numpy as np
import scipy.special as ss
import scipy.integrate as si
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import curve_fit

def normalise(a):
    '''
    normalises np array by maximum value
    '''
    return a / a.max()

def voigt(wn, mean, Gw, Lw, snr):
    '''
    Voigt with peak at snr, takes Gw and Lw in mK
    '''
    return snr * normalise(ss.voigt_profile(wn - mean, Gw, Lw))

def asy_voigt(wn, mean, Gw, Lw, snr, a):
    '''
    asym voigt with peak at snr, takes Gw and Lw in mK
    '''
    Gw = 2 * Gw / (1 + np.exp(a * (wn - mean)))
    return snr * normalise(ss.voigt_profile(wn - mean, Gw, Lw))

def add_voigts(wn, *params):
    '''
    Takes params of all lines (1D list) and add them as Voigt profiles onto y at wn
    '''
    width = 5 # /cm
    y = np.zeros_like(wn)
    param_array = np.array(params).reshape(-1, 4)
    for wn_mean, Gw, Lw, snr in param_array:
        mask = (wn > (wn_mean - width)) & (wn < (wn_mean + width))
        wn_calc = wn[mask]
        # Update y only in the masked regions
        y[mask] += voigt(wn_calc, wn_mean, Gw, Lw, snr)
    return y

def add_asy_voigts(wn, *params):
    '''
    Takes params of all lines (1D list) and add them as Voigt profiles onto y at wn
    '''
    y = np.zeros_like(wn)
    param_array = np.array(params).reshape(-1, 5)
    for wn_mean, Gw, Lw, snr, a in param_array:
        y += asy_voigt(wn, wn_mean, Gw, Lw, snr, a)
    return y

def voigt_fwhm(Gw, Lw):
    '''
    Calculate approximate Voigt FWHM given Gw and Lw
    '''
    f_g = Gw * (2 * np.sqrt(2 * np.log(2)))
    f_l = 2 * Lw
    return f_l * 0.5346 + np.sqrt(f_l ** 2 * 0.2166 + f_g ** 2)

def Lw_from_fwhm_Gw(fwhm, Gw):
    f_g = Gw * (2 * np.sqrt(2 * np.log(2)))
    a = 0.5346 ** 2 - 0.2166
    b = -2 * fwhm * 0.5346
    c = fwhm ** 2 - f_g ** 2
    #f_l_1 = (-b + np.sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    f_l_2 = (-b - np.sqrt(b ** 2 - 4 * a * c)) / (2 * a) # Lw is from this root
    Lw_2 = f_l_2 / 2
    return Lw_2

def pad_spec(spec, npo, delw):
    '''
    FT spec of npo that is not a power of 2 (but is even), pad interferogram until length is 2^N (equally on both sides), then transform back
    if npo is not even, padded spec might be a few pnts off a power of 2, in which case add zeros or remove edge pnts in the spectrum...
    '''
    interferogram = np.fft.fftshift(np.fft.fft(spec))
    new_npo = 2 ** np.ceil(np.log2(npo))
    pad_width = int((new_npo - npo) / 2)
    padded_interferogram = np.pad(interferogram, pad_width)
    
    L = 1 / (2 * delw)
    dx = L / npo
    new_L = dx * new_npo
    new_delw = 1 / (2 * new_L)
    return np.real(np.fft.ifft(np.fft.fftshift(padded_interferogram))) * (new_npo / npo), int(new_npo), new_delw

def preprocess_interp(wn, spec, N=1):
    '''
    Fourier interpolation by padding interferogram of spectrum to the next N powers of 2.
    e.g. N=1 means twice more points, N=2 means four times more points
    '''
    if N == 0:
        return [wn, spec]
    else:
        ifg = np.fft.fftshift(np.fft.fft(spec))
        for i in range(N):
            ifg = pad_to_next_power_of_2(ifg)
            spec_interp = np.real(np.fft.ifft(np.fft.fftshift(ifg))) * 2 ** N # SNR changes by 2 ** N
            wn_interp = np.empty(2 * len(wn))
            wn_inbetweens = (wn[:-1] + wn[1:]) / 2
            wn_interp[0::2] = wn
            wn_interp[1::2][:-1] = wn_inbetweens
            wn_interp[-1] = wn_interp[-2] + np.mean(np.diff(wn)) / 2        
            wn = wn_interp
            spec = spec_interp
    return [wn, spec]
        
def preprocess_chunk(arr, chunk_size=1024):
    '''
    2D numpy array [M, 2^N] to be split and shaped into [M, 2^N / chunk_size, chunk_size]
    M is at least 2, e.g., wn array and spec array with length 2^N.
    '''
    return np.moveaxis(np.array(np.array_split(arr, len(arr[0]) / chunk_size, axis=1)), 1, 0)

def pad_to_next_power_of_2(arr):
    # Current length of the array
    current_length = arr.shape[0]    
    # Calculate the next power of 2 greater than or equal to current length
    next_power_of_2 = 2 ** int(np.floor(np.log2(current_length)) + 1)
    # Calculate how much padding is needed
    padding_length = next_power_of_2 - current_length    
    # Pad the array with zeros at the end
    padded_arr = np.pad(arr, int(padding_length/2), mode='constant')

    return padded_arr

def mad(data, axis=None):
    '''
    Median Absolute Deviation
    '''
    median = np.median(data, axis=axis)  # Compute the median
    abs_deviation = np.abs(data - median)  # Compute absolute deviations from the median
    mad_value = np.median(abs_deviation, axis=axis)  # Compute the median of the absolute deviations
    return mad_value

def exp_spec_model_view(wn, spec, npo, line_region, mad_scaling, N_interp, chunk_size, plot=True):
    '''
    Approximately what the model sees, if mad_scaling, expect some discontinuities because different chunks my be scaled differently,
    but this is fine as prediction is done using overlapping windows.   
    Useful for checking interpolation results in interactive plot.
    
    Returns [wn, spec] arrays for scanning for width and snr distributions
    '''
    # Interpolate the same way simulation data was interpolated
    input_wn, input_spec = preprocess_interp(wn, spec, N=N_interp)
     
    # Need to split spectrum into chunks with overlapping windows (50% are overlaps)
    # The 50% of the first and final chunk of the overlapping chunks do not have overlaps, this is fine because we don't usually have lines at ends of spectra
    chunks = []
    chunk_prediction_size = int(chunk_size / 2)
    print('------------------------------------------------')
    print('Slicing up the spectrum and scaling noise for each slice')
    for i in tqdm(range(0, npo * 2 ** N_interp, chunk_prediction_size)):
        if mad_scaling:
            # !!!!! HUMAN DECISIONs !!!!!
            scaling = 0.6745 / mad(input_spec[i:i+chunk_size]) # The MAD of Gaussian white noise is 0.675
            if scaling > 1: # only apply to raised noise levels
                scaling = 1
        else:
            scaling = 1
        chunks.append([input_wn[i:i+chunk_size], input_spec[i:i+chunk_size] * scaling])#, conv_spec[i:i+chunk_size] * scaling])
    chunks = np.array(chunks[:-1]) # the final chunk is length chunk_prediction_size, does not fit a tensor, so remove it since it does not need to be evaluated as its at the end
    # chunks shape is (npo/chunk_prediction_size - 1, 3, chunk_size)
    
    chunks = chunks.transpose(0, 2, 1) # Shape (total_chunks, chunk_size, 3) the final dim index as [wn, spec, smoothed_spec]
    
    # Extract predictions using the overlapping window method, this might require chunk_size to be a multiple of 2
    chunks = chunks[:, int(chunk_prediction_size/2):-int(chunk_prediction_size/2) , :]
    
    idx = (chunks[:, :, 0] > line_region[0]) & (chunks[:, :, 0] < line_region[1])
    chunks_relevant = chunks[idx]
    
    if plot:
        fig, (legend_ax, ax) = plt.subplots(2, 1, figsize=(12,6), gridspec_kw={'height_ratios': [1, 20]})
        
        l1, = ax.plot(wn, spec, 'r', lw=1, label='original exp. spectrum')
        l2, = ax.plot(chunks_relevant[:, 0], chunks_relevant[:, 1], 'gray', marker='+', lw=2, label='stitched (and noise scaled if MAD=True) spec in line region')
        ax.set_xlabel('wn (/cm)')
        ax.set_ylabel('snr')
        legend_ax.axis("off")
        handles = [l1, l2]#, l3]
        labels = [line.get_label() for line in handles]
        legend_ax.legend(handles=handles, labels=labels, loc="center", ncol=len(labels))
        plt.tight_layout()
    
    # Not returning interpolated points because we are only estimating widths and snr distributions
    return chunks_relevant[:, 0][::int(2**N_interp)], chunks_relevant[:, 1][::int(2**N_interp)]



def fit_spec(wn, spec, line_pos, line_height, Gw_grad, Gw_cept, Lw_KDE, exp_res, blend_thresh=2, min_snr=2, plot=False):
    '''
    Fits the spectrum using asymmetric Voigt profiles given initial guesses for wn pos (line_pos) and peak pos (line_height)
    blend_tresh is multiples of FWHM, more means more lines per fit, which could be slow if line density is very high!
    min_snr here is the lower bound of fitted snr values
    '''
    # Initial guess for Lw
    if isinstance(Lw_KDE, float):
        Lw = Lw_KDE
    else:
        Lw= np.mean(Lw_KDE.sample(1000))
    
    # Rough FWHM estimate
    fwhm = voigt_fwhm(Gw_grad * line_pos + Gw_cept, Lw)
    
    # Group lines closer than blend_thresh to fit together
    line_groups = []
    for i, line_wn in enumerate(line_pos):
        if i == 0:
            line_groups += [[line_wn, Gw_grad * line_wn + Gw_cept, Lw, np.max([line_height[i], min_snr]), 0]] # new group of line parameters
        else:
            prev_line_wn = line_groups[-1][-5]
            if (line_wn - prev_line_wn) < blend_thresh * fwhm[i]: # if likely blended, fit together
                line_groups[-1] = line_groups[-1] + [line_wn, Gw_grad * line_wn + Gw_cept, Lw, np.max([line_height[i], min_snr]), 0]
            else:
                line_groups += [[line_wn, Gw_grad * line_wn + Gw_cept, Lw, np.max([line_height[i], min_snr]), 0]] # new group of line parameters
    
    print('------------------------------------------------')
    print('Using NN output as initial guesses for spectrum fitting')
    popt = []
    for g in tqdm(line_groups):
        # Make sure bounds cover the lines! (difficult for self-absorbed/wide gas lines)
        wn_lb = g[0] - 2 * blend_thresh * voigt_fwhm(g[1] * Gw_grad + Gw_cept, Lw) # wn lower bound
        wn_ub = g[-5] + 2 * blend_thresh * voigt_fwhm(g[-4] * Gw_grad + Gw_cept, Lw) # wn upepr bound
        # Fit within bounds, otherwise too many parameters and pnts
        wn_idx = (wn > wn_lb) & (wn < wn_ub)
        x = wn[wn_idx]
        y = spec[wn_idx]
        
        # Initialise lists for parameter bounds
        param_lb = []
        param_ub = []
        for p in range(0, len(g), 5):
            p_wn = g[p]
            p_Gw = g[p+1]
            #p_Lw = g[p+2]
            p_snr = g[p+3]
            #p_a = g[p+4]
            if p_snr < 10:
                asy_max = 1e-6
            else:
                asy_max = 10
            # !!!!! HUMAN DECISION !!!!! parameter bounds
            param_lb += [p_wn-voigt_fwhm(p_Gw, Lw)/3, p_Gw / 2, 0, min_snr, -asy_max]
            param_ub += [p_wn+voigt_fwhm(p_Gw, Lw)/3, p_Gw * 2, Lw * 10, (p_snr + 2) * 2, asy_max]
            # Sometimes fit cannot converge, so skip them and just keep the guesses
        try: # Fitting this group fo lines
            group_popt, _ = curve_fit(add_asy_voigts, x, y, p0=g, bounds=([param_lb, param_ub]))
        except RuntimeError as e: # If fit does not converge
            print(f"Optimal parameters not found for line(s) between {wn_lb:.3f} and {wn_ub:.3f}. Skipping and params are kept as original guesses... Error: {e}")
            group_popt = np.array(g)
        
        popt += group_popt.tolist()
    
    popt = np.reshape(popt, (-1, 5))
    
    print('------------------------------------------------')
    print('Extracting line list from optimal parameters')
    fitted_spec = np.zeros_like(wn)
    
    # Calculate COG wns for significantly asymmetric lines
    eqw = []
    wn_cog = []
    for p in tqdm(popt):
        wn_idx = (wn > (p[0] - 5)) & (wn < (p[0] + 5))
        x = wn[wn_idx]
        line_prof = asy_voigt(x, p[0], p[1], p[2], p[3], p[4])
        fitted_spec[wn_idx] += line_prof
        area = si.simpson(y=line_prof, x=x)
        cog = si.simpson(y=x*line_prof, x=x) / area
        eqw.append(area)
        wn_cog.append(cog)
    
    popt = popt.T
    
    if plot:
        plt.figure()
        plt.plot(wn, spec, 'k', lw=2, label='spec')
        plt.plot(wn, fitted_spec,'tab:green', lw=1, label='fitted spec')
        plt.vlines(wn_cog, 0, popt[3], 'tab:green', label='cog wn')
        plt.xlabel('wn (/cm)')
        plt.ylabel('snr')
        plt.legend(loc='upper right')
    
    # Statistical WN uncertainties
    wn_stat_unc = ((abs(popt[4]) > 1).astype(int) + 1) * np.sqrt(voigt_fwhm(popt[1], popt[2]) * exp_res) / np.clip(popt[3], 0, 100)
    
    linelist = pd.DataFrame()
    linelist['snr'] = popt[3]
    linelist['fwhm'] = voigt_fwhm(popt[1], popt[2])
    linelist['wn'] = popt[0]
    linelist['wn_cog'] = wn_cog
    linelist['wn_stat_unc'] = wn_stat_unc
    linelist['eqw'] = eqw
    return linelist, fitted_spec