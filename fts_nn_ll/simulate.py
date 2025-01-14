'''
Spectrum simulation
'''

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.fft import fft, ifft, fftshift
from .functions import voigt, preprocess_interp

class spec_gen():
    '''
    Spectra simulation based on experimental and spectrum scan parameters
    '''
    def __init__(self, exp_res=None, start_wn=None, npo=None, delw=None, line_region=[None, None], N_interp=0,
                 Gw_grad=None, Gw_cept=None, Gw_std=0.3, Lw_KDE=None, snr_hist=[None, None], snr_min=3):
        '''        
        snr_hist minimum SNR = 10, i.e. must have a bin covering 10 SNR
        Assumes delw < exp_res and npo to be a power of 2 because of FFT padding of interferogram, this is required to add instrumental apodisations
        '''
        # Histogram of log10 SNR of line list
        self.hist, self.bins = snr_hist
        self.cdf = self.get_sample_dist()
        
        # Spec Params
        self.exp_res = exp_res # cm-1
        self.start_wn = start_wn # cm-1
        self.npo = npo # number of points
        self.delw = delw # cm-1
        N = np.log2(npo)
        if N % 1 != 0: # if is not a power of 2
            print('Warning, npo is not a power of 2, please pad interferogram to the next power of 2 and transform into spectrum')
        self.wn = self.start_wn + np.arange(npo) * delw # cm-1 the wn axis of the experimental spectrum
        self.end_wn = self.wn[-1] # cm-1\
        self.spec_range = self.end_wn - start_wn
        self.line_region = line_region # no lines will be simulated outside this region, only a small number of chunks with pure noise regions will be trained on to avoid bias.
        self.snr_min = snr_min
        
        self.Gw_grad = Gw_grad # cm-1 per cm-1
        self.Gw_cept = Gw_cept # cm-1
        self.Gw_std = Gw_std # STD of a normal dist. centered at 1
        self.Lw_KDE = Lw_KDE # kernel distribution function of Lw against Gw
        
        # The threshold snr above which instrumental functions are simulated, 
        self.instrum_th = 2
        
        # log_2 of the multiple of npo to interpolate through FT padding
        self.N_interp = N_interp

    def init_lines(self, line_den=1):
        '''
        Sample line wn and their snr
        line_den is the avg number of lines per wn
        '''
        # Uniformly sample WN within region using number of lines determined from line density
        self.wn_lb = np.max([self.start_wn, self.line_region[0]]) + 5 # lower bound wn of the wn range to sample
        self.wn_ub = np.min([self.end_wn, self.line_region[1]]) - 5 # upper bound of the wn range to sample
        N = int((self.wn_ub - self.wn_lb) * line_den) # number of lines to be sampled        
        # Ensure 1 line from highest SNR bin, and 1 from second highest, and 1 from third highest... and so on until N_high_snr'th highest snr bin
        N_high_snr = 10 
        # Sample line wns
        self.line_wn = np.random.uniform(self.wn_lb, self.wn_ub, N + N_high_snr)
        # Ensure N_high_snr lines are from highest SNRs
        highest_snrs = [10 ** self.bins[-(i+1)] for i in range(N_high_snr)] 
        # Sample line snrs
        self.line_snr = np.array([self.sample_snr() for i in range(N)] + highest_snrs) 
    
        # Sample line Gw
        # !!!!! HUMAN DECISION !!!!! for width of the normal distribution, need to view Gw-wn plot from get_properties of scan.py
        std = self.Gw_std # ~30% unc tends to work, but really depends on Gw variations in the spec (see scan.py plots)
        self.line_Gw = (self.line_wn * self.Gw_grad + self.Gw_cept) * np.abs(np.random.normal(1, std, N + N_high_snr)) # make sure positive
        
        # Use Lw KDE to sample Lw, or specified Lw value
        if isinstance(self.Lw_KDE, float):
            self.line_Lw = self.Lw_KDE * np.abs(np.random.uniform(N + N_high_snr))
        else:    
            self.line_Lw = []
            for Gw in self.line_Gw:
                samples = abs(self.Lw_KDE.sample(1000)) # 1000 sample using KDE
                # Get sampled Lws within +- 0.001 cm-1 window of the sampled Gw
                abs_diff = np.abs(samples[:, 0] - Gw)
                possible_Lw = samples[abs_diff < 0.001][:, 1]
                if len(possible_Lw) > 0:
                    self.line_Lw.append(np.random.choice(possible_Lw)) # Randomly pick a Lw in the window
                else: # For outliers and rare occasions where none got sampled within 1 mK
                    self.line_Lw.append(samples[:, 1][np.argmin(abs_diff)]) # Pick the closest one (in terms of Gw)
            self.line_Lw = np.array(self.line_Lw)
        
        # !!!!! HUMAN DECISION !!!!!
        # Weak lines (snr < 5) close together might actually contribute together as a single detectable weak blend
        # Problematic if just setting a min. detectable snr, so removing such cases
        weak_th = 5
        
        # Sort line list by wn
        sorted_indices = np.argsort(self.line_wn)
        self.line_wn = self.line_wn[sorted_indices]
        self.line_Gw = self.line_Gw[sorted_indices]
        self.line_Lw = self.line_Lw[sorted_indices]
        self.line_snr = self.line_snr[sorted_indices]
        
        # Combine weak lines that are too close
        combined_wn = []
        combined_Gw = []
        combined_Lw = []
        combined_snr = []
        i = 0
        while i < len(self.line_wn):
            current_cog = self.line_wn[i]
            current_Gw = self.line_Gw[i]
            current_Lw = self.line_Lw[i]
            current_snr = self.line_snr[i]      
            # Continue combining while the next number is within the threshold
            # !!!!! HUMAN DECISION !!!!! for combine width (self.exp_res * 4) and weak_th!!!
            while i + 1 < len(self.line_wn) and abs(self.line_wn[i + 1] - current_cog) <= (self.exp_res * 4) and self.line_snr[i + 1] < weak_th:
                tot_snr = current_snr + self.line_snr[i + 1]
                current_cog = (current_cog * current_snr + self.line_wn[i + 1] * self.line_snr[i + 1]) / tot_snr
                current_Gw = (current_Gw * current_snr + self.line_Gw[i + 1] * self.line_snr[i + 1]) / tot_snr
                current_Lw = (current_Lw * current_snr + self.line_Lw[i + 1] * self.line_snr[i + 1]) / tot_snr
                current_snr = tot_snr
                i += 1 # Skip the lines combined
            
            # Append the combined value (average of combined numbers) to the result list
            combined_wn.append(current_cog)
            combined_Gw.append(current_Gw)
            combined_Lw.append(current_Lw)
            combined_snr.append(current_snr)
            i += 1
        
        # New line list for simulation
        self.line_wn = np.array(combined_wn)
        self.line_Gw = np.array(combined_Gw)
        self.line_Lw = np.array(combined_Lw)
        self.line_snr = np.array(combined_snr)

    def get_sample_dist(self, N = 5):
        '''
        Create SNR distribution to sample from by linear extrapolating the input histogram
        N is running average bin width to smooth the histogram values
        '''
        self.hist = np.array(self.hist, dtype=float)
        self.bins = np.array(self.bins, dtype=float)
        
        # Use straight line to extend histogram to zero, because weak lines are not always fitted.
        def line(x, a, b): # straight line
            return a * x + b
        
        # Fit the line to the bins near x = 1 (snr = 10 because log10)
        fit_bins = self.bins[self.bins > 1.][:-1] 
        fit_hist = self.hist[self.bins[:-1] > 1.]
        params, _ = curve_fit(line, fit_bins, fit_hist)
        
        # Replace bin values from 0 to 1 (0 to 10 SNR)
        self.hist[self.bins[:-1] <= 1.] = line(self.bins[self.bins <= 1.], *params)
        
        # Extend, smooth, then cut bins below 0
        bins_to_zero_snr = np.ceil(self.bins[0] / np.diff(self.bins)[0])
        extra_bins = (self.bins[0] - (np.arange(bins_to_zero_snr + 10) + 1) * np.diff(self.bins)[0])[::-1] # 10 is for smoothing, will be cut
        hist_ex = np.concatenate((line(extra_bins, *params), self.hist))
        # Smooth histogram
        self.hist = np.convolve(hist_ex, np.ones(N)/N, mode='same')[10:]
        self.new_bins = np.concatenate((extra_bins[10:], self.bins))
        # Normalize the new histogram 
        self.hist /= np.sum(self.hist)
        # Compute the CDF
        return np.cumsum(self.hist)

    def sample_snr(self):
        '''
        Sample a snr from the extrapolated histogram
        '''
        u = np.random.rand()  # Uniformly distributed random number between 0 and 1
        idx = np.searchsorted(self.cdf, u)  # Find index where CDF crosses u
        return 10 ** np.random.uniform(self.new_bins[idx], self.new_bins[idx+1])  # Sample uniformly within corresponding interval

    def add_instrumental_effects(self, spec, line_wn=None, cos_bell_ratio=0.1, exp_res=None, spec_trunc_size = 2048, plot=False):
        '''
        Add systematic instrumental effects: finite OPD, aperture, and 2^N padding from FFT to a line at line_wn
        cos_bell_ratio = 0.1 (two times the a parameter from paper) default for xgremlin, 10% points in OPD are a part of a cosine bell over 2pi
        no noise
        Can input entire spec with many lines and use an average line_wn to see roughly what the spec looks like with instrumental effects.
        '''
        old_spec = spec
        spec_trunc_size = spec_trunc_size * int(2 ** self.N_interp) # In case of Fourier interpolation
        
        # Truncate spectrum to region of the line, ideally spec_trunc_size contains all ringing > ~1 snr depending on exp_res
        if spec_trunc_size is not None:
            line_wn_idx = np.argmax(self.wn > line_wn)
            spec_trunc = spec[int(line_wn_idx - spec_trunc_size / 2) : int(line_wn_idx + spec_trunc_size / 2)]
        
        # If not wishing to specify max. OPD, then exp_res will be used for the max. OPD 
        if exp_res is None:
            exp_2L = 1 / self.exp_res
        else:
            exp_2L = 1 / exp_res
            
        sim_2L = 1 / self.delw
        
        if spec_trunc_size is not None:
            sim_dx = sim_2L / spec_trunc_size 
            inter = fftshift(fft(spec_trunc))
            x = -sim_2L / 2 + np.arange(spec_trunc_size) * sim_dx # Use spec_trunc size to not FT 10^5-10^6 pnts for each line, otherwise simulation is too slow
        else:
            sim_dx = sim_2L / self.npo
            inter = fftshift(fft(spec))
            x = -sim_2L / 2 + np.arange(self.npo) * sim_dx # OPD axis of 2^N padded interferogram
        
        if plot == True:
            plt.figure()
            plt.plot(x, np.real(inter), color='k', label='interferogram')

        # Generate apodisation function on the new OPD axis
        self.apo_func = np.ones_like(inter)

        # Expected experimental intergerogram aperture sinc apodisation
        O_max = 2 * np.pi * self.exp_res / self.end_wn 
        if plot == True:
            plt.plot(x, np.sinc(line_wn * O_max * x / 2 / np.pi) * np.max(np.real(inter)), color='m', label='aperture sinc')
        self.apo_func *= np.sinc(line_wn * O_max * x / 2 / np.pi) # sinc apodisation, applying wn depedence for each line

        # Expected cosine bell for box apodisation to reduce ringing
        outside_opd = (x < -exp_2L / 2) | (x > exp_2L / 2)
        self.apo_func[outside_opd] = 0 # Box apodisation from finite OPD
        if cos_bell_ratio != 0:
            npo_inside_opd = (outside_opd == False).sum()
            one_side_npo = npo_inside_opd * cos_bell_ratio / 2 # left and right are separate, e.g. 10% cosine bell means 5% on each side
            # left side
            x_left_start_idx = int(np.where(np.abs(x + exp_2L / 2) == np.abs(x + exp_2L / 2).min())[0][0])
            x_left_end_idx = int(x_left_start_idx + one_side_npo)
            x_left = x[x_left_start_idx : x_left_end_idx]
            cosine_start = x_left[0]
            cosine_end = x_left[-1]
            cosine_k = np.pi / (cosine_end - cosine_start)
            cosine_x0 = cosine_end
            cosine_left = np.cos(cosine_k * (x_left - cosine_x0)) / 2 + 0.5
            self.apo_func[x_left_start_idx : x_left_end_idx] *= cosine_left # cosine bell apodisation
            # right side
            x_right_end_idx = int(np.where(np.abs(x - exp_2L / 2) == np.abs(x - exp_2L / 2).min())[0][0])
            x_right_start_idx = int(x_right_end_idx - one_side_npo)
            x_right = x[x_right_start_idx : x_right_end_idx]
            cosine_start = x_right[0]
            cosine_end = x_right[-1]
            cosine_k = np.pi / (cosine_end - cosine_start)
            cosine_x0 = cosine_start
            cosine_right = np.cos(cosine_k * (x_right - cosine_x0)) / 2 + 0.5
            self.apo_func[x_right_start_idx : x_right_end_idx] *= cosine_right # cosine bell apodisation

        # Apodise interferogram
        inter *= self.apo_func
        if (plot == True) & (cos_bell_ratio != 0):
            plt.plot(x_left, cosine_left*np.max(np.real(inter)), color='b', label='cos bell')
            plt.plot(x_right, cosine_right*np.max(np.real(inter)), color='b')
            
        # Inverse FT for spec with systematic instrumental effects
        new_spec = ifft(fftshift(inter)) # if npo reduced by a factor of 2, so would SNR, but npo should be the same
        if spec_trunc_size is not None:
            output_spec = np.zeros_like(self.wn)
            output_spec[int(line_wn_idx - spec_trunc_size / 2) : int(line_wn_idx + spec_trunc_size / 2)] += np.real(new_spec)
        else:
            output_spec = np.real(new_spec)
        if plot == True:
            plt.plot(x, np.real(inter), color='r', label = 'instrumental interferogram')
            plt.legend(loc='lower right')
            plt.xlabel('Optical path difference (cm)')
            plt.ylabel('Intensity (arb.)')
            plt.figure()
            plt.plot(self.wn, old_spec, 'b-', label='lamp spectrum')   
            plt.plot(self.wn, output_spec, 'r-', label ='instrumental spectrum')
            plt.legend(loc='upper right')
        return output_spec

    def make_spec(self, line_den=1, plot=False):
        '''
        Simulate spectrum
        '''
        self.spec = np.zeros_like(self.wn)
        self.noise = np.random.normal(size=len(self.spec)) # Fixed noise beforehand in case we need to know lines are observable
        
        self.init_lines(line_den=line_den)
        
        # Adding strongest lines first (sort by snr), they have most significant ringing and are most likely detectable
        sorted_indices = np.argsort(self.line_snr)[::-1] 
        self.line_wn = self.line_wn[sorted_indices]
        self.line_Gw = self.line_Gw[sorted_indices]
        self.line_Lw = self.line_Lw[sorted_indices]
        self.line_snr = self.line_snr[sorted_indices]
        
        self.target_idx = [] # indices of lines that are identifiable and to be trained to be detectable
        
        print('------------------------------------------------')
        print('Simulating spectrum')
        # Don't calculate Voigt outside +- 20 cm-1 of the line to save resources
        # May need to be increased in extremely low max OPD (large resolution-limited ringing) situations
        width = 20
        for i in tqdm(range(len(self.line_snr))): # for each line in descending SNR order
            y = np.zeros_like(self.wn)
            wn = self.line_wn[i]
            snr = self.line_snr[i]
            Gw = self.line_Gw[i]
            Lw = self.line_Lw[i]
            mask = (self.wn > (wn - width)) & (self.wn < (wn + width))
            wn_to_calc = self.wn[mask]
            # Update y only in the masked regions
            y[mask] += voigt(wn_to_calc, wn, Gw, Lw, snr)
            
            if snr > self.instrum_th: # if above threshold snr for simulating instrumental effects
                # now need to add instrumental effects to y, which will be on the spectrum wn axis
                y_instrum = self.add_instrumental_effects(y, line_wn=wn)
                snr_reduction_factor = np.max(y_instrum) / np.max(y) # SNR/peak values get reduced after adding instrumental effects
                self.line_snr[i] *= snr_reduction_factor
                snr = self.line_snr[i]
            else:
                y_instrum = y
                
            # Now need to add to spec, compare with spec simulated so far, and decide whether detectable
            # Locate idx of the cloests pnt to line wn
            wn_idx = np.argmax(wn < self.wn)
        
            # Isolated weak lines vs. noise
            # !!!!! HUMAN DECISION !!!!!
            if snr < self.snr_min: # S/N_min in paper
                self.spec += y_instrum # add to spectrum but not to target
                continue # Not a detectable line, but is part of the spectrum, skip to next line
                 
            # Has neighbourhood changed significantly?
            neighbourhood = self.spec[max(0, wn_idx-2):wn_idx+2].copy() #  +  + *+  +, where + are spectrum pnts, * is the wn between spectrum pnts
            self.spec += y_instrum # Add to spectrum regardless of whether detectable 
            # !!!!! HUMAN DECISION !!!!!
            if snr < (1/5) * np.abs(neighbourhood).max():
                continue # Not a detectable line, but is part of a much stronger line, skip to next line
            
            # Passed the snr thresh and blending with stronger line test, so add to detectable list
            self.target_idx.append(i)
            
        # Adding noise after all lines are added
        self.spec += self.noise
        
        # For debugging and plotting, remove if computer memory is struggling
        self.line_wn_gen = self.line_wn
        self.line_snr_gen = self.line_snr
        
        # Apply targetable indices and also COG close lines, can get self.line_wn and self.line_snr back using e.g. self.line_wn_gen[self.target_idx]
        self.line_wn = self.line_wn[self.target_idx]
        self.line_Gw = self.line_Gw[self.target_idx]
        self.line_Lw = self.line_Lw[self.target_idx]
        self.line_snr = self.line_snr[self.target_idx]       
        self.cog_close_lines(1)
        # self.find_blend()
        
        if plot:
            self.plot_spec()

    def plot_spec(self):
        plt.figure()
        plt.plot(self.wn, self.spec, 'gray', lw=2)
        plt.vlines(self.line_wn_gen, ymin=0, ymax=self.line_snr_gen, color='k', linestyle='-', lw=2, label='1 - Generated lines')
        plt.vlines(self.line_wn, ymin=0, ymax=self.line_snr, color='gray', linestyle='-', lw=2, label='2 - Lines detectable in noise and blend')
        plt.vlines(self.line_wn_target, ymin=0, ymax=self.line_snr_target, color='b', linestyle='-', lw=2, label='3 - close lines combined')
        plt.xlabel('wn (/cm)')
        plt.ylabel('snr')
        plt.legend(loc='upper right')
        
    def cog_close_lines(self, combine_width=1):
        '''
        Combine lines that are too close (closer than combine_width * self.exp_res)
        Should happen after adding lines to spectrum in self.make_spec
        Different to line combination before adding lines to the spectrum (no weak_th)
        '''
        # Sort lines by wn
        sorted_indices = np.argsort(self.line_wn)
        self.line_wn = self.line_wn[sorted_indices]
        self.line_Gw = self.line_Gw[sorted_indices]
        self.line_Lw = self.line_Lw[sorted_indices]
        self.line_snr = self.line_snr[sorted_indices]     
        
        # Combine lines that are too close
        combined_wn = []
        combined_Gw = []
        combined_Lw = []
        combined_snr = []
        i = 0
        print('------------------------------------------------')
        print('Combining and COGing significant lines generated closer than resolution')
        while i < len(self.line_wn):
            current_cog = self.line_wn[i]
            current_Gw = self.line_Gw[i]
            current_Lw = self.line_Lw[i]
            current_snr = self.line_snr[i]      
            # Continue combining while the next number is within the threshold
            # !!!!! HUMAN DECISION !!!!! for combine width
            while i + 1 < len(self.line_wn) and abs(self.line_wn[i + 1] - current_cog) <= (self.exp_res * combine_width):
                tot_snr = current_snr + self.line_snr[i + 1]
                current_cog = (current_cog * current_snr + self.line_wn[i + 1] * self.line_snr[i + 1]) / tot_snr
                current_Gw = (current_Gw * current_snr + self.line_Gw[i + 1] * self.line_snr[i + 1]) / tot_snr
                current_Lw = (current_Lw * current_snr + self.line_Lw[i + 1] * self.line_snr[i + 1]) / tot_snr
                current_snr = tot_snr
                i += 1 # skip the lines combined
            
            # Append the combined value (average of combined numbers) to the result list
            combined_wn.append(current_cog)
            combined_Gw.append(current_Gw)
            combined_Lw.append(current_Lw)
            combined_snr.append(current_snr)
            i += 1
        
        # New line list
        self.line_wn_target = np.array(combined_wn)
        self.line_Gw_target = np.array(combined_Gw)
        self.line_Lw_target = np.array(combined_Lw)
        self.line_snr_target = np.array(combined_snr)
        print('Simulated line density =', round(len(self.line_wn_target) / (self.wn_ub - self.wn_lb), 3))
                    
    def xy(self, line_den=1, plot=False):
        '''
        Line density (no. of lines per /cm) should be slightly higher than exp spectra to make it harder for the NN,
        to simulate weak lines not in human line lists, 
        and so that it sees many different cases of line blending,
        but not too high to make the NN start to expect blends everywhere.
        10 times line density of lines above 10 snr appear reasonable
        
        Returns x and y for training
        xy: [spec_wn, spec, class_array, loss_fn_weights] Fourier interpolated
        linelist: [line_wn, line_snr, line_Gw, line_Lw] 
        Classes:
            1 - closest point to line_wn
            0 - not a point closest to line_wn
        '''
        self.make_spec(line_den=line_den, plot=plot)
        
        # Interpolation
        # !!!!! HUMAN DECISION !!!!! Fourier interpolation by integer number of power of 2 points
        N = self.N_interp
        spec_wn, spec = preprocess_interp(self.wn, self.spec, N)
        
        print('------------------------------------------------')
        print('Creating class array')
        class_array = np.zeros_like(spec_wn)
        for wn in tqdm(self.line_wn_target):
            idx = np.argmin(abs(spec_wn - wn))
            class_array[idx] = 1 # closest pnt gets marked as peak
        
        xy = [spec_wn, spec, class_array]
        linelist = [self.line_wn_target, self.line_snr_target, self.line_Gw_target, self.line_Lw_target]
        if plot:
            plt.figure()
            plt.plot(spec_wn, spec, 'k+-', lw=1, label='Interpolated simulated spectrum')
            plt.scatter(spec_wn[class_array == 1], spec[class_array == 1], color='green', marker='^', s=150, label='Point closest to wn')
            plt.xlabel('wn (/cm)')
            plt.ylabel('snr')
            plt.legend(loc='upper right')
            # Gw-wn and Lw-Gw analysis for simulated lines
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
            ax1.scatter(self.line_wn_target, self.line_Gw_target, color='k', marker='+', lw=0.5)
            ax1.set_xlabel(r'Wavenumber (cm$^{-1}$)')
            ax1.set_ylabel(r'$G_{\text{w}}$ (cm$^{-1}$)')
            ax1.set_xlim(self.line_wn_target.min(), self.line_wn_target.max())
            ax1.legend(loc='upper left', framealpha=1)
            ax2.scatter(self.line_Gw_target, self.line_Lw_target, color='gray', marker='o', alpha=0.5, label='samples', zorder=-10)
            ax2.legend(loc='upper right', framealpha=1)
            ax2.set_xlabel(r'$G_w$ (cm$^{-1}$)')
            ax2.set_ylabel(r'$L_w$ (cm$^{-1}$)')
            plt.tight_layout()
        return [np.array(xy), np.array(linelist)]
            