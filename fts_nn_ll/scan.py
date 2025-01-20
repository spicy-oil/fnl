import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
plt.rc('xtick', direction='in')
plt.rc('ytick', direction='in')

from scipy.signal import find_peaks
from .functions import add_voigts, voigt, voigt_fwhm
from scipy.optimize import curve_fit
from tqdm import tqdm
from sklearn.linear_model import HuberRegressor
from sklearn.neighbors import KernelDensity

#%%
def get_experimental_spec_properties(x, y, fwhm=0.1, plot=False):
    '''
    The function used for scanning spectrum for the parameters used as inputs for spec_gen of simulate.py,
    such as SNR, Gw & Lw width distributions

    Parameters
    ----------
    x : 1D numpy array
        Wavenumber axis (/cm)
    y : 1D numpy array
        SNR values
    fwhm : float, optional
        Initial guess for FWHM of spectral line (/cm). The default is 0.1.

    Returns
    -------
    Gw_grad, Lw KDE, snr_hist, snr_bins

    '''
    # find peaks above 10 SNR for reasonably reliable width determinations
    peaks, _ = find_peaks(y, height=(10, np.inf))

    # Get initial parameters
    wn0 = x[peaks]
    snr0 = y[peaks]
    params0 = [] 
    for i, w in enumerate(wn0):
        # Division to convert from FWHM to Gw, guessing Lw as 0.005
        params0 += [w, fwhm / (2 * np.sqrt(2 * np.log(2))), 0.005, snr0[i]] 
        
    print('------------------------------------------------')
    print('Creating initial guess fit to the whole spectrum')
    spec0 = add_voigts(x, *params0)
    residual = y - spec0

    # Fit the spectrum one line (> 10 snr) at a time
    print('------------------------------------------------')
    print('Fitting the spectrum, one line at a time')
    popt = [] # Total list of line parameters
    Gw_lb = fwhm / 10 # Gw lower bound
    Gw_ub = fwhm # Gw upper bound
    Lw_lb = 0 # Lw lower bound
    Lw_ub = fwhm # Lw upper bound
    for i in tqdm(range(0, len(params0), 4)):
        wn_mean = params0[i]
        Gw = params0[i+1]
        Lw = params0[i+2]
        snr = params0[i+3]
        # Fit in a window fully containing the line (+- fwhm) to save resources
        idx = (x > (wn_mean - fwhm)) & (x < (wn_mean + fwhm))
        x_fit = x[idx]
        to_fit = residual[idx] + voigt(x_fit, wn_mean, Gw, Lw, snr) # get the line back
        # In rare occasions, fit for a line cannot converge, so skip them and just keep the guess values
        try:
            # Fit [wn, Gw, Lw, snr] values for line i
            single_line_popt, _ = curve_fit(voigt, x_fit, to_fit, p0=[wn_mean, Gw, Lw, snr], 
                                            bounds=([wn_mean-1, Gw_lb, Lw_lb, snr * 0.6], [wn_mean+1, Gw_ub, Lw_ub, snr * 1.5]))
        except RuntimeError as e:
            print(f"Optimal parameters not found for this line at {wn_mean:.3f}. Skipping and params are kept as original guesses... Error: {e}")
            single_line_popt = np.array([wn_mean, Gw, Lw, snr])
        popt += single_line_popt.tolist() # Add [wn, Gw, Lw, snr] values for line i to the total list of line parameters
        
    print('------------------------------------------------')
    print('Adding the fitted voigts together')
    spec_fit = add_voigts(x, *popt)
    if plot:
        plt.figure()
        plt.title('Spectrum Overview and Initial Fit')
        plt.plot(x, y, 'gray', lw=2, label='spec')
        plt.plot(x, spec0, 'k--', label='initial guess')
        plt.plot(x, spec_fit, 'r--', label='fit')
        plt.xlabel('wn (/cm)')
        plt.ylabel('snr')
        plt.legend(loc='upper right')

    # Deduce distributions
    params = np.reshape(popt, (-1, 4)) 
    
    # Remove Gw and Lw outliers fitted to values within 1mK of boundary values
    params = params[(params[:, 1] > Gw_lb + 0.001) & (params[:, 1] < Gw_ub - 0.001)]
    params = params[(params[:, 2] > Lw_lb + 0.001) & (params[:, 2] < Lw_ub - 0.001)]
    Gws = params[:, 1]
    Lws = params[:, 2]
    Lw = np.median(Lws)
    
    # Linear relationship between Gw and wn, fit with Huber loss
    gw_fit = HuberRegressor(epsilon=1.05, fit_intercept=True) # this is more immune to outliers than MSE as loss function, no intercept because there is no intercept in Doppler width relation
    gw_fit.fit(params[:, 0].reshape(-1, 1), params[:, 1])
    # Attempt with MSE loss
    mse_grad, mse_cept = np.polyfit(params[:, 0], params[:, 1], 1)
    
    Gw_grad = gw_fit.coef_[0] # multiply this number by wn and add cept (next line) to get Gw for the wn
    Gw_cept = gw_fit.intercept_

    
    # Fit a 2D kernel density estimator for Gw and Lw
    # Suitable because Gw and Lw have similar units & size & spread
    widths = np.vstack((Gws, Lws)).T
    # Silverman's Rule for selecting bandwidth (smoothness, the width of the individual 2D Gaussians)
    n = len(Gws)  # Number of data points
    bandwidth = 0.9 * min(np.std(Gws), np.std(Lws)) * n ** (-1/5)
    # KDE and fitting
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth)
    kde.fit(widths)
    
    
    if plot:
        # Gw-wn and Lw-Gw analysis
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
        ax1.scatter(params[:, 0], params[:, 1], color='k', marker='+', lw=0.5)
        x = np.linspace(params[:, 0].min() - 1000, params[:, 0].max() + 2000, 10)
        ax1.plot(x, gw_fit.predict(x.reshape(-1, 1)), 'r-', alpha=1, label='Huber loss')
        ax1.plot(x, mse_grad * x + mse_cept, 'r--', alpha=1, label='MSE loss')
        #print(mse_cept)
        ax1.set_xlabel(r'Wavenumber (cm$^{-1}$)')
        ax1.set_ylabel(r'$G_{\text{w}}$ (cm$^{-1}$)')
        ax1.set_xlim(x.min(), x.max())
        ax1.legend(loc='upper left', framealpha=1)
        n_samples = 1000
        samples = abs(kde.sample(n_samples)) # sample using KDE
        ax2.scatter(params[:, 1], params[:, 2], color='k', marker='+', lw=0.5, label='Exp. samples', zorder=-9)
        ax2.scatter(samples[:, 0], samples[:, 1], color='gray', marker='o', alpha=0.5, label='KDE samples', zorder=-10)
        ax2.legend(loc='upper right', framealpha=1)
        ax2.set_xlabel(r'$G_w$ (cm$^{-1}$)')
        ax2.set_ylabel(r'$L_w$ (cm$^{-1}$)')
        plt.tight_layout()
    
        # Lw-wn analysis, typically no observable trend
        plt.figure()
        plt.title('Lorentzian Width vs. WN Heatmap Scatter Plot')
        plt.hist2d(params[:, 0], params[:, 2], bins=50, cmap='OrRd', norm=mcolors.LogNorm())
        plt.colorbar(label='Log-scaled Count in Bin')
        plt.xlabel('wn (/cm)')
        plt.ylabel('Lw (/cm)')

        # snr analysis
        plt.figure()
        snr_hist, snr_bins, _ = plt.hist(np.log10(params[:, 3]), bins = 50)
        plt.title('SNR Histogram')
        plt.xlabel('log10(snr)')
        plt.ylabel('count')

    else:
        snr_hist, snr_bins = np.histogram(np.log10(params[:, 3]), bins = 50)
    print('------------------------------------------------')
    print('Results')
    print('Gw per wavenumber is ' + '{:.3e}'.format(Gw_grad) + ' with intercept,', round(Gw_cept, 4), ' this is ' + str(round(Gw_grad * x[len(x) // 2] + Gw_cept, 3)) + ' cm-1 in the middle of the spectrum.')
    print('Lw median = ' + '{:.3f}'.format(Lw) + ' cm-1')
    print('FWHM is about', str(round(voigt_fwhm(Gw_grad * x[len(x) // 2] + Gw_cept, Lw), 3)), ' cm-1 in the middle of the spectrum')
    print('Number of lines above 10 SNR = ' + str(len(peaks)))
    line_den = round(len(peaks) / (x[-1] - x[0]), 3)
    print('Estimated line density above 10 SNR is at least ~', line_den, 'lines per cm-1, total is probably a couple of times higher if including lines with < 10 SNR')
    return Gw_grad, Gw_cept, kde, snr_hist, snr_bins, line_den