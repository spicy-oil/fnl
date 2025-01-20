#%%
import fts_nn_ll as nl
import numpy as np
import time
import matplotlib
matplotlib.use('TkAgg') # For interactive plotting

def between(t1, t2):
    '''
    For readable timings
    '''
    return round((t2 - t1) / 60, 1)


#%% Step 1 - Configuration
############################################################

# Experimental parameters
exp_res = 0.0364008  # experimental spectral resolution decided by largest optical path difference
start_wn = 19472.7898759 # starting wn
npo = 466944 # number of points in spectrum
delw = 0.0177391757119 # /cm distance between each point in the spectrum
file = './data/example_spec.npy' # Get the spectrum
scale_factor = 1 / .9e6 # to convert to snr, use 1 if already in snr scale
spec = np.load(file) * scale_factor
# if npo is not a power of 2, uncomment next line and use pad_spec()
spec, npo, delw = nl.pad_spec_to_power_of_2(spec, npo, delw)
wn = start_wn + np.arange(npo) * delw # wn (cm-1), 1D-numpy array
line_region = [20801,  26000] # Within which we wish to extract lines, e.g., if alias 1, spec around wn = 0 would be bad, so please change this accordingly
fwhm = 0.1 # guess fwhm for the lines for scanning the spectrum for simulation parameters, this also determines Gw and Lw fit boundaries in the scan (see scan.py)!


# NN training parameters
snr_min = 3 # minimum snr for a simulated line to be detectable
N_interp = 0 # log_2 of the factor of increase in the number of points from Fourier interpolation (0 = no interp, 1 = 2x, 2 = 4x, etc.)
chunk_size = 1024 * int(2 ** N_interp) # post interpolation chunk size for training (cutting interpolated spectrum into pieces of length chunk_size)
batch_size = 256 # number of chunks per loss function calculation
num_epochs = 888 # maximum number of times trained over all simulation data, early stopping is implemented so training might not reach this number of epochs
lr = 0.001 # ADAM optimiser learning rate
hidden_size = 64 # No. of LSTMs per layer per direction

# Prediction and linelist extraction parameters
mad_scaling = True # Scales noise level greater than 1 to 1 using median absolute deviation of each chunk, not recommended for high line densities
peak_prob_th = 0.5 # threshold probability to be identified as possible peak position
blend_th = 3 # multiple of approx. FWHM to determine whether lines should be fitted together, more means more lines per fit, but could be slow if line density is very high! Reduce if 


# Obtain the spectrum to be scanned for width and snr distributions
scan_wn, scan_spec = nl.process_experimental_spec_for_model(wn, spec, npo, line_region, mad_scaling, N_interp, chunk_size, plot=True)
matplotlib.pyplot.show()



#%% Step 2 - Get properties using lines in specified line_region
############################################################
spec_scan_time = time.time()

# Assuming Gw and Lw dominates line profile by fitting Voigt profiles
# line_den_10 is lien density estimate for lines above 10 SNR, there are likely a few factor more lines below 10 SNR than the number of lines > 10 SNR 
Gw_grad, Gw_cept, Lw_KDE, snr_hist, snr_bins, line_den_10 = nl.get_experimental_spec_properties(scan_wn, scan_spec, fwhm=fwhm, plot=True)
matplotlib.pyplot.show()

print('Spec scan time ', between(spec_scan_time, time.time()), ' mins')



#%% Step 3 - Generating spectra for training
############################################################
spec_sim_time = time.time()

s = nl.SpecSimulator(
    exp_res=exp_res, # From manual config in step 1
    start_wn=start_wn, # From manual config in step 1
    npo=npo, # From manual config in step 1
    delw=delw, # From manual config in step 1
    line_region=line_region, # From manual config in step 1
    N_interp=N_interp, # From manual config in step 1
    Gw_grad=Gw_grad, # From nl.get_properties() in scan.py in step 2, or specify manually
    Gw_cept=Gw_cept, # From nl.get_properties() in scan.py in step 2, or specify manually
    Gw_std=0.1, # From human viewing of plot from From nl.get_properties() in step 2, specify manually
    Lw_KDE=Lw_KDE, # From nl.get_properties() in scan.py in step 2, or specify a float (cm-1) manually to sample uniformly at +- 100% of the value
    snr_hist=[snr_hist, snr_bins], # From nl.get_properties() in scan.py  in step 2
    snr_min=snr_min # From manual config in step 1
    ) 

N_specs = 4 # Train with chunks from 4 simulated specs
# Specifying line_den as 20 times the line density of lines above 10 snr
line_den = line_den_10 * 20
# XY has shape (3, npo * N_specs), index as [wn, spec, class], linelist shape (4, number_of_lines) index as [wn, snr, Gw, Lw]
XY, _ = s.xy(line_den=line_den, plot=True) # Each s.xy() call simulates a different spectrum using the same wn axis, snr and width distributions
matplotlib.pyplot.show() # Plot this one as an example, may need to close these plots to continue simulation (next two lines)
for i in range(N_specs - 1):
    XY = np.concatenate((XY, s.xy(line_den=line_den, plot=False)[0]), axis=1)

print('Spec simulation time ', between(spec_sim_time, time.time()), ' mins')



#%% Step 4 - Initiate and train model, the final model will be the one with the lowest loss
############################################################
nn_training_time = time.time()

model = nl.train_model(XY, line_region, model=None, chunk_size=chunk_size, batch_size=batch_size, num_epochs=num_epochs, lr=lr, plot=True) 
# model=None initialises new model, otherwise can continue training for an existing model of class PeakIdentifier from nn.py
matplotlib.pyplot.show() # Plot an example simulation chunk evaluation

print('NN training time ', between(nn_training_time, time.time()), ' mins')


#%% Step 5- Use trained model to predict line positions
############################################################
nn_prediction_time = time.time()

# line_pos are the class 1 positions, line_height are the corresponding snrs
# wn_for_pred is the wn axis used for predictions (varies with N_interp)
# peak_prob is prob. curve for class 1
line_pos, line_height, wn_for_pred, peak_prob = nl.get_model_predictions(model, batch_size, wn, spec, npo, exp_res, line_region, chunk_size, N_interp, peak_prob_th, mad_scaling, plot=True)
matplotlib.pyplot.show() # Plot shows detected lines in spec and peak probabilities

print('NN prediction and plotting time ', between(nn_prediction_time, time.time()), ' mins')



#%% Step 6 - Fit the spectrum using Voigt profiles to obtain line list
############################################################
linelist_extraction_time = time.time()

linelist, fitted_spec = nl.fit_and_get_linelist(wn, spec, line_pos, line_height, Gw_grad, Gw_cept, Lw_KDE, exp_res, blend_th, plot=True)
matplotlib.pyplot.show() # Plot shows the fitted spectrum

# If needs saving (linelist is pd DataFrame, fitted_spec is 1D np array)
linelist.to_excel('./data/example_spec_linelist.xlsx', index=False) # save line list
np.save('./data/example_spec_voigt_fitted.npy', fitted_spec)

print('Spec linelist extraction time ', between(linelist_extraction_time, time.time()), ' mins')