import yaml
import time
import numpy as np
import matplotlib
matplotlib.use('TkAgg') # For interactive plotting

from .functions import pad_spec_to_power_of_2, process_experimental_spec_for_model, fit_and_get_linelist
from .scan import get_experimental_spec_properties
from .simulate import SpecSimulator
from .nn import train_model, get_model_predictions


class Manager():
    def __init__(self, config_file):
        '''
        Step 1 - Configuration and preprocess spectrum
        '''
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        file = self.config['file']
        self.spec = np.load(file) * self.config['scale_factor']
        self.npo = len(self.spec)
        self.delw = self.config['delw']
        self.exp_res = self.config['exp_res']
        self.line_region = self.config['line_region']

        # Pad length to power of 2 if needed
        if not self.is_power_of_two(self.npo):
            self.spec, self.npo, self.delw = pad_spec_to_power_of_2(self.spec, self.npo, self.delw)
        
        # Make WN axis
        self.wn = self.config['start_wn'] + np.arange(self.npo) * self.delw # wn (cm-1), 1D-numpy array

        # post interpolation chunk size for training (cutting interpolated spectrum into pieces of length chunk_size)
        self.chunk_size = self.config['chunk_size'] * int(2 ** self.config['N_interp']) 
        
        # Plot preprocess
        self.scan_wn, self.scan_spec = process_experimental_spec_for_model(self.wn, self.spec, 
                                                        self.npo, self.line_region, 
                                                        self.config['mad_scaling'], 
                                                        self.config['N_interp'], 
                                                        self.chunk_size, plot=True)
        matplotlib.pyplot.show()

    def scan(self):
        '''
        Step 2 - Get properties using lines in specified line_region
        '''
        spec_scan_time = time.time()
        # Assuming Gw and Lw dominates line profile by fitting Voigt profiles
        # line_den_10 is lien density estimate for lines above 10 SNR, there are likely a few factor more lines below 10 SNR than the number of lines > 10 SNR 
        tmp = get_experimental_spec_properties(self.scan_wn, self.scan_spec, 
                                                fwhm=self.config['fwhm'], plot=True)
        self.Gw_grad, self.Gw_cept, self.Lw_KDE, self.snr_hist, self.snr_bins, self.line_den_10 = tmp
        print('Spec scan time ', self.between(spec_scan_time, time.time()), ' mins')
        matplotlib.pyplot.show()

    def simulate(self):
        '''
        Step 3 - Simulate spectrum (all stored in memory, cost depends on N_specs and N_interp)
        '''
        spec_sim_time = time.time()

        s = SpecSimulator(
            exp_res=self.exp_res,  # From manual config in step 1
            start_wn=self.config['start_wn'],  # From manual config in step 1
            npo=self.npo,  # From manual config in step 1
            delw=self.delw,  # From manual config in step 1
            line_region=self.line_region,  # From manual config in step 1
            N_interp=self.config['N_interp'],  # From manual config in step 1
            Gw_grad=self.Gw_grad,  # From get_properties() in scan.py in step 2, or specify manually
            Gw_cept=self.Gw_cept,  # From get_properties() in scan.py in step 2, or specify manually
            Gw_std=self.config['Gw_std'],  # Ideally from human viewing of plot from step 2, specify manually
            Lw_KDE=self.Lw_KDE,  # From get_properties() in scan.py in step 2, or specify a float (cm-1) manually to sample uniformly at +- 100% of the value
            snr_hist=[self.snr_hist, self.snr_bins],  # From get_properties() in scan.py  in step 2
            snr_min=self.config['snr_min']  # From manual config in step 1
            ) 

        # Specifying line_den as X times the line density of lines above 10 snr
        line_den = self.line_den_10 * self.config['line_den_mult']
        # XY has shape (3, npo * N_specs), index as [wn, spec, class], linelist shape (4, number_of_lines) index as [wn, snr, Gw, Lw]
        self.XY, _ = s.xy(line_den=line_den, plot=True) # Each s.xy() call simulates a different spectrum using the same wn axis, snr and width distributions
        for i in range(self.config['N_specs'] - 1):
            self.XY = np.concatenate((self.XY, s.xy(line_den=line_den, plot=False)[0]), axis=1)
        print('Spec simulation time ', self.between(spec_sim_time, time.time()), ' mins')
        matplotlib.pyplot.show() # Plot this one as an example, may need to close these plots to continue simulation (next two lines)

    def train(self, exist_model=None, lr=None, bs=None):
        '''
        Step 4 - Initiate and train model, the final model will be the one with the lowest loss
        '''

        nn_training_time = time.time()

        if not lr:
            lr = self.config['lr']
        if not bs:
            bs = self.config['batch_size']

        self.model = train_model(self.XY, self.line_region, 
                                 model=exist_model, chunk_size=self.chunk_size, 
                                 batch_size=bs, 
                                 num_epochs=self.config['num_epochs'], 
                                 lr=lr, plot=True) 
        # model=None initialises new model, otherwise can continue training for an existing model of class PeakIdentifier from nn.py
        print('NN training time ', self.between(nn_training_time, time.time()), ' mins')
        matplotlib.pyplot.show() # Plot an example simulation chunk evaluation

    def predict(self):
        '''
        Step 5 - Use model to detect lines
        '''
        nn_prediction_time = time.time()
        # line_pos are the class 1 positions, line_height are the corresponding snrs
        # wn_for_pred is the wn axis used for predictions (varies with N_interp)
        # peak_prob is prob. curve for class 1
        tmp = get_model_predictions(self.model, 
                                    self.config['batch_size'], 
                                    self.wn, self.spec, self.npo, 
                                    self.exp_res, self.line_region, 
                                    self.chunk_size, self.config['N_interp'], 
                                    self.config['peak_prob_th'], 
                                    self.config['mad_scaling'], plot=True)
        # These data are generated and used for post-processing
        self.line_pos, self.line_height, self.wn_for_pred, self.peak_prob = tmp
        print('NN prediction and plotting time ', self.between(nn_prediction_time, time.time()), ' mins')
        matplotlib.pyplot.show() # Plot shows detected lines in spec and peak probabilities

    def fit(self, ll_out='./data/example_spec_linelist.xlsx', fit_out='./data/example_spec_voigt_fitted.npy'):
        '''
        Step 6 - Use model detected lines to fit for a line list
        '''
        linelist_extraction_time = time.time()

        linelist, fitted_spec = fit_and_get_linelist(self.wn, self.spec, 
                                                     self.line_pos, self.line_height, 
                                                     self.Gw_grad, self.Gw_cept, self.Lw_KDE,
                                                     self.exp_res, self.config['blend_th'], plot=True)

        # If needs saving (linelist is pd DataFrame, fitted_spec is 1D np array)
        linelist.to_excel(ll_out, index=False)  # save line list
        np.save(fit_out, fitted_spec)
        print('Spec linelist extraction time ', self.between(linelist_extraction_time, time.time()), ' mins')
        matplotlib.pyplot.show() # Plot shows the fitted spectrum

    def is_power_of_two(self, n):
        return n > 0 and (n & (n - 1)) == 0

    def between(self, t1, t2):
        '''
        For readable timings
        '''
        return round((t2 - t1) / 60, 1)