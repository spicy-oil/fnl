#%%
from fnl import Manager

config_file = './data/config.yaml'

# Each step shows plots for diagnostics

#%% Step 1 - Initialise and preprocess
# The API is a class storing raw + processed data, config parameters, 
# and functions for preprocessing, simulations, training, post-processing
m = Manager(config_file)

#%% Step 2 - Get stats from exp. spec. for simulation
m.scan()

#%% Step 3 - Simulate spectra
m.simulate()

#%% Step 4 - Train NN
m.train(exist_model=None)  # None if training model from scratch

#%% Step 5 - Detect lines
m.predict()

#%% Step 6 - Fit for a line list
m.fit(ll_out='./data/example_spec_linelist.xlsx', 
      fit_out='./data/example_spec_voigt_fitted.npy')
# %%
