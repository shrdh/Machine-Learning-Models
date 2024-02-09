import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
from getdata import get_data
from Regression import Regression
from plotter import *

# Load data
matlab_data = sio.loadmat('HeatRamp/matlab_data.mat')
allIntensities_, allTemps_, wavelengths_ = [matlab_data[key] for key in ['allIntensities', 'allTemps', 'wavelengths']]
allIntensities, allTemps, wavelengths, mu = get_data(allIntensities_, allTemps_, wavelengths_, minWavelength=630, maxWavelength=670)

# Select regression to use
reg = Regression(allIntensities, allTemps)

timeseries, modes, fitInds = reg.fit(method='PCA', maxModes=5, slices=30, alpha=0.4)


# Plot results
figdir = 'experiments/PCA/'
if not os.path.exists(figdir):
    os.makedirs(figdir) 


#Plot the modes    
plot_modes(reg, 'PCA', wavelengths, figdir, saveplots=False, maxModes=3)

# #Plot the  time series
plot_time_series(reg,'PCA',figdir,saveplots=False) 

# Plot the out-of-sample regression results  
plot_regressions_out_of_sample(reg, 'PCA', figdir, saveplots=False)

# Plot the in-sample regression results
plot_regressions_in_sample(reg, 'PCA', figdir, saveplots=False)
