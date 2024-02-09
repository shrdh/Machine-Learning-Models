import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import numpy as np

mpl.rcParams.update({'font.size': 20, 'axes.labelsize': 22, 'legend.fontsize': 20, 'legend.loc': 'best', 'axes.titlesize': 24})

def plot_modes(reg, method, wavelengths, savedir, saveplots, maxModes):
    """
    Plots the regression modes for the given regression object and method.

    Args:
        reg (Regression): The regression object to plot modes for.
        method (str): The regression method to use. Can be 'PCA' or 'LDA'.
        wavelengths (numpy.ndarray): An array of shape (n_features,) containing the wavelengths for each feature.
        savedir (str): The directory to save the plot to.
        saveplots (bool): Whether to save the plot to a file or show it in a window.
        maxModes (int): The maximum number of modes to plot.
    """
    if method == 'PCA':
       # plt.plot(wavelengths, reg.modes[:maxModes, :].T)
        plt.plot(wavelengths, reg.modes[:maxModes, :].T, label=f'Mode {3}')
        
    elif method == 'LDA':
        #wavelengths = wavelengths[:3]
        plt.plot(wavelengths, reg.modes)
     
        
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Intensity')
    plt.title(f'{method} Modes')
    if saveplots:
        plt.savefig(f'{savedir}{method}Modes.pdf')
    else:
        plt.show()

def plot_time_series(reg, method, savedir, saveplots):
    """
    Plots the time series data for the given regression object and method.

 """
    plt.plot(reg.timeSeries)
    plt.xlabel('Time')
    plt.title(f'{method} Time Series')
    if saveplots:
        plt.savefig(f'{savedir}{method}TimeSeries.pdf')
    else:   
        plt.show()

def plot_regressions_in_sample(reg, method, savedir, saveplots):
    """
    Plots the in-sample regression results for the given regression object and method.

    """
    plt.plot(reg.labels[reg.fitInds], 'k') 
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    y_pred = reg.predict(reg.features[reg.fitInds,:])
    plt.plot(y_pred) 
    # Compute the RMS error
    err = np.std(y_pred - reg.labels[reg.fitInds])
    plt.legend(['Truth'] + [f' {method}, RMS={err:.2f}'])
    plt.title(f'{method} Regressions (In Sample)')
    if saveplots:
        plt.savefig(f'{savedir} {method}Regressions-in_sample.pdf')
    else:
        plt.show()

def plot_regressions_out_of_sample(reg, method, savedir, saveplots):
    """
    Plots the out-of-sample regression results for the given regression object and method.

    """
    plt.plot(reg.labels[reg.testInds], 'k') 
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    y_pred = reg.predict(reg.features[reg.testInds,:])
    plt.plot(y_pred) 
   # Compute the RMS error
    err = np.std(y_pred - reg.labels[reg.testInds])
    plt.legend(['Truth'] + [f' {method}, RMS={err:.2f}'])
    plt.title(f'{method} Regressions (Out Sample)')
    if saveplots:
        plt.savefig(f'{savedir} {method}Regressions-out_sample.pdf')
    else:
     plt.show()

