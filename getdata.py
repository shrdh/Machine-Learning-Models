import numpy as np

def get_data(features, labels, wavelengths, minWavelength=630, maxWavelength=670):
    """
    Preprocesses the input data by selecting a subset of wavelengths and subtracting the mean.

    Args:
        features (list): A list of input feature arrays, each of shape (n_samples, n_features).
        labels (numpy.ndarray): An array of shape (n_samples,) containing the target variable.
        wavelengths (numpy.ndarray): An array of shape (n_features,) containing the wavelengths for each feature.
        minWavelength (float): The minimum wavelength to include in the subset. Default is 630 nm.
        maxWavelength (float or str): The maximum wavelength to include in the subset. Default is 670 nm.

    Returns:
        tuple: A tuple containing the preprocessed features, labels, wavelengths, and mean of the features.
    """

     # Select a subset of wavelengths
    wavinds = np.where((wavelengths < float(maxWavelength)) & (wavelengths >float( minWavelength)))[0]
    wavinds = np.where((wavelengths < maxWavelength) & (wavelengths > minWavelength))[0]
    features = np.stack(features, axis=0)[:, wavinds]
    #features = np.stack(features, axis=0)[wavinds]
    wavelengths = wavelengths[wavinds]

    # Subtract the mean
    mu = np.mean(features, axis=0)
    features = features - mu

    return features, labels, wavelengths, mu