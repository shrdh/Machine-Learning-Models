# PCA, LDA Analysis on HeatRamp Data

This project performs a Principal Component Analysis (PCA)and Linear Discriminant Analysis on Heat Ramp Data and visualizes the results.

## Dependencies

The project requires the following Python libraries:

- numpy
- matplotlib
- scipy
- os

Additionally, it uses the modules :

- getdata - preprocess the data loaded from the MATLAB file. It takes the intensities, temperatures, and wavelengths data, along with a minimum and maximum wavelength. It returns the preprocessed intensities, temperatures, and wavelengths, along with a variable mu.
- Regression - contains the Regression class used in the script. The Regression class is created with the preprocessed intensities and temperatures. The fit method of this class is then used to perform PCA, LDA on the data. The fit method returns the time series, modes, and fit indices. 
- plotter - contains several functions for plotting the results of the PCA/LDA. The functions plot_modes, plot_regressions_out_of_sample, and plot_regressions_in_sample are used in the script.
- DimensionalReductionMethods- contains various methods for dimensionality reduction, such as PCA, LDA, SLDA etc. These methods used to reduce the dimensionality of the data, which can be useful for visualizing high-dimensional data or for improving the efficiency of machine learning algorithms.

## How to Run

1. Clone this repository.
2. Ensure that you have the necessary Python libraries installed.
3. Run `demo.py` to perform the PCA and generate the plots.

## What does `demo.py` do?

- Loads HeatRamp data from a MATLAB file.
- Preprocesses the data to filter based on a wavelength range.
- Performs PCA on the preprocessed data.
- Plots the PCA modes, out-of-sample regression results, and in-sample regression results.

## Output

The script generates several plots related to the PCA and saves them in the `experiments/PCA/` directory.

## Note

For plotting of LDA modes there is a commented out version in DimensionalReductionMethods file.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

