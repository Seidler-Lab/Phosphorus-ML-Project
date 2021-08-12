# Phosphorus Machine Learning Project

This repository contains the analysis of Valence to Core X-ray Emission Spectroscopy (VTC-XES) and X-ray Absoprtion Near-Edge Structure (XANES) spectra belonging to over 1000 organic phosphorus compounds. We implement various unsupervised machine learning techniques and visualization tools in Jupyter notebooks.

#### Authors: Samantha Tetef and Vikram Kashyap

All spectral data is located in:

1. `ProcessedData/`

The `ProcessedData` directory has .dat files, formatted as CID_xes.dat or CID_xanes.dat, along with each data file's respective .processedspectrum file, which are obtained from broadening the dipole transitions in the .dat files.


All metedata belong to individual compounds is located in:

2. `Database/`

The `Database` directory has .jmf files named by the PubChem compound identification number (CID). The database is managed by our package `moldl`. See [github.com/vikramkashyap/moldl](https://github.com/vikramkashyap/moldl) for the `moldl` repository.