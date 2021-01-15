# commonly used functions

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import subprocess
import os
import shutil

def read_tddft_spectrum_file(path):
    return np.loadtxt(path).T


def get_Data(compound_list, directory='XES'):
    Data = []
    iterator = 1
    for compound in compound_list:
        spectrum = read_tddft_spectrum_file(f'ProcessedData/{compound}/{directory}/{compound}.processedspectrum')
        transitions = read_tddft_spectrum_file(f'ProcessedData/{compound}/{directory}/{compound}.dat')
         
        temp_dict = {'CID': compound, 'Spectra': spectrum, 'Transitions': np.flip(transitions, axis=1)}
        Data.append(temp_dict)
        print(f'{iterator}\r', end="")
        iterator += 1

    return Data


def get_Property(Dict_list, myproperty):
    temp = []
    for ele in Dict_list:
        temp.append(ele[myproperty])
    return temp


def plot_spectrum_and_trans(spectrum, transitions, compound):
    
    x, y = spectrum
    xs, ys = transitions

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y, 'k-', label=compound)
    
    markerline, stemlines, baseline = ax.stem(xs, ys / max(ys), linefmt='r-', markerfmt='ro')
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, 'linewidth', 1)
    plt.setp(markerline, 'markersize', 4)

    plt.title(f'CID {compound}', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=16)
    plt.tick_params(labelsize=14)
    
    ax.tick_params(direction='in', width=2, length=8)
    
    plt.show()


def plot_spectrum(spectrum, compound):
    x, y = spectrum

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y, 'k-', linewidth=1, label=compound)

    plt.title(f'CID {compound}', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=16)
    plt.tick_params(labelsize=14)

    ax.tick_params(direction='in', width=2, length=8)
    
    plt.show()