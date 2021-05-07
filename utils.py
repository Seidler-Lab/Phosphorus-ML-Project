"""
This module contains commonly used functions for phosphorus data analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import subprocess
import os
import shutil


def read_tddft_spectrum_file(path):
    return np.loadtxt(path).T


def get_Data(compound_list, mode='xes'):
    Data = []
    iterator = 1
    for compound in compound_list:
        spectrum = read_tddft_spectrum_file(
            f'ProcessedData/{compound}_{mode}.processedspectrum')
        transitions = read_tddft_spectrum_file(
            f'ProcessedData/{compound}_{mode}.dat')
         
        temp_dict = {'CID': compound, 'Spectra': spectrum,
                     'Transitions': np.flip(transitions, axis=1)}
        Data.append(temp_dict)
        print(f'{iterator}\r', end="")
        iterator += 1
    return Data


def get_Property(Dict_list, myproperty):
    temp = []
    for ele in Dict_list:
        temp.append(ele[myproperty])
    return temp


def plot_spectrum_and_trans(spectrum, transitions, compound,
                            verbose=True, label=None):
    
    x, y = spectrum
    xs, ys = transitions

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y, 'k-')
    
    markerline, stemlines, baseline = ax.stem(xs, ys / max(ys), linefmt='r-',
                                              markerfmt='ro',
                                              use_line_collection=True)
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, 'linewidth', 1)
    plt.setp(markerline, 'markersize', 4)

    plt.title(f'CID {compound}', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=18)
    plt.tick_params(labelsize=16)

    if verbose:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.tick_params(direction='in', width=1, length=5, which='minor')
    
    ax.tick_params(direction='in', width=2, length=8)
    if label is not None:
        legend = ax.legend([label], handlelength=0, handletextpad=0,
                           fancybox=True, fontsize=22)
        for item in legend.legendHandles:
            item.set_visible(False)

    plt.show()


def plot_spectrum(spectrum, compound, verbose=True, label=None):
    x, y = spectrum

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y, 'k-', linewidth=2, label=compound)

    plt.title(f'CID {compound}', fontsize=20)
    plt.xlabel('Energy (eV)', fontsize=18)
    plt.ylabel('Intensity (arb. units)', fontsize=18)
    plt.tick_params(labelsize=16)

    if verbose:
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.tick_params(direction='in', width=1, length=5, which='minor')

    ax.tick_params(direction='in', width=2, length=8, which='major')

    if label is not None:
        legend = ax.legend([label], handlelength=0, handletextpad=0,
                           fancybox=True, fontsize=22)
        for item in legend.legendHandles:
            item.set_visible(False)
    
    plt.show()
