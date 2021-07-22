"""This module contains commonly used functions for phosphorus data analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import subprocess
import os
import shutil
from pathlib import Path


# GLOBAL VARIABLES
TYPE_DICT = {
    'phosphorane': 1,
    'trialkyl_phosphine': 2,
    'phosphaalkene': 2,
    'phosphinite': 3,
    'phosphine_oxide': 3,
    'phosphinate': 4,
    'phosphonite': 4,
    'phosphonate': 5,
    'phosphite_ester': 5,
    'hypophosphite': 5,
    'phosphate': 6,
    'None': 7
}

def read_tddft_spectrum_file(path):
    return np.loadtxt(path).T

def exclude_None_class(ele):
    c = ele['Class']
    if c == 'None':
        return False
    else:
        return True

def get_Data(cidlistdir, mode='xes'):
    Data = []
    counter = 0
    for typelist in cidlistdir.glob('*.list'):
        with open(typelist) as f:
            compound_list = [int(cid) for cid in f.read().splitlines()]
        for compound in compound_list:
            spectrum = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_{mode}.processedspectrum')
            transitions = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_{mode}.dat')

            temp_dict = {'CID': compound, 'Spectra': spectrum,
                         'Transitions': np.flip(transitions, axis=1),
                         'Class': typelist.stem,
                         'Type': TYPE_DICT[typelist.stem]}
            Data.append(temp_dict)
            print(f'{counter}\r', end="")
            counter += 1
    return Data

def get_Property(Dict_list, myproperty, applyfilter=None):
    temp = []
    if applyfilter is not None:
        Dict_list = filter(applyfilter, Dict_list)
    for ele in Dict_list:
        temp.append(ele[myproperty])
    return temp


def plot_spectrum_and_trans(spectrum, transitions, compound,
                            verbose=True, label=None):
    
    x, y = spectrum
    xs, ys = transitions

    # rescaling transitions
    ys = ys / np.max(ys)
    ys = ys * np.max(y)

    fig, ax = plt.subplots(figsize=(8,6))

    ax.plot(x, y, 'k-')
    
    markerline, stemlines, baseline = ax.stem(xs, ys, linefmt='r-',
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


def esnip(trans, spectra, energy=[], mode='xes', emin=0):
    x, y = trans
    
    if mode == 'xes':
        for i, e in enumerate(x):
            if e >= emin:
                break
        x = x[i:]
        y = y[i:]
    
        if x[-1] < 0:
            return x[:-1] - 19, y[:-1]
        x = x - 19
        y = y/np.max(spectra)

    elif mode == 'xanes':
        x = x + 50
        whiteline = energy[np.argmax(spectra)]
        maxE = whiteline + 20
        bool_arr = x < maxE
        x = x[bool_arr]
        y = y[bool_arr]

    return x, y
