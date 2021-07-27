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
TYPECODES = {
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
     'dithiophosphate': 7,
    'None': 8
}

Colors1 = ['#440154', '#3b66b3', '#037369', '#54bf58', '#DB6400', '#e4ce0c']
# CMAP1 = ListedColormap(Colors1)
CMAP1 = plt.cm.viridis

Colors2 = [\

# type 1
'#440154',
# type 2
'#77b6fe',
'#03506F',

# type 3,
'#ffba93',
'#DB6400',

# type 4
'#54bf58',
'#355E3B',

# type 5
'#AA524E',
'#E17C7B',
'#9F5F80',

# type 6
'#fde725']

# CMAP2 = ListedColormap(Colors2)
CMAP2 = plt.cm.tab20

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
                         'Type': TYPECODES[typelist.stem]}
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


def hist(bins, classnames, label='Category', verbose=False):
    x = classnames
    x_pos = np.array([i for i, _ in enumerate(x)])

    n = len(bins)
    if label == 'Type':
        cmap = CMAP1
    else:
        cmap = CMAP2
    
    Colors = cmap(np.arange(n)/(n-1))

    fig, ax = plt.subplots(figsize=(12,6))

    width=0.9
    bars = ax.bar(x_pos, bins, width=width, color=Colors)
    
    if verbose:
        max_h = 0
        for i,bar in enumerate(bars.patches):
            ax.annotate(f'{i+1}\n({bar.get_height()})', 
                        (bar.get_x() + bar.get_width() / 2, 
                        bar.get_height()), ha='center', va='bottom',
                        size=22, xytext=(0, 8),
                        textcoords='offset points')
            if max_h < bar.get_height():
                max_h = bar.get_height()

    plt.yticks(fontsize=22)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    plt.xticks(x_pos, x, fontsize=22)

    ax.set_ylabel('Counts', fontsize=24)
    ax.set_xlabel(label, fontsize=24)

    ax.tick_params(axis='y', direction='in', width=3, length=9)
    
    if label=='Type':
        ax.tick_params(axis='x',direction='out', width=3, length=9, labelrotation=0)
        size=24
        if verbose: plt.ylim(1,max_h + 25)
    else:
        ax.tick_params(axis='x',direction='out', width=3, length=9, labelrotation=90)
        size=18
        if verbose: plt.ylim(1,max_h + 30)

    plt.setp(ax.get_xticklabels(), Fontsize=size)
    plt.setp(ax.get_yticklabels(), Fontsize=20)

    plt.show()


def Rainbow_spaghetti_plot_types_stack(subplot, energy, X, types, CIDS, mode='VtC-XES', MINIMAX=[0,-1]):
    mn, mx = MINIMAX
    
    fig, ax = subplot
    
    lines = []
    n = max(TYPECODES.values())
    Colors = plt.cm.viridis(np.arange(n)/(n-1))
    for x,cid,moltype in zip(X,CIDS,types):
        bin_num = TYPECODES[moltype]
        lines.append(plt.plot(energy[mn:mx], x[mn:mx] + bin_num, '-', color=Colors[bin_num], alpha=0.1,\
                              label=(str(cid)+','+str(moltype)))[0])
        
    plt.title(f"{mode} Spectra", fontsize=30)
    
    plt.xlabel('Energy (eV)', fontsize=26)
    ax.tick_params(direction='in', width=2, length=8)
    plt.xticks(fontsize=20)
    
    if mode == 'XANES' or mode == 'VtC-XES':
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.xaxis.set_major_locator(MultipleLocator(10))
    
    ax.tick_params(direction='in', width=2, length=10, which='major')
    ax.tick_params(direction='in', width=1, length=8, which='minor')
    plt.yticks([])
    
    '''
    mplcursors.cursor(lines, highlight=True, \
                      highlight_kwargs={'color':'pink', 'alpha':1, 'linewidth':3, 'markeredgewidth':0})
                     #.connect("add", lambda sel: webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={CIDS[sel.target.index]}&t=l"))
    '''
    
    plt.show()
    
    return lines


def plot_dim_red(plot, X_red, types, method, fontsize=16, mode='VtC-XES', cmap=CMAP1):

    fig, ax = plot
    
    colors = [TYPECODES[t] for t in types]
    
    dots = plt.scatter(X_red[:, 0], X_red[:, 1], c=colors, cmap=cmap)
    
    plt.xticks(fontsize=fontsize+3)
    plt.yticks(fontsize=fontsize+3)
    
    ax.set_xlabel(f"{method} [0]", fontsize=fontsize+6)
    ax.set_ylabel(f"{method} [1]", fontsize=fontsize+6)
    ax.tick_params(direction='in', width=2, length=8)
    
    legend = ax.legend([f'{mode}:\n{method}'], handlelength=0, handletextpad=0,
                           fancybox=True, fontsize=22)
    for item in legend.legendHandles:
        item.set_visible(False)
    
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
        
    plt.show()
    
    return dots