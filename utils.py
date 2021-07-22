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
from pathlib import Path
from collections import defaultdict
import mplcursors as mpl
import webbrowser

colorbynumber = lambda n,a=1: np.array(plt.cm.tab20(n%20))-[0,0,0,1-a]

def read_tddft_spectrum_file(path):
    return np.loadtxt(path).T

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
                         'Type': typelist.stem}
            Data.append(temp_dict)
            print(f'{counter}\r', end="")
            counter += 1
            
    scalefactor = np.max([compound['Spectra'][1] for compound in Data])
    for compound in Data:
        compound['Normalized'] = compound['Spectra'][1]/scalefactor
    return Data

def get_Property(Dict_list, myproperty):
    temp = []
    for ele in Dict_list:
        temp.append(ele[myproperty])
    return temp

def enumerate_unique(a):
    d = {x:i for i, x in enumerate(set(a))}
    return [d[item] for item in a]

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
    
    

def plot_spaghetti(plot, X_data, colormap=None, binmap=None, mode='XES', energyrange=None, coloralpha=1):
    fig, ax = plot
    
    if energyrange is not None:
        plt.xlim(energyrange)
    
    if binmap is None:
        binmap = defaultdict(lambda: 0)

    if colormap is None:
        colormap = defaultdict(lambda: 0)
    
    lines = []
    for compound in X_data:
        cid = compound['CID']
        bin_num = binmap[cid]
        lines.append(plt.plot(compound['Spectra'][0], compound['Normalized']+bin_num, '-',\
                              color=colorbynumber(colormap[cid]), alpha=coloralpha, \
                              label=(str(cid)+','+str(compound['Type'])))[0])
        
    plt.title(f"{mode} Spectra", fontsize=30)
    
    plt.xlabel('Energy (eV)', fontsize=26)
    ax.tick_params(direction='in', width=2, length=8)
    plt.xticks(fontsize=20)
    
    if mode == 'XANES' or mode == 'XES':
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


def plot_dim_red(plot, X_data, redspacemap, colormap=None, mode='VtC-XES', method='t-SNE', hiddencids=None, fontsize=16):
    fig, ax = plot
    
    if colormap is not None:
        if hiddencids is not None:
            colors = [colorbynumber(colormap[compound['CID']]) if compound['CID'] not in hiddencids else (0,0,0,0.01) \
                  for compound in X_data if compound['CID']]
        else:
            colors = [colorbynumber(colormap[compound['CID']]) for compound in X_data if compound['CID']]
    else:
        colors = 'b'
    points = [redspacemap[compound['CID']] for compound in X_data]
    dots = ax.scatter(*zip(*points), c=colors)
    
    plt.xticks(fontsize=fontsize+3)
    plt.yticks(fontsize=fontsize+3)
    
    ax.set_xlabel(f"{method} [0]", fontsize=fontsize+6)
    ax.set_ylabel(f"{method} [1]", fontsize=fontsize+6)
    ax.tick_params(direction='in', width=2, length=8)
    
    legend = ax.legend([f'{mode}:\n{method}'], handlelength=0, handletextpad=0,
                           fancybox=True, fontsize=fontsize)
    for item in legend.legendHandles:
        item.set_visible(False)
    
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
        
    plt.show()
    
    return dots

    
def add_label(pickable, X_data, otherdatamap=None):
    def onselect(sel):
        compound = X_data[sel.target.index]
        annotation = str(compound['CID'])+','+str(compound['Type'])
        if otherdatamap is not None:
           annotation += '\n'+str(otherdatamap[compound['CID']])
        sel.annotation.set_text(annotation)
    mpl.cursor(pickable).connect("add", onselect)
    
def add_pubchem_link(pickable, X_data):
    def onselect(sel):
        webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={X_data[sel.target.index]['CID']}&t=l")
        sel.annotation.set_text("")
    mpl.cursor(pickable).connect("add", onselect)