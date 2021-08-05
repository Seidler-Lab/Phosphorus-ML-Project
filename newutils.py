"""This module contains commonly used functions for phosphorus data analysis."""

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


# GLOBAL VARIABLES
TYPECODES = {
    'phosphorane': 1,
    'phosphane': 2,
    'trialkyl_phosphine': 2,
    'phosphaalkene': 2,
    'phosphinite': 3,
    'phosphine_oxide': 3,
    'phosphinate': 4,
    'phosphonite': 4,
    'phosphenic_acid': 4,
    'phosphonate': 5,
    'phosphite_ester': 5,
    'hypophosphite': 5,
    'phosphonic_acid': 5,
    'phosphate': 6,
    'dithiophosphate': 7,
    'phosphorothioate': 8,
    'methylphosphonothioate': 9,
    'None': 10
}

CLASSCODES = {
    'phosphorane': 1,
    'phosphane': 2,
    'trialkyl_phosphine': 3,
    'phosphaalkene': 4,
    'phosphinite': 5,
    'phosphine_oxide': 6,
    'phosphinate': 7,
    'phosphonite': 8,
    'phosphenic_acid': 9,
    'phosphonate': 10,
    'phosphite_ester': 11,
    'hypophosphite': 12,
    'phosphonic_acid': 13,
    'phosphate': 14,
    'dithiophosphate': 15,
    'phosphorothioate': 16,
    'methylphosphonothioate': 17,
    'None': 18
}

Colors1 = ['#440154', '#3b66b3', '#037369', '#54bf58', '#DB6400', '#e4ce0c']
# CMAP1 = Listedcolorcodemap(Colors1)
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

# CMAP2 = Listedcolorcodemap(Colors2)
CMAP2 = plt.cm.Paired

def read_tddft_spectrum_file(path):
    return np.loadtxt(path).T

def exclude_None_class(ele):
    c = ele['Class']
    if c == 'None':
        return False
    else:
        return True

def get_Data(cidlistdir, exclude_Nonetype_normalization=True):
    Data = []
    counter = 0
    for typelist in cidlistdir.glob('*.list'):
        with open(typelist) as f:
            compound_list = [int(cid) for cid in f.read().splitlines()]
        for compound in compound_list:
            XANES_spectrum = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_xanes.processedspectrum')
            XANES_transitions = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_xanes.dat')
            XES_spectrum = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_xes.processedspectrum')
            XES_transitions = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_xes.dat')

            temp_dict = {'CID': compound,
                         'XANES_Spectra': XANES_spectrum,
                         'XANES_Transitions': np.flip(XANES_transitions, axis=1),
                         'XES_Spectra': XES_spectrum,
                         'XES_Transitions': np.flip(XES_transitions, axis=1),
                         'Class': typelist.stem,
                         'Type': TYPECODES[typelist.stem]}
            
            Data.append(temp_dict)
            print(f'{counter}\r', end="")
            counter += 1
            
    XANES_scalefactor = np.max([compound['XANES_Spectra'][1] for compound in Data if compound['Class']!=None])
    XES_scalefactor = np.max([compound['XES_Spectra'][1] for compound in Data if compound['Class']!=None])
    for compound in Data:
        compound['XANES_Normalized'] = compound['XANES_Spectra'][1]/XANES_scalefactor
        compound['XES_Normalized'] = compound['XES_Spectra'][1]/XES_scalefactor
    return Data

def get_Property(Dict_list, myproperty, applyfilter=None):
    temp = []
    if applyfilter is not None:
        Dict_list = filter(applyfilter, Dict_list)
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
        Colors = cmap(x_pos/(n-1))
    else:
        cmap = CMAP2
        Colors = cmap(x_pos)

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
        if verbose: plt.ylim(1,max_h + 35)
    else:
        ax.tick_params(axis='x',direction='out', width=3, length=9, labelrotation=90)
        size=18
        if verbose: plt.ylim(1,max_h + 30)

    plt.setp(ax.get_xticklabels(), Fontsize=size)
    plt.setp(ax.get_yticklabels(), Fontsize=20)

    plt.show()


def checkmode(mode):
    if mode not in ('XANES', 'XES'): raise ValueError('mode must be XANES or XES')

def Rainbow_spaghetti_plot_types_stack(subplot, energy, X, types, CIDS,
    mode='VtC-XES', cmap=CMAP1, MINIMAX=[0,-1]):
    mn, mx = MINIMAX
    
    fig, ax = subplot
    
    lines = []

    if cmap == CMAP1:
        n = max(TYPECODES.values())
    else:
        classes = list(TYPECODES.keys())[:-1]
        x = np.array([i for i, _ in enumerate(classes)])
        n = len(x)

    Colors = cmap(np.arange(n)/(n-1))

    for x, cid, moltype in zip(X, CIDS, types):
        if cmap == CMAP1:
            bin_num = TYPECODES[moltype]
        else:
            bin_num = [i for i, ele in enumerate(classes) if ele == moltype][0]
        lines.append(plt.plot(energy[mn:mx], x[mn:mx] + bin_num, '-', color=Colors[bin_num], alpha=0.1,\
                              label=(str(cid)+','+str(moltype)))[0])
    

def plot_spaghetti(plot, X_data, colorcodemap=None, binmap=None, mode='XANES', energyrange=None, \
                   hiddencids=[], colormap=plt.cm.tab20, coloralpha=1, hiddenalpha=0.01):
    checkmode(mode)
    fig, ax = plot
    
    if energyrange is not None:
        plt.xlim(energyrange)
    
    if binmap is None:
        binmap = defaultdict(lambda: 0)

    if colorcodemap is None:
        colorcodemap = defaultdict(lambda: 0)
    
    lines = []
    for compound in X_data:
        cid = compound['CID']
        bin_num = binmap[cid]
        if cid in hiddencids: color = (0,0,0,hiddenalpha)
        else:
            color = list(colormap(colorcodemap[cid]))
            color[3]=coloralpha
        lines.append(plt.plot(compound[f'{mode}_Spectra'][0], compound[f'{mode}_Normalized']+bin_num, '-',\
                              color=color, \
                              label=(str(cid)+','+str(compound['Type'])))[0])
    plt.title(f"{mode} Spectra", fontsize=30)
    
    plt.xlabel('Energy (eV)', fontsize=26)
    ax.tick_params(direction='in', width=2, length=8)
    plt.xticks(fontsize=20)
    
    ax.xaxis.set_minor_locator(MultipleLocator(2))
    ax.xaxis.set_major_locator(MultipleLocator(10))
    
    ax.tick_params(direction='in', width=2, length=10, which='major')
    ax.tick_params(direction='in', width=1, length=8, which='minor')
    plt.yticks([])
    
    plt.show()
    return lines


def plot_dim_red(plot, X_data, redspacemap, colorcodemap=None, mode='VtC-XES', method='t-SNE', \
                 hiddencids=[], colormap=plt.cm.tab20, fontsize=16):
    fig, ax = plot
    
    if colorcodemap is not None:
        colors = [colorbynumber(colorcodemap[compound['CID']]) if compound['CID'] not in hiddencids else (0,0,0,0.01) \
              for compound in X_data if compound['CID']]
    else:
        colors = 'b'
    points = [redspacemap[compound['CID']] for compound in X_data if compound['CID'] not in hiddencids]
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


def add_point_label(pickable, X_data, otherdatamap=None):
    def onselect(sel):
        compound = X_data[sel.target.index]
        cid = compound['CID']
        annotation = str(cid)+','+str(compound['Type'])+','+str(compound['Class'])
        if otherdatamap is not None:
           annotation += '\n'+str(otherdatamap[cid])
        sel.annotation.set_text(annotation)
    mpl.cursor(pickable, highlight=True).connect("add", onselect)

def add_line_label(pickable, X_data, otherdatamap=None):
    def onselect(sel):
        cid = int(sel.artist.get_label().split(',')[0])
        compound = next(c for c in X_data if c['CID']==cid)
        annotation = str(cid)+','+str(compound['Type'])+','+str(compound['Class'])
        if otherdatamap is not None:
           annotation += '\n'+str(otherdatamap[cid])
        sel.annotation.set_text(annotation)
    mpl.cursor(pickable, highlight=True).connect("add", onselect)
    
def add_point_pubchem_link(pickable, X_data):
    def onselect(sel):
        webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={X_data[sel.target.index]['CID']}&t=l")
        sel.annotation.set_text("")
    mpl.cursor(pickable).connect("add", onselect)
    
def add_line_pubchem_link(pickable, X_data):
    def onselect(sel):
        cid = int(sel.artist.get_label().split(',')[0])
        compound = next(c for c in X_data if c['CID']==cid)
        webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={compound['CID']}&t=l")
        sel.annotation.set_text("")
    mpl.cursor(pickable).connect("add", onselect)
    
def get_correlation(cids, clusters1, clustermap1, clustermap2):
    matchfrac_count = 0
    for cluster in clusters1:
        clustercids = [cid for cid in cids if clustermap1[cid]==cluster]
        matchcount = 0
        if len(clustercids)==1:
            continue
        for i,cid1 in enumerate(clustercids):
            for cid2 in clustercids[i+1:]:
                if clustermap2[cid1]==clustermap2[cid2]:
                    matchcount+=1
        matchfrac_count += matchcount/(2*len(clustercids)*(len(clustercids)-1))
    return matchfrac_count/len(clusters1)