"""Module contains commonly used functions for phosphorus data analysis."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.patches as mpatches
import subprocess
import os
import shutil
from pathlib import Path
from collections import defaultdict
import mplcursors as mpl
import webbrowser
from sklearn.neighbors import KNeighborsRegressor

# GLOBAL VARIABLES
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
    'None': 0
}

COORDCODES = {
    'phosphorane': 4,
    'trialkyl_phosphine': 3,
    'phosphinite': 3,
    'phosphine_oxide': 4,
    'phosphinate': 4,
    'phosphonite': 3,
    'phosphenic_acid': 4,
    'phosphonate': 4,
    'phosphite_ester': 3,
    'phosphonic_acid': 4,
    'phosphate': 4,
}

PHOSPHORANECODES = {
    'phosphorane': 1,
    'nitrogen_phosphorane': 2,
    'phosphine_oxide': 3,
    'sulfur_phosphorane': 4,
}

SULFURCODES = {
    'phosphate': 1,
    'phosphorothioate': 2,
    'dithiophosphate': 3
}


def colorbynumber(n, a=1, colormap=plt.cm.tab20):
    """Colormap using tab20."""
    return np.array(colormap((n - 1) % 20 / 19)) - [0, 0, 0, 1 - a]


def read_tddft_spectrum_file(path):
    """Read spectrum file."""
    return np.loadtxt(path).T


def exclude_None_class(ele):
    """Filter unclassified elements."""
    c = ele['Class']
    if c == 'None':
        return False
    else:
        return True


def Merge(dict1, dict2):
    """Merge two dictionaries."""
    res = {**dict1, **dict2}
    return res


def read_esp_file(compound):
    """Esp file parser."""
    filename = f'{compound}.esp'
    if os.path.exists(f'ProcessedData/{filename}'):
        with open(f'ProcessedData/{filename}') as f:
            chargelist = f.read().splitlines()
        Pcharge = [ele for ele in chargelist if 'P' in ele]
        return float(Pcharge[0].split(' ')[-1])
    else:
        # no esp file available
        return -1


def get_Data(cidlistdir, exclude_Nonetype_normalization=True):
    """Return list of dictionaries, one for each compound."""
    Data = []
    counter = 0
    for classlist in cidlistdir.glob('*.list'):
        with open(classlist) as f:
            compound_list = [int(cid) for cid in f.read().splitlines()]
        for compound in compound_list:
            XANES_spectrum = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_xanes.processedspectrum')
            XANES_trans = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_xanes.dat')
            XES_spectrum = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_xes.processedspectrum')
            XES_transitions = read_tddft_spectrum_file(
                f'ProcessedData/{compound}_xes.dat')
            charge = read_esp_file(compound)

            temp_dict = {'CID': compound,
                         'XANES_Spectra': XANES_spectrum,
                         'XANES_Transitions': np.flip(XANES_trans, axis=1),
                         'XES_Spectra': XES_spectrum,
                         'XES_Transitions': np.flip(XES_transitions, axis=1),
                         'Class': classlist.stem,
                         'Charge': charge}

            Data.append(temp_dict)
            print(f'{counter}\r', end="")
            counter += 1

    XANES_scalefactor = np.max([compound['XANES_Spectra'][1]
                               for compound in Data
                               if compound['Class'] is not None])
    XES_scalefactor = np.max([compound['XES_Spectra'][1] for compound in Data
                             if compound['Class'] is not None])
    for c in Data:
        c['XANES_Normalized'] = c['XANES_Spectra'][1] / XANES_scalefactor
        c['XES_Normalized'] = c['XES_Spectra'][1] / XES_scalefactor
    return Data


def get_Property(Dict_list, myproperty, applyfilter=None):
    """Return list just of a speciified property."""
    temp = []
    if applyfilter is not None:
        Dict_list = filter(applyfilter, Dict_list)
    for ele in Dict_list:
        temp.append(ele[myproperty])
    return temp


def enumerate_unique(a):
    """Get uniques values and enumerate."""
    d = {x: i for i, x in enumerate(set(a))}
    return [d[item] for item in a]


def plot_spectrum_and_trans(spectrum, transitions, compound,
                            verbose=True, label=None):
    """Plot spectrum with transition lines."""
    x, y = spectrum
    xs, ys = transitions

    # rescaling transitions
    ys = ys / np.max(ys)
    ys = ys * np.max(y)

    fig, ax = plt.subplots(figsize=(8, 6))

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
    """Just plot spectrum."""
    x, y = spectrum

    fig, ax = plt.subplots(figsize=(8, 6))

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
    """Energy snip of transitions."""
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
        y = y / np.max(spectra)

    elif mode == 'xanes':
        x = x + 50
        whiteline = energy[np.argmax(spectra)]
        maxE = whiteline + 20
        bool_arr = x < maxE
        x = x[bool_arr]
        y = y[bool_arr]

    return x, y


def hist(bins, labels, verbose=False, xlabel=None, colormap=plt.cm.tab20):
    """Make a histogram."""
    if verbose:
        fig, ax = plt.subplots(figsize=(len(labels) * 1.3, 6))
    else:
        if len(labels) > 3:
            fig, ax = plt.subplots(figsize=(len(labels) - 1, 1))
        else:
            fig, ax = plt.subplots(figsize=(4, 1))

    width = 0.9
    x_pos = np.array([i + 1 for i, _ in enumerate(labels)])
    if type(labels[0]) == str:
        if xlabel == 'Class':
            classnums = np.array([CLASSCODES[clsname] for clsname in labels])
            colors = list(colorbynumber(classnums, colormap=colormap))
        else:
            colors = list(colorbynumber(x_pos, colormap=colormap))
    else:
        colors = list(colorbynumber(labels, colormap=colormap))  # xpos and labels in()?
    bars = ax.bar(x_pos, bins, width=width, color=colors)

    if verbose:
        max_h = 0
        for i, bar in enumerate(bars.patches):
            ax.annotate(f'({bar.get_height()})',
                        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        ha='center', va='bottom',
                        size=22, xytext=(0, 8),
                        textcoords='offset points')
            if max_h < bar.get_height():
                max_h = bar.get_height()
        plt.ylim(1, max_h * 1.2)

    plt.yticks(fontsize=22)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    plt.xticks(x_pos, labels, fontsize=24)

    if verbose:
        ax.set_ylabel('Counts', fontsize=24)
    else:
        ax.axes.yaxis.set_visible(False)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=24)
    ax.tick_params(axis='y', direction='in', width=3, length=9)

    if len(labels) < 10:
        if not verbose:
            angle = 90
        else:
            angle = 0
        size = 20
    else:
        angle = 90
        size = 20

    ax.tick_params(axis='x', direction='out', width=3, length=9,
                   labelrotation=angle)
    plt.setp(ax.get_xticklabels(), Fontsize=size)
    plt.setp(ax.get_yticklabels(), Fontsize=20)

    plt.show()


def checkmode(mode):
    """Check if mode is XES or XANES."""
    if mode not in ('XANES', 'XES'):
        raise ValueError('mode must be XANES or XES')


def plot_spaghetti(plot, compoundmap, colorcodemap=None, binmap=None,
                   mode='XANES', energyrange=None, hiddenalpha=0.01,
                   hiddencids=None, colormap=plt.cm.tab20, coloralpha=1,
                   linewidth=1, scale=False, average_bins=False, **kwargs):
    """Make a spaghetti line plot."""
    checkmode(mode)

    if binmap is None:
        binmap = defaultdict(lambda: 1)

    if colorcodemap is None:
        colorcodemap = defaultdict(lambda: 1)

    if len(colorcodemap) > 0 or len(binmap) > 0:
        if len(colorcodemap) >= len(binmap):
            select_map = colorcodemap
        else:
            select_map = binmap
        X_data = [c for c in list(compoundmap.values())
                  if c['CID']in select_map.keys()]
    else:
        X_data = [c for c in list(compoundmap.values())]

    fig, ax = plot
    title = f"{mode} Spectra"

    if hiddencids is None:
        hiddencids = []
    else:
        hiddencids = hiddencids.copy()
    if 'CID' in kwargs:
        if type(kwargs['CID']) == int:
            hiddencids += [c['CID'] for c in X_data
                           if c['CID'] != kwargs['CID']]
        else:
            hiddencids += [c['CID'] for c in X_data
                           if c['CID'] not in kwargs['CID']]
    elif 'Class' in kwargs:
        hiddencids += [c['CID'] for c in X_data
                       if c['Class'] not in kwargs['Class']]

    if energyrange is not None:
        plt.xlim(energyrange)

    lines = []
    if average_bins:
        bin_averages = {bin_num: [] for bin_num in
                        np.unique(list(binmap.values()))}
        bin_colors = {bin_num: 0 for bin_num in
                      np.unique(list(binmap.values()))}

    for compound in X_data:
        cid = compound['CID']
        bin_num = binmap[cid]
        plot = True
        if cid in hiddencids:
            if hiddenalpha == 0:
                plot = False
            else:
                color = (0, 0, 0, hiddenalpha)
        else:
            color = list(colorbynumber((colorcodemap[cid])))
            color[3] = coloralpha
        if plot:
            y = compound[f'{mode}_Normalized']
            if scale:
                y = 0.9 * y / np.max(y)
            if average_bins:
                bin_averages[bin_num].append(y)
                if bin_colors[bin_num] == 0:
                    bin_colors[bin_num] = color
                lines.append(plt.plot(compound[f'{mode}_Spectra'][0],
                                      y + bin_num, '-',
                                      color='gray', alpha=0.03,
                                      linewidth=linewidth,
                                      label=(str(cid) + ', ' +
                                             str(compound['Class'])))[0])
            else:
                lines.append(plt.plot(compound[f'{mode}_Spectra'][0],
                                      y + bin_num, '-',
                                      color=color, linewidth=linewidth,
                                      label=(str(cid) + ', ' +
                                             str(compound['Class'])))[0])

    if average_bins:
        for bin_num, spectra in bin_averages.items():
            color = bin_colors[bin_num]
            spectra = np.array(spectra)
            averaged_spectra = np.average(spectra, axis=0)
            lines.append(plt.plot(X_data[0][f'{mode}_Spectra'][0],
                                  averaged_spectra + bin_num, '-',
                                  color=color, linewidth=linewidth,
                                  label=(str(bin_num))[0]))

    num_bins = max(np.unique(list(binmap.values())))
    plt.plot([2140], num_bins, 'w.', markersize=0.1)

    # if bool(kwargs):
    #     title = title + f': {[v for k, v in kwargs.items()][0][0]}'
    if mode == 'XES':
        title = 'VtC-' + title
    if average_bins:
        title = title + "\naveraged by bin"
    plt.title(title, fontsize=30)

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


def train_KNN(x_train, y_train, n_neighbors):
    """Return trained KNN model."""
    clf = KNeighborsRegressor(n_neighbors, weights='distance')
    clf.fit(x_train, y_train)

    x_min, x_max = np.min(x_train[:, 0]), np.max(x_train[:, 0])
    y_min, y_max = np.min(x_train[:, 1]), np.max(x_train[:, 1])

    buffer = np.abs(x_max - x_min) * 0.05
    h = (np.abs(x_max - x_min) + 2 * buffer) / 100

    xx, yy = np.meshgrid(np.arange(x_min - buffer, x_max + buffer, h),
                         np.arange(y_min - buffer, y_max + buffer, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    return clf, xx, yy, Z


def plot_dim_red(plot, X_data, redspacemap, colorcodemap=None, mode='VtC-XES',
                 method='t-SNE', hiddencids=None, hiddenalpha=0.01,
                 colormap=plt.cm.tab20, fontsize=16, heatmap=False,
                 size=5, verbose=False, colorbar=False, label=None,
                 scalecolor=True, cbarlim=None, edgecolors=None, 
                 show_legend=True, **kwargs):
    """Plot reduced dimension figure."""
    fig, ax = plot

    if hiddencids is None:
        hiddencids = []
    else:
        hiddencids = hiddencids.copy()

    if 'Class' in kwargs:
        hiddencids += [c['CID'] for c in X_data
                       if c['Class'] not in kwargs['Class']]
    elif 'Type' in kwargs:
        if type(kwargs['Type']) == int:
            hiddencids += [c['CID'] for c in X_data
                           if c['Type'] != kwargs['Type']]
        else:
            hiddencids += [c['CID'] for c in X_data
                           if c['Type'] not in kwargs['Type']]

    if scalecolor:
        colorbynumber = lambda n, a=1: np.array(colormap((n - 1) % 20 / 19))-[0, 0, 0, 1-a]
    else:
        colorbynumber = lambda n, a=1: np.array(colormap(n))-[0, 0, 0, 1-a]


    if colorcodemap is not None:
        colors = [colorbynumber(colorcodemap[compound['CID']])
                  if compound['CID'] not in hiddencids
                  else (0, 0, 0, hiddenalpha) for compound in X_data]
    else:
         colors = ['k' if compound['CID'] not in hiddencids
                   else (0, 0, 0, hiddenalpha)
                   for compound in X_data]

    points = np.array([redspacemap[compound['CID']] for compound in X_data])
    dots = ax.scatter(*zip(*points), s=size, c=colors, edgecolors=edgecolors)

    if hiddenalpha == 0:
        pts = np.array([redspacemap[compound['CID']] for compound in X_data
                        if compound['CID'] not in hiddencids])
        ymn, ymx = min(pts[:,1]), max(pts[:,1])
        xmn, xmx = min(pts[:,0]), max(pts[:,0])
        plt.ylim(ymn - (ymx-ymn)*0.05, ymx + (ymx-ymn)*0.05)
        plt.xlim(xmn - (xmx-xmn)*0.05, xmx + (xmx-xmn)*0.05)

    ax.set_xlabel(f"{method} [0]", fontsize=fontsize + 6)
    ax.set_ylabel(f"{method} [1]", fontsize=fontsize + 6)
    ax.tick_params(direction='in', width=2, length=8)

    if mode == 'XES':
        mode = 'VtC-' + mode

    if 'CID' in kwargs:
        for i, cid in enumerate(kwargs['CID']):
            x, y = redspacemap[cid]
            ax.scatter(x, y, s=size+30, facecolors='none', edgecolors='k', linewidth=2)
            if label is not None:
                text = label[i]
            else:
                text = i +  1 
            ax.annotate(text, (x, y),
                        ha='center', va='bottom',
                        size=fontsize, xytext=(0, 4),
                        textcoords='offset points')

    if show_legend: 
        if mode == 'VtC-XES':
            loc = 2
        else:
            loc = 1                      
        legend = ax.legend([f'{mode}:\n{method}'], handlelength=0, handletextpad=0,
                           fancybox=True, fontsize=fontsize, loc=loc)

        for item in legend.legendHandles:
            item.set_visible(False)

    if not verbose:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    if heatmap:
        x_train = np.array([redspacemap[compound['CID']] for compound in X_data
                            if compound['CID'] not in hiddencids])
        y_train = [colorcodemap[compound['CID']] for compound in X_data
                   if compound['CID'] not in hiddencids]
        clf, xx, yy, Z = train_KNN(x_train, y_train, 5)
        ax.pcolormesh(xx, yy, Z, cmap=colormap, alpha=0.1)
        plt.ylim(min(np.unique(yy)), max(np.unique(yy)))
        plt.xlim(min(np.unique(xx)), max(np.unique(xx)))

    if colorbar:
        if cbarlim is not None:
            vmin, vmax = cbarlim
        else:
            vmin, vmax = (0, 1)
        vrange = vmax - vmin
        norm = Normalize(vmin=vmin, vmax=vmax)
        cax = fig.add_axes([0.125, 0.075, 0.775, 0.05])
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        cbar = plt.colorbar(sm, cax=cax,
                            ticks=np.round(np.linspace(vmin, vmax, 5), 3),
                            orientation='horizontal',
                            boundaries=np.linspace(vmin - vrange * 0.08,
                                                   vmax + vrange * 0.08,
                                                   500))
        cbar.set_label(label, fontsize=fontsize- 2)
        cbar.ax.tick_params(labelsize=fontsize- 2, width=2, length=8)

    return dots


def add_point_label(pickable, X_data, otherdatamap=None):
    """"""
    def onselect(sel):
        compound = X_data[sel.target.index]
        cid = compound['CID']
        annotation = str(cid) + ',' + str(compound['Type']) + ','\
                     + str(compound['Class'])
        if otherdatamap is not None:
            annotation += '\n'+str(otherdatamap[cid])
        sel.annotation.set_text(annotation)
    mpl.cursor(pickable, highlight=True).connect("add", onselect)


def add_line_label(pickable, X_data, otherdatamap=None):
    """
    Add line label.

    NOTE: This doesn't filter out hidden lines so unless you click exactly
    on a line you might get the label
    for a hidden line next to the line you actually want
    """
    def onselect(sel):
        cid = int(sel.artist.get_label().split(',')[0])
        compound = next(c for c in X_data if c['CID']==cid)
        annotation = str(cid)+','+str(compound['Type'])+','+str(compound['Class'])
        if otherdatamap is not None:
            annotation += '\n' + str(otherdatamap[cid])
        sel.annotation.set_text(annotation)
    mpl.cursor(pickable, highlight=True).connect("add", onselect)


def add_point_pubchem_link(pickable, X_data):
    """Add point label."""
    def onselect(sel):
        webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?" + 
                        f"cid={X_data[sel.target.index]['CID']}&t=l")
        sel.annotation.set_text("")
    mpl.cursor(pickable).connect("add", onselect)


def add_line_pubchem_link(pickable, X_data):
    """Add pubchem structure link."""
    def onselect(sel):
        cid = int(sel.artist.get_label().split(',')[0])
        compound = next(c for c in X_data if c['CID']==cid)
        webbrowser.open(f"https://pubchem.ncbi.nlm.nih.gov/image/imgsrv.fcgi?cid={compound['CID']}&t=l")
        sel.annotation.set_text("")
    mpl.cursor(pickable).connect("add", onselect)


def get_correlation(cids, cluster_label1, cluster_label2, clustermap1, clustermap2):
    """Get correlation between two clustermap labels."""
    cluster1cids = set([cid for cid in cids if clustermap1[cid] == cluster_label1])
    cluster2cids = set([cid for cid in cids if clustermap2[cid] == cluster_label2])
    fraction_same = len(cluster1cids.intersection(cluster2cids))
    n_biggest = max(len(cluster1cids), len(cluster2cids))
    return fraction_same / n_biggest


def make_legend(plot, labels, pattern, codes=CLASSCODES, include_structures=True):
    """Make legend."""
    fig, ax = plot
    
    if include_structures:
        structures = [plt.imread(f'../Figures/{clsname}.png') for clsname in labels]
    else:
        structures = labels

    N = len(structures)
    textx = 0.08
    if N == 5:
        w, h = 1/N, 0.8/N
        x = lambda i: 1. - w*1.5 + 0.1*(i%2)
        y = lambda i: .9 - h*(i + 1)
        texty = 0.2
    elif N == 4:
        w, h = 1/N, 0.7/N
        x = lambda i: 1. - w*.9 - 0.05*(i%3)
        y = lambda i: .80 - h*(i + 1)*0.95
        texty = 0.2
    elif N == 3:
        w, h = 1/N, 0.9/N
        x = lambda i: 1.05 - w*1.35 + 0.08*(i%2)
        y = lambda i: .90 - h*(i + 1)*0.9
        texty = 0.0
    else:
        w, h = 1/N, 0.4
        x = lambda i: 1. - w*1.1 - 0.1*(i%2)
        y = lambda i: .90 - h*(i + 1)
        textx, texty = 0.06, 0.1
    fontsize = 25
        
    classnums = np.array([codes[clsname] for clsname in labels])
    colors = list(colorbynumber(classnums, colormap=plt.cm.tab20))

    if include_structures:
        patches = [mpatches.Patch(color=colors[i], label=("\n" + label.replace('_',' ') + "\n")) 
                   for i, label in enumerate(labels)]
    else:
        patches = [mpatches.Patch(color=colors[i], label=(label.replace('_',' '))) 
                   for i, label in enumerate(labels)]
    if include_structures:
        loc = 'center'
    else:
        loc = 7
    legend = ax.legend(handles=patches, loc=loc, fontsize=fontsize, framealpha=0, frameon=False)
    if pattern is not None:
        plt.arrow(x=0.05, y=0, dx=0, dy=0.9, width=0.02, facecolor='k', edgecolor='w') 
        plt.annotate(pattern, (textx,texty), fontsize=fontsize, rotation=90)

    if include_structures:
        for i in range(N):
            subax = fig.add_axes([x(i), y(i), w, h], anchor='NE')
            subax.imshow(structures[i])
            subax.axis('off')

    ax.axis('off')
    return fig


def get_scaled_chargemap(X_subset, hiddencids=[], **kwargs):
    """"Get scaled charge map."""
    for compound in X_subset:
        add = 0
        for k in kwargs.keys():
            if k == 'Charge':
                min_c, max_c = kwargs[k]
                if compound[k] <= max_c and compound[k] >= min_c:
                    add += 1
            if compound[k] in kwargs[k]:
                add += 1
        if add != len(kwargs.keys()):   
            hiddencids += [compound['CID']]

    shown_cidmap = {compound['CID']:compound['Charge'] for compound in X_subset if compound['CID'] not in hiddencids}

    charge_values = list(shown_cidmap.values())
    min_charge = np.min(charge_values)
    max_charge = np.max(charge_values)

    scaled_chargemap = {cid:(charge - min_charge) / (max_charge - min_charge)
                        for cid,charge in shown_cidmap.items()}
    return scaled_chargemap, min_charge, max_charge


def make_charge_hist(chargemap_coord, label='Charge on P', bins=50, colorcodemap=None):
    """Make histogram of charges."""
    fig, ax = plt.subplots(figsize=(8,6))
    if colorcodemap is None:
        charges = [v for k,v in chargemap_coord.items() if v != -1]
        histogram = plt.hist(charges, bins=bins, color=plt.cm.tab20(0.3), edgecolor='w')
    else: 
        unique_codes = np.unique(list(colorcodemap.values()))
        colors = [colorbynumber(code) for code in unique_codes]
        charges = [[v for k,v in chargemap_coord.items() if v != -1 and colorcodemap[k] == code]
                   for code in unique_codes]
        histogram = plt.hist(charges, bins=bins, color=colors, edgecolor='w', histtype='barstacked')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)
    plt.xlabel(label, fontsize=26)
    ax.tick_params(direction='out', width=3, length=9)