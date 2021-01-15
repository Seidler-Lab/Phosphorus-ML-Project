import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import plotly.offline as py
# import plotly.tools as tls
# py.init_notebook_mode()

def plotly_show():
    #get fig and convert to plotly
    fig = plt.gcf()
    plotlyfig = tls.mpl_to_plotly(fig, resize=True)
    
    #fix dumb automatic formatting choices
    plotlyfig['layout']['xaxis1']['tickfont']['size']=14
    plotlyfig['layout']['xaxis1']['titlefont']['size']=16
    plotlyfig['layout']['yaxis1']['tickfont']['size']=14
    plotlyfig['layout']['yaxis1']['titlefont']['size']=16
    plotlyfig['layout']['showlegend'] = True
    
    #add a fix to bring back automatic sizing
    plotlyfig['layout']['height'] = None
    plotlyfig['layout']['width'] = None
    plotlyfig['layout']['autosize'] = True
    
    # plot
    py.iplot(plotlyfig)
    
def read_tddft_transitions_file(path):
    return np.loadtxt(path).T


#https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
def Lorentzian(x, xc, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / ((x-xc)**2 + gamma**2)



def spectrum_from_transitions(transitions, lorentz_ev=1, erange=None, numpoints=1000, peaknorm=True):
    x, y = transitions
    if erange is not None:
        good = np.logical_and(x >= erange[0], x <= erange[1])
        x, y = x[good], y[good]
        x_eval = np.linspace(erange[0], erange[1], numpoints)
    else:
        xmin = np.min(x)
        xmax = np.max(x)
        padding = (xmax - xmin) / 2
        x_eval = np.linspace(xmin - padding, xmax + padding, numpoints)
    
    spectrum = np.zeros_like(x_eval)
    for e, a in zip(x, y):
        spectrum += a * Lorentzian(x_eval, e, lorentz_ev/2)
    
    if peaknorm:
        spectrum = spectrum / np.max(spectrum)

    return np.array([x_eval, spectrum])

def index_from_energy(x, e):
    return np.argmin((x - e)**2)

# To be used for XANES broadening
# Adapted from spectrum_from_transitions
def linear_broaden_from_transitions(transitions, lorentz_ev_start=1.2, numpoints=1000, peaknorm=True):
    
    x, y = transitions

    max_amp = np.max(y)
        
    xmin = 2095
    xmax = 2130
    padding = (xmax - xmin)*1 / 5
    x_eval = np.linspace(xmin - padding, xmax + padding, numpoints)
    
    spectrum = np.zeros_like(x_eval)
    
    whiteline_E = x[np.argmax(y[:5])]
    whiteline_index = index_from_energy(x_eval, whiteline_E)
    
    lorentz_ev = np.zeros_like(x_eval)
    lorentz_ev[:whiteline_index] = lorentz_ev_start
    
    index_of_max_broadening = index_from_energy(x_eval, whiteline_E + 15)
    lorentz_ev_max = 8.0
    
    lorentz_ev[index_of_max_broadening:] = lorentz_ev_max
    
    num_indices = index_of_max_broadening  - whiteline_index
    step = (lorentz_ev_max - lorentz_ev_start)/num_indices
    for i in range(num_indices):
        lorentz_ev[i + whiteline_index] = lorentz_ev[i + whiteline_index - 1] + step
    
    for e, a in zip(x, y):

        if not (e > xmax-10):
            spectrum += a * Lorentzian(x_eval, e, lorentz_ev[index_from_energy(x_eval, e)]/2)

    if peaknorm:
        spectrum = spectrum / np.max(spectrum)

    return np.array([x_eval, spectrum])



#Thank you stackoverflow
#https://stackoverflow.com/questions/24143320/gaussian-sum-filter-for-irregular-spaced-points
def gaussian_broaden(spectrum, width_ev=2, numpoints=1000, xmin=None, xmax=None):
    x, y = spectrum
    if xmin is None:
        xmin = np.min(x)
    if xmax is None:
        xmax = np.max(x)
    x_eval = np.linspace(xmin, xmax, numpoints)
    sigma = width_ev/(2*np.sqrt(2*np.log(2)))

    delta_x = x_eval[:, None] - x
    weights = np.exp(-delta_x*delta_x / (2*sigma*sigma)) / (np.sqrt(2*np.pi) * sigma)
    weights /= np.sum(weights, axis=1, keepdims=True)
    y_eval = np.dot(weights, y)

    return np.array([x_eval, y_eval])


def plot_spectrum_and_transitions(transitions, lorentz_ev=1, erange=None, 
                                numpoints=1000, gaussian_ev=None, show=True):
    
    spectrum = spectrum_from_transitions(transitions, lorentz_ev=lorentz_ev, 
                            erange=erange, numpoints=numpoints, peaknorm=False)
    x, y = spectrum
    norm = np.max(y)

    fig, ax = plt.subplots(figsize=(12,8))
    
    ax.plot(x, y / norm, 'k-')

    #rescale so that stem matches spectral height
    rescale = Lorentzian(0, 0, lorentz_ev/2)
    xs, ys = transitions
    if erange is not None:
        good = np.logical_and(xs >= erange[0], xs <= erange[1])
        xs, ys = xs[good], ys[good]
    markerline, stemlines, baseline = ax.stem(xs, ys * rescale / norm, basefmt='k', linefmt='C0-')
    plt.setp(baseline, visible=False)
    plt.setp(stemlines, 'linewidth', 1)
    plt.setp(markerline, 'markersize', 3)
    plt.tick_params(labelsize=14)
    plt.xlabel('Energy (eV)', fontsize=18)

    if show:
        plt.show()

    return fig


def energy_shift(spectra, amt):
    x, y = spectra
    return np.array([x - amt, y])


def y_shift(spectra, amt):
    x, y = spectra
    return np.array([x, y + amt])


def peak_normalize(spectra):
    x, y = spectra
    return np.array([x, y/np.max(y)])


def integral_normalize(spectra):
    x, y = spectra
    return np.array([x, y/np.sum(y)])


def scale_height(spectra, amt):
    x, y = spectra
    return np.array([x, y * amt])


if __name__ == '__main__':
    import argparse
    import os.path

    parser = argparse.ArgumentParser(description=('Script to generate spectra '
        'and/or plots from transitions calculated by NWChem.'))

    parser.add_argument('-f', action='store', dest='filename', type=str,
        help='File of transitions to generate spectrum from.')
    parser.add_argument('-l', action='store', dest='lorentz_width', 
        type=float, default=1.0,
        help='Set the Lorentz width of the transitions (eV).')
    parser.add_argument('-g', action='store', dest='gaussian_broadening', 
        type=float,
        help='Set Gaussian instrumental broadening (eV).')
    parser.add_argument('-emin', action='store', dest='emin', 
        type=float,
        help='Set energy min (eV).')
    parser.add_argument('-emax', action='store', dest='emax', 
        type=float,
        help='Set energy max (eV).')
    parser.add_argument('-eshift', action='store', dest='eshift',
    	type=float, 
        help='Energy shift of spectra (eV) to match experiment.')
    parser.add_argument('-p', action='store_true', default=False, dest='plot', 
        help='Make a png plot of the spectrum and transitions.')
    parser.add_argument('-lb', action='store_true', default=False, dest='linear_broaden', 
        help='Linearly broaden spectra (use for XANES only)')

    result = parser.parse_args()

    transitions = read_tddft_transitions_file(result.filename)
    
    if result.emin is not None and result.emax is not None:
        spectrum = spectrum_from_transitions(transitions, lorentz_ev=result.lorentz_width,
                                peaknorm=False, erange=np.array([result.emin, result.emax]))
    elif result.linear_broaden:
    	spectrum = linear_broaden_from_transitions(transitions)
    else:
        spectrum = spectrum_from_transitions(transitions, lorentz_ev=result.lorentz_width, peaknorm=False) 

    if result.gaussian_broadening is not None:
        spectrum = gaussian_broaden(spectrum, width_ev=result.gaussian_broadening)

    # ENERGY SHIFT
    # For XES spectra: 18.6 eV
    # For XANES spectra: -53.3 eV
    if result.eshift is not None:
        transitions = energy_shift(spectrum, result.eshift)
    
    # integral normalize
    spectrum = integral_normalize(spectrum)

    outputfile = result.filename.split('/')[-1].split('.')[0]
    if outputfile == '':
        outputfile = result.filename.split('\\')[-1].split('.')[0]
    if outputfile == '':
        print('WARNING: Filename could not be parsed. Saving as outspectrum.dat')
        outputfile = 'outspectrum'

    assert not os.path.exists('{}.processedspectrum'.format(outputfile)), 'File with outputname already exists!'
    assert not os.path.exists('{}.png'.format(outputfile)), 'File with outputname already exists!'

    
    np.savetxt('{}.processedspectrum'.format(outputfile), spectrum.T)

    if result.plot:
        
        if result.emin is not None and result.emax is not None:
            fig = plot_spectrum_and_transitions(transitions, lorentz_ev=result.lorentz_width, 
                                erange=np.array([result.emin, result.emax]), gaussian_ev=result.gaussian_broadening)
        else:
            fig = plot_spectrum_and_transitions(transitions, lorentz_ev=result.lorentz_width)


        fig.savefig('{}.png'.format(outputfile))
