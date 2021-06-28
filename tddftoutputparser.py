import numpy as np


def read_tddft_transitions_file(path):
    return np.loadtxt(path).T


#https://scipython.com/book/chapter-8-scipy/examples/the-voigt-profile/
def Lorentzian(x, xc, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return gamma / np.pi / ((x - xc)**2 + gamma**2)


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
def linear_broaden_from_transitions(transitions, lorentz_ev_start=0.5, numpoints=1000, peaknorm=True):

    x, y = transitions

    xmin = 2090
    xmax = 2125
    padding = 5
    x_eval = np.linspace(xmin - padding, xmax + padding, numpoints)

    spectrum = np.zeros_like(x_eval)

    whiteline_E = x[np.argmax(y[:5])]
    whiteline_index = index_from_energy(x_eval, whiteline_E)

    lorentz_ev = np.zeros_like(x_eval)
    lorentz_ev[:whiteline_index] = lorentz_ev_start

    index_of_max_broadening = index_from_energy(x_eval, whiteline_E + 15)
    lorentz_ev_max = 4.0

    lorentz_ev[index_of_max_broadening:] = lorentz_ev_max

    num_indices = index_of_max_broadening - whiteline_index
    step = (lorentz_ev_max - lorentz_ev_start) / num_indices
    for i in range(num_indices):
        lorentz_ev[i + whiteline_index] = lorentz_ev[i + whiteline_index - 1] + step

    for e, a in zip(x, y):

        if not (e > whiteline_E + 20):
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


def energy_shift(spectra, amt):
    x, y = spectra
    return np.array([x - amt, y])


def y_shift(spectra, amt):
    x, y = spectra
    return np.array([x, y + amt])


def peak_normalize(spectra):
    x, y = spectra
    return np.array([x, y / np.max(y)])


def integral_normalize(spectra):
    x, y = spectra
    return np.array([x, y / np.sum(y)])


def transition_normalize_xanes(transitions, xes_trans):
    x, y = transitions
    xs, ys = xes_trans
    kalpha_trans_sum = 0
    for i in range(len(xs)):
        if xs[i] < 2030:
            kalpha_trans_sum += ys[i]
    return np.array([x, y / kalpha_trans_sum])


def transition_normalize_xes(transitions):
    xs, ys = transitions
    kalpha_trans_sum = 0
    for i in range(len(xs)):
        if xs[i] < 2030:
            kalpha_trans_sum += ys[i]
    return np.array([xs, ys / kalpha_trans_sum])


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
    	default = 0,
        type=float,
        help='Energy shift of spectra (eV) to match experiment.')
    parser.add_argument('-lb', action='store_true', default=False, dest='linear_broaden', 
        help='Linearly broaden spectra (use for XANES only)')
    parser.add_argument('-mode', action='store', default='xes', dest='mode', type=str,
        help='Mode of dat files. Should be "xes" or "xanes". Default = "xes"')

    result = parser.parse_args()
    transitions = read_tddft_transitions_file(result.filename)

    if result.mode == 'xes':
        # xes
        transitions = read_tddft_transitions_file(result.filename)
        transitions = transition_normalize_xes(transitions)
    elif result.mode == 'xanes':
        # xanes
        transitions = read_tddft_transitions_file(result.filename)
        xes_dat_file = result.filename.replace('xanes', 'xes')
        xes_trans = read_tddft_transitions_file(xes_dat_file)
        transitions = transition_normalize_xanes(transitions, xes_trans)

    if result.emin is not None and result.emax is not None:
        spectrum = spectrum_from_transitions(transitions, lorentz_ev=result.lorentz_width,
                                peaknorm=False, erange=np.array([result.emin, result.emax]))
    elif result.linear_broaden:
        spectrum = linear_broaden_from_transitions(transitions, peaknorm=False)
    else:
        spectrum = spectrum_from_transitions(transitions, lorentz_ev=result.lorentz_width, peaknorm=False)

    if result.gaussian_broadening is not None:
        spectrum = gaussian_broaden(spectrum, width_ev=result.gaussian_broadening)

    # energy shift to match experimental data
    # xes: 19
    # from Holden
    # xanes: 50
    # A K-edge P XANES study of phosphorus compounds in solution, Persson
    spectrum = energy_shift(spectrum, result.eshift)

    outputfile = result.filename.split('/')[-1].split('.')[0]
    if outputfile == '':
        outputfile = result.filename.split('\\')[-1].split('.')[0]
    if outputfile == '':
        print('WARNING: Filename could not be parsed. Saving as outspectrum.dat')
        outputfile = 'outspectrum'

    assert not os.path.exists('{}.processedspectrum'.format(outputfile)), 'File with outputname already exists!'
    assert not os.path.exists('{}.png'.format(outputfile)), 'File with outputname already exists!'

    np.savetxt('{}.processedspectrum'.format(outputfile), spectrum.T)
