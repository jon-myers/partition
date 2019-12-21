import numpy as np, pretty_midi, itertools
from matplotlib import pyplot as plt
from scipy.stats import skewnorm
from inspect import signature

def dc_alg(choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0):
    selections = []
    if np.all(counts == 0):
        counts = [
         1] * len(choices)
    weights = np.array(weights)
    if np.all(weights) == 0:
        weights = [
         1] * len(choices)
    for q in range(epochs):
        sum_ = sum([weights[i] * counts[i] ** alpha for i in range(len(choices))])
        probs = [weights[i] * counts[i] ** alpha / sum_ for i in range(len(choices))]
        selection_index = np.random.choice((list(range(len(choices)))), p=probs)
        counts = [i + 1 for i in counts]
        counts[selection_index] = 0
        selections.append(choices[selection_index])

    selections = np.array(selections)
    counts = np.array(counts)
    if verbosity == 0:
        return selections
    if verbosity == 1:
        return (
         selections, counts)


def hz_to_cents(hz, root):
    return 1200 * np.log2(hz / root)

def dc_weight_finder(choices, alpha, weights, test_epochs=500):
    choices = np.arange(len(choices))
    weights_ = [i / sum(weights) for i in weights]
    max_off = .011
    # cts_ = 0
    while max_off > 0.01:
        # print(cts_)
        y = dc_alg(choices, test_epochs, alpha, weights)
        #this should be rewritten as a np function
        results = np.array([np.count_nonzero(y==choices[i]) / test_epochs for i in choices])
        diff = weights_ / results
        weights *= diff
        weights /= sum(weights)
        max_off = np.max(1 - diff)
        # cts_+=1
        # print(cts_)
    return weights

def weighted_dc_alg(choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0, weights_dict={}):
    if np.any(weights) != 0:
        # this basically says if its not going to work, just double the length
        #of the choice array, and try again. Might be better to just double the
        # one value thats above 0.5 . Or, might make more sense to just do a straight
        # random choice.
        if np.max(weights) >= 0.5:
            choices = np.tile(choices, 2)
            weights = np.tile(weights/2, 2)
            counts = np.tile(counts, 2)
        weights = dc_weight_finder(choices, alpha, weights)
    selections = dc_alg(choices, epochs, alpha, weights, counts, verbosity)
    return selections


def easy_midi_generator(notes, file_name, midi_inst_name):
    notes = sorted(notes, key=(lambda x: x[1]))
    score = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(midi_inst_name)
    instrument = pretty_midi.Instrument(program=0)
    for n, note in enumerate(notes):
        if type(note[3]) == np.float64:
            vel = np.int(np.round(127 * note[3]))
        elif type(note[3]) == float:
            vel = np.int(np.round(127 * note[3]))
        elif type(note[3]) == int:
            vel = note[3]
        else: print(note[3])
        note = pretty_midi.Note(velocity=vel, pitch=(note[0]), start=(note[1]), end=(note[1] + note[2]))
        instrument.notes.append(note)
    score.instruments.append(instrument)
    score.write(file_name)


def nPVI(d):
    m = len(d)
    return 100 / (m - 1) * sum([abs((d[i] - d[(i + 1)]) / (d[i] + d[(i + 1)]) / 2) for i in range(m - 1)])

def nPVI_averager(window_width, durs):
    return [nPVI(durs[i:i + window_width]) for i in range(len(durs) - window_width)]


def nCVI(d):
    matrix = [list(i) for i in itertools.combinations(d, 2)]
    matrix = [nPVI(i) for i in matrix]
    return sum(matrix) / len(matrix)


def pc_to_note(pc):
    notes = [
     'C', 'C#', 'D', 'E#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note = notes[pc]
    return note


def normal_distribution_maker(bins):
    distribution = np.random.normal(size=10000)
    distribution = np.histogram(distribution, bins=bins, density=True)[0]
    distribution /= np.sum(distribution)
    return distribution


def tunable_distribution_maker(bins, center=0, spread=1):
    """Bins are size of distribution, center is location of peak, from 0 - 1,
    spread is sharpness of peak, with 0 being sharper than 1 """
    spread = 1 - spread
    center = 1-center
    distribution = np.random.normal(size=10000)
    dmax = np.max(distribution)
    dmin = np.min(distribution)
    dmax = np.max(np.abs(np.array([dmax, dmin])))
    spread *= (dmax)
    if bins %2 == 1:
        bins_ = 2 * bins - 1
        bottom = np.int(np.round(center*(bins-1)))
        distribution = np.histogram(distribution, range=(-1*spread, spread), bins=bins_, density=True)[0][bottom:bottom+bins]
    else:
        bins_ = 2 * bins
        bottom = np.int(np.round(center*(bins)))
        distribution = np.histogram(distribution, range=(-1*spread, spread), bins=bins_, density=True)[0][bottom:bottom+bins]
    distribution /= np.sum(distribution)
    return distribution

def juiced_distribution_maker(bins):
    out = normal_distribution_maker(bins + 4)[2:bins + 2]
    return out / np.sum(out)

# print(tunable_distribution_maker(3, 0.1, 0.3))


def skewnorm_distribution_maker(bins, skew, focus=1.0, size=10000):
    distro = skewnorm.rvs(skew, 1, 1, size=10000)
    distro_histro = np.histogram(distro, bins=bins, range=(0,10), density=True)[0]
    distro_histro = [i ** 0.25 for i in distro_histro]
    distro_histro /= np.sum(distro_histro)
    return distro_histro

def incremental_create_section_durs(num_of_thoughts,nCVI_average,factor=2.0):
    section_durs = factor ** np.random.normal(size=2)
    while abs(nCVI(section_durs) - nCVI_average) > 1.0:
        section_durs = factor ** np.random.normal(size=2)
    for i in range(num_of_thoughts - 2):
        next_section_durs = np.append(section_durs,[factor ** np.random.normal()])
        ct=0
        while abs(nCVI(next_section_durs) - nCVI_average) > 1.0:
            ct+=1
            next_section_durs = np.append(section_durs, [factor ** np.random.normal()])
        section_durs = next_section_durs
        # print(ct)
    section_durs /= np.sum(section_durs)
    return section_durs

# midi pitch to note name
def mp_to_nn(midi_pitch, pcs=0):
    note = pc_to_note(midi_pitch % 12)
    octave = midi_pitch // 12 - 1
    if pcs == 1:
        return str(note)
    else:
        return str(note) + str(octave)

# note name to midi pitch
def nn_to_mp(note_name):
    if type(note_name) == int:
        return note_name
    else:
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        return 12 * int(note_name[(-1)]) + notes.index(note_name[:-1])

def lin_interp(x, start, end):
    dist = end - start
    return start + x * dist

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd='\r'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "
", "
") (Str)
    """
    percent = ('{0:.' + str(decimals) + 'f}').format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix)), end=printEnd)
    if iteration == total:
        print()


def spread(init, max_ratio):
    exponent = np.clip(np.random.normal() / 3, -1, 1)
    return init * (max_ratio ** exponent)


def get_partition(n):
    a = [0 for i in range(n + 1)]
    k = 1
    y = n - 1
    while k != 0:
            x = a[(k - 1)] + 1
            k -= 1
            while 2 * x <= y:
                    a[k] = x
                    y -= x
                    k += 1
            l = k + 1
            while x <= y:
                    a[k] = x
                    a[l] = y
                    yield a[:k + 2]
                    x += 1
                    y -= 1

            a[k] = x + y
            y = x + y - 1
            yield a[:k + 1]

def secs_to_mins(secs):
    mins = np.int(secs // 60)
    secs = np.int(secs % 60)
    mins = str(mins)
    secs = str(secs)
    if len(secs) == 1:
        secs = '0' + secs
    return mins + ':' + secs

# get rest ratio
def get_rr(rr_min, rr_max):
    out = np.random.uniform(rr_min, rr_max)
    return out


def get_rdur_nCVI(rdur_nCVI_max):
    return np.random.uniform(0, rdur_nCVI_max)


def get_rspread_nCVI(rspread_nCVI_max):
    return np.random.uniform(0, rspread_nCVI_max)


def get_rtemp_density(rtemp_density_min=5, octaves=3):
    return rtemp_density_min * 2 ** np.random.uniform(0, octaves)


def generalized_delegator(locale, get_function, parameter, midpoints, rr_min=0):
    if locale == 0:
        if get_function == get_rr:
            out = get_function(rr_min, parameter)
        else:
            out = get_function(parameter)
        out = [out for mp in midpoints]
    elif locale == 1:
        if get_function == get_rr:
            start = get_function(rr_min, parameter)
            end = get_function(rr_min, parameter)
        else:
            start = get_function(parameter)
            end = get_function(parameter)
        out = [lin_interp(mp, start, end) for mp in midpoints]
    elif get_function == get_rr:
        out = [get_function(rr_min, parameter) for mp in midpoints]
    else:
        out = [get_function(parameter) for mp in midpoints]
    return out

def auto_args(target):
    """
    A decorator for automatically copying constructor arguments to `self`.
    """
    # Get a signature object for the target method:
    sig = signature(target)
    def replacement(self, *args, **kwargs):
        # Parse the provided arguments using the target's signature:
        bound_args = sig.bind(self, *args, **kwargs)
        # Save away the arguments on `self`:
        for k, v in bound_args.arguments.items():
            if k != 'self':
                setattr(self, k, v)
        # Call the actual constructor for anything else:
        target(self, *args, **kwargs)
    return replacement

# the golden ratio; You never know when you'll need it!
golden = (1 + 5**0.5) / 2


def fill_space(text):
    return text.replace(' ', '_')
