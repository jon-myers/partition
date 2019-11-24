# uncompyle6 version 3.5.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (default, Oct 25 2019, 10:52:18)
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/Jon/Documents/2019/brass_quintet/funcs.py
# Size of source mod 2**32: 9193 bytes
import numpy as np, pretty_midi, itertools
from matplotlib import pyplot as plt
from scipy.stats import skewnorm

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


def dc_weight_finder(choices, alpha, weights, dictionary='no', weights_dict={}):
    choices = np.arange(len(choices))
    if dictionary == 'yes':
        if (
         weights, alpha) in weights_dict:
            pass
        return weights_dict[(weights, alpha)]
    else:
        weights_ = [i / sum(weights) for i in weights]
        for q in range(2):
            for i in range(len(choices)):
                y = dc_alg(choices, 1000, alpha, weights)
                result = np.count_nonzero(y == choices[i]) / 1000
                count = 0
                while abs(result - weights_[i]) > 0.005:
                    if result > weights_[i]:
                        weights[i] = weights[i] / 8
                    elif result < weights_[i]:
                        weights[i] = weights[i] * 7
                    else:
                        weights = [j / sum(weights) for j in weights]
                        z = dc_alg(choices, 1000, alpha, weights)
                        result = np.count_nonzero(z == choices[i]) / 1000
                        count += 1

        if dictionary == 'yes':
            weights_dict[(weights, alpha)] = weights
        return weights


def weighted_dc_alg(choices, epochs, alpha=1.0, weights=0, counts=0, verbosity=0, weights_dict={}):
    if np.any(weights) != 0:
        weights = dc_weight_finder(choices, alpha, weights, weights_dict=weights_dict)
    selections = dc_alg(choices, epochs, alpha, weights, counts, verbosity)
    return selections


def easy_midi_generator(notes, file_name, midi_inst_name, pitch_bends='no'):
    notes = [[int(i[0]), float(i[1]), float(i[2]), i[3]] for i in notes]
    notes = sorted(notes, key=(lambda x: x[1]))
    score = pretty_midi.PrettyMIDI()
    instrument_program = pretty_midi.instrument_name_to_program(midi_inst_name)
    instrument = pretty_midi.Instrument(program=instrument_program)
    if type(pitch_bends) != str:
        for i in pitch_bends:
            instrument.pitch_bends.append(pretty_midi.PitchBend(pitch=(i[0]), time=(i[1])))

    for n, note in enumerate(notes):
        for later_note in notes[n + 1:]:
            if later_note[0] == note[0]:
                note[2] = later_note[1] <= note[1] + note[2] and 0.9 * (later_note[1] - note[1])

        note = pretty_midi.Note(velocity=(note[3]), pitch=(note[0]), start=(note[1]), end=(note[1] + note[2]))
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


def juiced_distribution_maker(bins):
    out = normal_distribution_maker(bins + 4)[2:bins + 2]
    return out / np.sum(out)


def skewnorm_distribution_maker(bins, skew, focus=1.0, size=10000):
    distro = skewnorm.rvs(skew, 1, 1, size=10000)
    distro_histro = np.histogram(distro, bins=bins, density=True)[0]
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


def midi_pitch_to_note_name(midi_pitch, pcs=0):
    note = pc_to_note(midi_pitch % 12)
    octave = midi_pitch // 12 - 1
    if pcs == 1:
        return str(note)
    else:
        return str(note) + str(octave)


def note_name_to_midi_pitch(note_name):
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return 12 * int(note_name[(-1)]) + notes.index(note_name[:-1])


def lin_interp(x, start, end):
    dist = end - start
    return start + x * dist


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd='\r'):
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
    return max_ratio ** exponent


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


def get_rest_ratio(rr_min, rr_max):
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
        if get_function == get_rest_ratio:
            out = get_function(rr_min, parameter)
        else:
            out = get_function(parameter)
        out = [out for mp in midpoints]
    elif locale == 1:
        if get_function == get_rest_ratio:
            start = get_function(rr_min, parameter)
            end = get_function(rr_min, parameter)
        else:
            start = get_function(parameter)
            end = get_function(parameter)
        out = [lin_interp(mp, start, end) for mp in midpoints]
    elif get_function == get_rest_ratio:
        out = [get_function(rr_min, parameter) for mp in midpoints]
    else:
        out = [get_function(parameter) for mp in midpoints]
    return out
# okay decompiling __pycache__/funcs.cpython-37.pyc
