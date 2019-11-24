# uncompyle6 version 3.5.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (default, Oct 25 2019, 10:52:18) 
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/Jon/Documents/2019/wind_quintet/instrument.py
# Size of source mod 2**32: 1104 bytes
from funcs import note_name_to_midi_pitch
import numpy as np
from matplotlib import pyplot as plt

class Instrument:
    """'Object to collect all the relevant details of each individual instrument'"""

    def __init__(self, instrumentName, min, max, instnum, color):
        self.name = instrumentName
        self.min = min
        self.max = max
        self.mp_min = note_name_to_midi_pitch(self.min)
        self.mp_max = note_name_to_midi_pitch(self.max) + 1
        self.range = np.arange(self.mp_min, self.mp_max)
        self.color = color

    def plot_range(self):
        fig = plt.figure(figsize=[6, 1.5])
        plt.plot((self.range), [1 for i in self.range], marker='|', color=(self.color))
        plt.xlim(24, 84)
        plt.ylim(0, 2)
        plt.xticks([12 * (2 + j) for j in range(6)], ['C' + str(j + 2) for j in range(6)])
        plt.yticks([])
        plt.title(self.name)
        plt.annotate((self.min), (self.mp_min, 1.25), ha='center')
        plt.annotate((self.max), (self.mp_max, 1.25), ha='center')
        plt.tight_layout()
        plt.savefig('saves/figures/ranges/' + str(self.name) + '.png')
# okay decompiling __pycache__/instrument.cpython-37.pyc
