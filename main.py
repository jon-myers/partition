from compose import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from inspect import signature

c_dict = mcolors.CSS4_COLORS
keys = list(c_dict.keys())
c_indexes = np.random.choice(keys, size = 5, replace = False)
# print(c_dict[c_indexes[0]])
tr1 = Instrument('Trumpet 1', 'C4', 'C6', 1, c_dict[c_indexes[0]])
tr2 = Instrument('Trumpet 2', 'A3', 'A5', 2, c_dict[c_indexes[1]])
hn = Instrument('Horn', 'C3', 'C5', 3, c_dict[c_indexes[2]])
trb = Instrument('Trombone', 'G2', 'G4', 4, c_dict[c_indexes[3]])
btrb = Instrument('Bass Trombone', 'D2', 'D4', 5, c_dict[c_indexes[4]])

dur_tot = 14 * 60
chord = [0, 2, 3, 7, 8, 10]
insts = [tr1, tr2, hn, trb, btrb]
# number of sections
nos = 5 + np.random.choice(np.arange(5))
# root structure nCVI - or root structure "wobble"
rsw = np.random.uniform(5, 20)
# event level rhythmic wobble
ewob_max = 15
# minimum temporal density
td_min = 5
# "octaves" of temporal density, above td_min
td_oct = 4
dyns = ['pp', 'p', 'mp', 'mf']
# minimum rest ratio
rr_min = 0.25
# maximum rest ratio
rr_max = 0.75
# rest duration maximum nCVI
rdm = 20
# rrest spread maximum nCVI
rsm = 20
# minimum rest temporal density (is this actually tied to seconds?)
rtd_min = 8
# minimum rest ratio
rr_min = 0.25

piece = Piece(dur_tot, chord, insts, nos, rsw, ewob_max, td_min, td_oct, dyns,\
                rr_max, rdm, rsm, rtd_min, rr_min)
pickle.dump(piece, open('saves/pickles/piece.p', 'wb'))
