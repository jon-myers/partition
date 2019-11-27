"""
This generates the piece. (or, it will, once I'm done composing)
"""
import pickle
import numpy as np
import matplotlib.colors as mcolors
from compose import Instrument, Piece
from funcs import golden
cols = mcolors.CSS4_COLORS
keys = list(cols.keys())
inds = np.random.choice(keys, size=5, replace=False)
td_mults = [(1/golden**0.5)**i for i in range(5)]

tr1 = Instrument('Trumpet 1', 'C4', 'C6', 1, cols[inds[0]], td_mults[0])
tr2 = Instrument('Trumpet 2', 'A3', 'A5', 2, cols[inds[1]], td_mults[1])
hn = Instrument('Horn', 'C3', 'C5', 3, cols[inds[2]], td_mults[2])
trb = Instrument('Trombone', 'G2', 'G4', 4, cols[inds[3]], td_mults[3])
btrb = Instrument('Bass Trombone', 'D2', 'D4', 5, cols[inds[4]], td_mults[4])

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
td_max = 5
# "octaves" of temporal density, "below" td_max
td_oct = 3
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

piece = Piece(dur_tot, chord, insts, nos, rsw, ewob_max, td_max, td_oct, dyns,\
                rr_max, rdm, rsm, rtd_min, rr_min)
pickle.dump(piece, open('saves/pickles/piece.p', 'wb'))
