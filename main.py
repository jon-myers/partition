"""
This generates the piece. (or, it will, once I'm done composing)
"""
import pickle
import numpy as np
import matplotlib.colors as mcolors
from compose import Instrument, Piece
from funcs import golden, cent_array_to_pc, ratio_to_cents
cols = mcolors.CSS4_COLORS
keys = list(cols.keys())

f = open('saves/pitch_sets/heptagrams.txt', 'r')
exec('heptagrams = '+f.read())
heptagrams = np.array(np.sort(heptagrams))
tuning = heptagrams[30]
print(tuning)
chord = cent_array_to_pc(ratio_to_cents(tuning))[0]
print(chord)
num_of_insts = 16
inds = np.random.choice(keys, size=num_of_insts, replace=False)
td_mults = [(1/golden**0.15)**i for i in range(num_of_insts)]
insts = [Instrument(str(i), 36 + 2*i , 72 + 2*i, i+1, cols[inds[i]], td_mults[::-1][i], 'Trumpet') for i in range(num_of_insts)]

dur_tot = 20 * 60
# chord = [0,2,3,5,7,8,10]
# insts = [tr1, tr2, hn, trb, btrb]
# number of sections (6 - 12)
nos = 5 + np.random.choice(np.arange(5))
# root structure nCVI - or root structure "wobble"
rsw = 20
# event level rhythmic wobble
rhythm_nCVI_max = 20
# minimum temporal density
td_max = 10
# "octaves" of temporal density, "below" td_max
td_oct = 10
# dynamics
vel_max = 0.8
vel_min = 0.2
# minimum rest ratio
rr_min = 0.2
# maximum rest ratio
rr_max = 0.6
# rest duration maximum nCVI
rdm = 20
# rrest spread maximum nCVI
rsm = 20
# minimum rest temporal density (is this actually tied to seconds?)
rtd_min = 5



piece = Piece(dur_tot, chord, insts, nos, rsw, rhythm_nCVI_max, td_max, td_oct, \
        vel_max, rr_max, rdm, rsm, rtd_min, rr_min, vel_min, tuning)
pickle.dump(piece, open('saves/pickles/piece.p', 'wb'))
