# uncompyle6 version 3.5.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (default, Oct 25 2019, 10:52:18)
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/Jon/Documents/2019/brass_quintet/inspect.py
# Size of source mod 2**32: 71 bytes
from compose import *
import pickle
from funcs import weighted_dc_alg
piece = pickle.load(open('saves/pickles/piece.p', 'rb'))

for section in piece.sections:
    print(section.section_weights)
    print('')
    for group in section.groups:
        print(group.weights)
        print(group.weight_traj)
        print('')
        for phrase in group.phrases:
            print(phrase.weights)
            print(phrase.weight_traj)
            print(phrase.weights_)
            print('')
        print('')
    print('')
#         for phrase in group.phrases:
#             print(phrase.cs_probs)
# for i in range(len(piece.sections)):
#     for j in range(len(piece.sections[i].groups)):
#         phrase = piece.sections[i].groups[j].phrases[0]
#         choices = phrase.pitch_set
#         print(choices)
#         weights = phrase.weights
#         print(weights)
#         print('')
# note_stream = weighted_dc_alg()
