# uncompyle6 version 3.5.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (default, Oct 25 2019, 10:52:18)
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/Jon/Documents/2019/brass_quintet/inspect.py
# Size of source mod 2**32: 71 bytes
from compose import *
import pickle
piece = pickle.load(open('saves/pickles/piece.p', 'rb'))

for section in piece.sections:
    for group in section.groups:
        for phrase in group.phrases:
            print(phrase.register)
            print('')
        print('')
    print('')
