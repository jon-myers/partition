# uncompyle6 version 3.5.1
# Python bytecode 3.7 (3394)
# Decompiled from: Python 3.7.5 (default, Oct 25 2019, 10:52:18)
# [Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: /Users/Jon/Documents/2019/brass_quintet/compose.py
# Size of source mod 2**32: 19065 bytes
import numpy as np
from matplotlib import pyplot as plt
from funcs import incremental_create_section_durs as icsd
from funcs import *
from instrument import Instrument
import random, os, itertools, shutil, pickle

class Group:
    """'Object which contains all the parametric info for a given group'"""

    def __init__(self, instruments, pitch_set, weights, group_num, rest_ratio, rest_dur_nCVI, rest_spread_nCVI, rtemp_density, section_num, start_time, duration, section_partition):
        self.section_partition = section_partition
        self.duration = duration
        self.start_time = start_time
        self.instruments = instruments
        self.pitch_set = pitch_set
        self.weights = weights
        self.group_num = group_num
        self.rest_ratio = rest_ratio
        self.rest_dur_nCVI = rest_dur_nCVI
        self.rest_spread_nCVI = rest_spread_nCVI
        self.rtemp_density = rtemp_density
        self.stitch = 0
        self.section_num = section_num
        os.mkdir('saves/figures/sections/section_' + str(self.section_num) + '/group_' + str(self.group_num))
        self.plot_group_ranges()
        'getting_to_here?'
        self.set_phrase_bounds()
        self.plot_group_phrase_bounds()

    def print_rest_params(self, file_):
        print(('\n\nGroup Number: ' + str(round(self.group_num, 3))), file=file_)
        print(('\nRest Ratio: ' + str(round(self.rest_ratio, 3))), file=file_)
        print(('Rest Duration nCVI: ' + str(round(self.rest_dur_nCVI, 3))), file=file_)
        print(('Rest Spread nCVI: ' + str(round(self.rest_spread_nCVI, 3))), file=file_)
        print(('Rest Temporal Density: ' + str(round(self.rtemp_density, 3))), file=file_)

    def full_range(self):
        rmin = min([inst.mp_min for inst in self.instruments])
        rmax = max([inst.mp_max for inst in self.instruments])
        return range(rmin, rmax)

    def plot_group_ranges(self):
        fig = plt.figure(figsize=[8, 1.0 + 0.5 * len(self.instruments)])
        ax = fig.add_subplot(111)
        for i, inst in enumerate(self.instruments[::-1]):
            plt.plot((inst.range), [i + 0.5 for x in inst.range], marker='|', color=(inst.color),
              label=(inst.name))
            plt.annotate((inst.min), (inst.mp_min, 0.75 + i), ha='center')
            plt.annotate((inst.max), (inst.mp_max, 0.75 + i), ha='center')

        plt.xlim(24, 84)
        plt.ylim(0, len(self.instruments) + 1)
        plt.xticks([12 * (2 + j) for j in range(7)], ['C' + str(j + 2) for j in range(7)])
        plt.yticks([])
        plt.title('Section ' + str(self.section_num) + ' Group ' + str(self.group_num))
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1])
        plt.tight_layout()
        plt.savefig(f"saves/figures/sections/section_{self.section_num}/group_{self.group_num}/ranges.png")
        plt.close()

    def set_phrase_bounds(self):
        num_of_rests = int(round(self.duration / self.rtemp_density))
        if self.stitch == 0:
            pass
        if num_of_rests < 2:
            num_of_rests = 2
        if self.stitch == 1 or self.stitch == 2:
            if num_of_rests < 1:
                num_of_rests == 1
        else:
            num_of_plays = num_of_rests
        rest_dur_tot = self.rest_ratio * self.duration
        rest_durs = icsd(num_of_rests, self.rest_dur_nCVI) * rest_dur_tot
        if self.stitch == 0:
            num_of_plays = num_of_rests - 1
        play_durs = icsd(num_of_plays, self.rest_spread_nCVI) * (self.duration - rest_dur_tot)
        self.phrase_bounds = self.get_phrase_bounds(play_durs, rest_durs)

    def get_phrase_bounds(self, plays, rests):
        if self.stitch == 1:
            play_starts = [sum(rests[:i]) + sum(plays[:i]) for i in range(len(plays))]
        else:
            play_starts = [rests[0] + sum(rests[1:i + 1]) + sum(plays[:i]) for i in range(len(plays))]
        out = [(play_starts[i], plays[i]) for i in range(len(plays))]
        return out

    def plot_group_phrase_bounds(self):
        fig = plt.figure(figsize=[8, 1.5])
        ax = fig.add_subplot(111)
        pb = np.array(self.phrase_bounds)
        pb[:, 0] = pb[:, 0] + self.start_time
        ax.broken_barh(pb, (1, 1), color=(self.instruments[0].color))
        plt.title(f"Section {str(self.section_num)}, Group {str(self.group_num)} - {', '.join([str(i.name) for i in self.instruments])}")
        plt.yticks([], [])
        plt.xlim(self.start_time, self.start_time + self.duration)
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.tight_layout()
        plt.savefig(f"saves/figures/sections/section_{self.section_num}/group_{self.group_num}/phrase_bounds.png")
        plt.close()

class Section:
    """'Object which sets groupings / parameter settings for section'"""

    def __init__(self, chord, instruments, partition, global_chord_weights, section_num, duration, rest_ratio, rest_dur_nCVI, rest_spread_nCVI, rtemp_density, num_of_sections, start_time):
        self.num_of_sections = num_of_sections
        self.instruments = instruments
        self.rr = rest_ratio
        self.rdn = rest_dur_nCVI
        self.rsn = rest_spread_nCVI
        self.rtd = rtemp_density
        self.partition = partition
        self.start_time = start_time
        self.section_num = section_num
        self.chord = chord
        self.global_chord_weights = global_chord_weights
        self.get_section_chord()
        self.get_weights()
        self.get_pitch_sets()
        self.duration = duration
        self.make_groups()
        self.get_rparam_spread_for_groups()
        os.mkdir('saves/figures/sections/section_' + str(section_num))
        self.instantiate_groups()
        self.plot_section_phrase_bounds()
        self.progress()

    def progress(self):
        printProgressBar((self.section_num), (self.num_of_sections), prefix='Progress:', suffix='Complete', length=50)

    def plot_section_phrase_bounds(self):
        fig = plt.figure(figsize=[8, 1 + 0.5 * len(self.groups)])
        ax = fig.add_subplot(111)
        for group in self.groups:
            pb = np.array(group.phrase_bounds)
            pb[:, 0] = pb[:, 0] + self.start_time
            ax.broken_barh(pb, (sum(group.section_partition[:group.group_num - 1]), len(group.instruments)), color=(group.instruments[0].color))

        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.yticks([], [])
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.tight_layout()
        plt.savefig(f"saves/figures/sections/section_{self.section_num}/phrases.png")
        plt.savefig(f"saves/figures/phrases/section_{self.section_num}.png")

    def instantiate_groups(self):
        groups = []
        sw = self.section_weights
        sn = self.section_num
        st = self.start_time
        dur = self.duration
        p = self.partition
        for i in range(len(self.grouping)):
            gp = self.grouping[i]
            ps = self.pitch_sets[i]
            psi = self.ps_indexes[i]
            rrs = self.rr_spread[i]
            rds = self.rdnCVI_spread[i]
            rss = self.rsnCVI_spread[i]
            rts = self.rtd_spread[i]
            g = Group(gp, ps, sw[psi], i + 1, rrs, rds, rss, rts, sn, st, dur, p)
            groups.append(g)

        self.groups = groups

    def make_groups(self):
        partial_insts = self.instruments.copy()
        groups = []
        for part in self.partition:
            group = random.sample(partial_insts, part)
            groups.append(group)
            for inst in group:
                partial_insts.remove(inst)

        self.grouping = groups

    def get_rparam_partitions(self):
        partitions = []
        for param in range(4):
            partition = [i for i in get_partition(len(self.grouping))]
            if len(partition) != 1:
                partition = random.choice(partition)
            else:
                partition = partition[0]
            random.shuffle(partition)
            partitions.append(partition)

        return partitions

    def get_rparam_spread_for_groups(self, max_ratio=1.75):
        rp_partition = self.get_rparam_partitions()
        spread_ = []
        params = [self.rr, self.rdn, self.rsn, self.rtd]
        for i, param in enumerate(params):
            s = [[spread(param, max_ratio) * param for k in range(j)] for j in rp_partition[i]]
            s = [j for j in itertools.chain.from_iterable(s)]
            spread_.append(s)

        self.rr_spread, self.rdnCVI_spread, self.rsnCVI_spread, self.rtd_spread = spread_

    def get_section_chord(self):
        size = np.random.choice((np.arange(len(self.chord) - 2) + 3), p=(juiced_distribution_maker(len(self.chord) - 2)))
        self.section_chord = np.random.choice((self.chord), p=(self.global_chord_weights), size=size, replace=False)

    def get_weights(self, standard_dist=0.2):
        weights = np.random.normal(0.5, standard_dist, len(self.section_chord))
        while np.all((weights == np.abs(weights)), axis=0) == False:
                weights = np.random.normal(0.5, standard_dist, len(self.section_chord))

        self.section_weights = weights / np.sum(weights)

    def print_groups(self):
        print('Section: ' + str(self.section_num))
        for i, group in enumerate(self.grouping):
            print('Group ' + str(i + 1) + ':')
            for inst in group:
                print(inst.name)

            print('\n')

    def get_pitch_sets(self):
        dist = juiced_distribution_maker(len(self.section_chord))
        pitch_sets = []
        indexes = []
        for group in range(len(self.partition)):
            pcs_size = np.random.choice((np.arange(len(self.section_chord)) + 1), p=dist)
            ps_index = np.random.choice((np.arange(len(self.section_chord))), p=(self.section_weights), size=pcs_size, replace=False)
            ps = self.section_chord[ps_index]
            pitch_sets.append(ps)
            indexes.append(ps_index)

        self.pitch_sets = pitch_sets
        self.ps_indexes = indexes


class Piece:
    """'Object which generates all sections, delegates top level params, etc'"""

    def __init__(
        self, dur_tot, chord, instruments, num_of_sections, section_dur_nCVI, \
        rhythmic_nCVI_max, td_min, td_octaves, dyns, rr_max, rdur_nCVI_max, \
        rspread_nCVI_max, rtemp_density_min, rr_min):
        self.chord = chord
        self.instruments = instruments
        self.num_of_sections = num_of_sections
        self.chord = chord
        self.rr_min = rr_min
        self.dur_tot = dur_tot
        self.rr_max = rr_max
        self.rdur_nCVI_max = rdur_nCVI_max
        self.rspread_nCVI_max = rspread_nCVI_max
        self.rtemp_density_min = rtemp_density_min
        self.rr_min = rr_min
        self.set_weights()
        self.section_durs = icsd(num_of_sections, section_dur_nCVI) * self.dur_tot
        self.set_midpoints()
        self.set_partitions()
        self.partitions = dc_alg(list(get_partition(len(instruments))), num_of_sections)
        self.rest_delegation()
        self.init_dirs()
        self.init_progressBar()
        self.make_sections()
        self.print_rest_params()
        self.print_pitch_params()
        self.plot_piece_phrase_bounds()

    def init_progressBar(self):
        nos = self.num_of_sections
        printProgressBar(0, nos, prefix='Progress:', suffix='Complete', length=50)

    def make_sections(self):
        sections = []
        c = self.chord
        ins = self.instruments
        gcw = self.global_chord_weights
        nos = self.num_of_sections
        sd = self.section_durs
        start_times = [sum(sd[:i]) for i in range(nos)]
        for i in range(nos):
            p = self.partitions[i]
            sdi = sd[i]
            rr = self.rest_ratio[i]
            rdn = self.rdur_nCVI[i]
            rsn = self.rspread_nCVI[i]
            rtd = self.rtemp_density[i]
            st = start_times[i]
            sec = Section(c, ins, p, gcw, i + 1, sdi, rr, rdn, rsn, rtd, nos, st)
            sections.append(sec)

        self.sections = sections

    def init_dirs(self):
        path1 = 'saves/figures/sections'
        if os.path.exists(path1):
            shutil.rmtree(path1)
        os.mkdir(path1)
        path2 = 'saves/figures/phrases'
        if os.path.exists(path2):
            shutil.rmtree(path2)
        os.mkdir(path2)

    def set_partitions(self):
        self.partitions = dc_alg(list(get_partition(len(self.instruments))), self.num_of_sections)

    def set_midpoints(self):
        mps = []
        for i in range(self.num_of_sections):
            mps.append((sum(self.section_durs[:i]) + self.section_durs[i] / 2) / self.dur_tot)

        self.midpoints = mps

    def rest_delegation(self):
        probs = [
         0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
        rr_locale, rdur_locale, rspread_locale, rtemp_density_locale = np.random.choice((np.arange(3)), p=probs, size=4)
        rest_ratio = generalized_delegator(rr_locale, get_rest_ratio, self.rr_max, self.midpoints, self.rr_min)
        rdur_nCVI = generalized_delegator(rdur_locale, get_rdur_nCVI, self.rdur_nCVI_max, self.midpoints)
        rspread_nCVI = generalized_delegator(rspread_locale, get_rspread_nCVI, self.rspread_nCVI_max, self.midpoints)
        rtemp_density = generalized_delegator(rtemp_density_locale, get_rtemp_density, self.rtemp_density_min, self.midpoints)
        self.rest_ratio = rest_ratio
        self.rdur_nCVI = rdur_nCVI
        self.rspread_nCVI = rspread_nCVI
        self.rtemp_density = rtemp_density

    def set_weights(self, standard_dist=0.2):
        weights = np.random.normal(0.5, standard_dist, len(self.chord))
        while np.all((weights == np.abs(weights)), axis=0) == False:
                weights = np.random.normal(0.5, standard_dist, len(self.chord))

        self.global_chord_weights = weights / np.sum(weights)

    def print_rest_params(self):
        file = open('saves/text_printouts/rest_params.txt', 'w')
        for section_number, section in enumerate(self.sections):
            print(('\nSection Number: ' + str(section_number + 1)), file=file)
            for group in section.groups:
                group.print_rest_params(file)

        file.close()

    def print_pitch_params(self):
        file = open('saves/text_printouts/pitch_params.txt', 'w')
        print('Pitch Parameters', file=file)
        print('', file=file)
        for section in self.sections:
            print(f"Section {str(section.section_num)}: ", file=file)
            for group in section.groups:
                print(f"Group {str(group.group_num)}: ", file=file)
                print(f"pitch set: {', '.join([str(i) for i in group.pitch_set])}", file=file)
                print('', file=file)

            print('', file=file)

        file.close()

    def plot_piece_phrase_bounds(self):
        fig = plt.figure(figsize=[12, 3.5])
        ax = fig.add_subplot(111)
        for section in self.sections:
            for group in section.groups:
                pb = np.array(group.phrase_bounds)
                pb[:, 0] = pb[:, 0] + group.start_time
                ax.broken_barh(pb, (sum(group.section_partition[:group.group_num - 1]), len(group.instruments)), color=(group.instruments[0].color))

        plt.xlim(0, self.dur_tot)
        plt.yticks([], [])
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(0, self.dur_tot)
        plt.tight_layout()
        plt.savefig('saves/figures/phrases/piece.png')
# okay decompiling __pycache__/compose.cpython-37.pyc
