import numpy as np
from matplotlib import pyplot as plt
from funcs import incremental_create_section_durs as icsd
from funcs import juiced_distribution_maker
from funcs import *
import random, os, itertools, shutil, pickle
import functools

class Instrument:
    """Object to collect all the relevant details of each individual instrument"""

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

class Phrase:
    """Contains parametric info and generative methods for phrases"""
    @auto_args
    def __init__(
        self, instruments, section_start_time, phrase_start_time, duration, \
        full_reg, reg_width, reg_center, full_piece_range
        ):
        self.midpoint = self.phrase_start_time + (self.duration / 2)
        self.set_register()

    def set_register(self):
        rmin = min(self.full_reg)
        rmax = max(self.full_reg)
        max_extent = rmax - rmin
        min_extent = 6
        self.extent = int(round(lin_interp(self.reg_width, min_extent, max_extent)))
        min_center = rmin + (self.extent/2)
        max_center = rmax - (self.extent/2)
        self.center = int(round(lin_interp(self.reg_center, min_center, max_center)))
        self.register = range(int(self.center - (self.extent/2)), int(self.center + (self.extent/2)))



        # tm = self.td_min
        # to = self.td_octaves
        # tw = self.td_widths
        # tc = self.td_centers

class Group:
    """Contains parametric info and generative methods for a given group"""
    @auto_args
    def __init__(
        self, instruments, pitch_set, weights, group_num, rest_ratio, \
        rest_dur_nCVI, rest_spread_nCVI, rtemp_density, section_num, \
        start_time, duration, section_partition, reg_width, reg_center, \
        full_piece_range, td_min, td_octaves, td_widths, td_centers
        ):
        self.stitch = 0
        self.make_save_dir()
        self.plot_group_ranges()
        self.set_phrase_bounds()
        self.plot_group_phrase_bounds()
        self.set_full_group_range()
        # number of phrases
        self.nop = len(self.phrase_bounds)
        self.make_registration()
        self.make_phrases()
        self.plot_group_regs()

    def make_save_dir(self):
        os.mkdir('saves/figures/sections/section_' + str(self.section_num) + '/group_' + str(self.group_num))

    def make_registration(self):
        # for each phrase, assess register width and register center at midpoint
        rws = []
        rcs = []
        for pb in self.phrase_bounds:
            # start time
            st = pb[0] + self.start_time
            # midpoint
            mp = st + pb[1]/2
            # in context of section duration
            mp_x = (mp - self.start_time) / self.duration
            # register width
            rw = lin_interp(mp_x, self.reg_width[0], self.reg_width[1])
            # register center
            rc = lin_interp(mp_x, self.reg_center[0], self.reg_center[1])
            rws.append(rw)
            rcs.append(rc)
        self.phrase_rws = rws
        self.phrase_rcs = rcs

    # using linear interp for this ... haven't really thought about if it
    # would be better to be log scale? Should think about it at some point...

    def make_phrases(self):
        phrases = []
        ins = self.instruments
        fgr = self.full_group_range
        fpr = self.full_piece_range
        sst = self.start_time
        for i in range(self.nop):
            pst = self.phrase_bounds[i][0] + sst
            dur = self.phrase_bounds[i][1]
            rw = self.phrase_rws[i]
            rc = self.phrase_rcs[i]
            phrase = Phrase(ins, sst, pst, dur, fgr, rw, rc, fpr)
            phrases.append(phrase)
        self.phrases = phrases

    # assess the full range, from instrumetns in group
    def set_full_group_range(self):
        rmin = min([inst.mp_min for inst in self.instruments])
        rmax = max([inst.mp_max for inst in self.instruments])
        self.full_group_range = range(rmin, rmax)

    def print_rest_params(self, file_):
        print(('\n\nGroup Number: ' + str(round(self.group_num, 3))), file=file_)
        print(('\nRest Ratio: ' + str(round(self.rest_ratio, 3))), file=file_)
        print(('Rest Duration nCVI: ' + str(round(self.rest_dur_nCVI, 3))), file=file_)
        print(('Rest Spread nCVI: ' + str(round(self.rest_spread_nCVI, 3))), file=file_)
        print(('Rest Temporal Density: ' + str(round(self.rtemp_density, 3))), file=file_)

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

    def plot_group_regs(self):
        fig = plt.figure(figsize=[8, 4])
        ax = fig.add_subplot(111)
        mins = [min(phrase.register) for phrase in self.phrases]
        maxs = [max(phrase.register) for phrase in self.phrases]
        mps = [phrase.midpoint for phrase in self.phrases]
        for inst in self.instruments:
            ax.fill_between(mps, mins, maxs, color = inst.color, alpha = 1 / len(self.instruments))
        plt.ylim(min(self.full_piece_range), max(self.full_piece_range))
        # plt.xlim(self.start_time, self.start_time + self.duration)
        # plt.tight_layout()
        plt.savefig(f'saves/figures/sections/section_{str(self.section_num)}/group_{str(self.group_num)}/range_envelope.png')
        plt.close()

class Section:
    """'Object which sets groupings / parameter settings for section'"""
    @auto_args
    def __init__(
        self, chord, instruments, partition, global_chord_weights, section_num, \
        duration, rest_ratio, rest_dur_nCVI, rest_spread_nCVI, rtemp_density, \
        nos, start_time, reg_widths, reg_centers, full_piece_range, td_min, \
        td_octaves, td_widths, td_centers
        ):
        # number of groups
        self.nog = len(self.partition)
        self.get_section_chord()
        self.get_weights()
        self.get_pitch_sets()
        self.make_groups()
        self.set_rparam_spread_for_groups()
        # self.set_td_delegation()
        os.mkdir('saves/figures/sections/section_' + str(section_num))
        self.instantiate_groups()
        self.plot_section_phrase_bounds()
        self.plot_section_regs()
        self.plot_section_phrase_ranges()
        self.progress()

    def plot_section_regs(self):
        fig = plt.figure(figsize=[8, 4])
        ax = fig.add_subplot(111)
        for group in self.groups:
            mins = [min(phrase.register) for phrase in group.phrases]
            maxs = [max(phrase.register) for phrase in group.phrases]
            mps = [phrase.midpoint for phrase in group.phrases]
            for inst in group.instruments:
                ax.fill_between(mps, mins, maxs, color = inst.color, alpha = (1 / len(group.instruments)))
        plt.ylim(min(self.full_piece_range), max(self.full_piece_range))
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.tight_layout()
        plt.savefig(f'saves/figures/sections/section_{str(self.section_num)}/range_envelope.png')
        plt.close()

    def plot_section_phrase_ranges(self):
        fig = plt.figure(figsize = [10, 4])
        ax = fig.add_subplot(111)
        for group in self.groups:
            mins = [min(phrase.register) for phrase in group.phrases]
            extents = [max(phrase.register) - min(phrase.register) for phrase in group.phrases]
            combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
            pb = np.array(group.phrase_bounds)
            pb[:, 0] = pb[:, 0] + self.start_time
            for p_index in range(group.nop):
                for inst in group.instruments:
                    ax.broken_barh([pb[p_index]], combs[p_index], \
                    color=(inst.color), alpha = (2/3) / len(group.instruments))
        plt.ylim(24, 72)
        plt.yticks(12 * (2 + np.arange(5)), ['C1','C2','C3', 'C4', 'C5'])
        plt.xlim(self.start_time, self.start_time + self.duration)
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.tight_layout()
        plt.savefig(f'saves/figures/sections/section_{str(self.section_num)}/phrase_ranges.png')
        plt.savefig(f'saves/figures/ranges/phrase_range_{str(self.section_num)}.png')
        plt.close()

    def progress(self):
        printProgressBar((self.section_num), (self.nos), prefix='Progress:', suffix='Complete', length=50)

    def plot_section_phrase_bounds(self):
        fig = plt.figure(figsize=[8, 1 + 0.5 * len(self.groups)])
        ax = fig.add_subplot(111)
        for group in self.groups:
            pb = np.array(group.phrase_bounds)
            pb[:, 0] = pb[:, 0] + self.start_time
            for inst in group.instruments:
                ax.broken_barh(pb, (sum(self.partition[:group.group_num - 1]), \
                    len(group.instruments)), color=(inst.color), \
                    alpha = 1 / len(group.instruments))
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
        fpr = self.full_piece_range
        tm = self.td_min
        to = self.td_octaves
        tw = self.td_widths
        tc = self.td_centers
        for i in range(self.nog):
            gp = self.grouping[i]
            ps = self.pitch_sets[i]
            psi = self.ps_indexes[i]
            rrs = self.rr_spread[i]
            rds = self.rdnCVI_spread[i]
            rss = self.rsnCVI_spread[i]
            rts = self.rtd_spread[i]
            rw = self.reg_widths[i]
            rc = self.reg_centers[i]
            g = Group(
                gp, ps, sw[psi], i+1, rrs, rds, rss, rts, sn, st, dur, p, rw, \
                rc, fpr, tm, to, tw, tc)
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

    def set_rparam_spread_for_groups(self, max_ratio=1.5):
        rp_partition = self.get_rparam_partitions()
        spread_ = []
        params = [self.rest_ratio, self.rest_dur_nCVI, self.rest_spread_nCVI, self.rtemp_density]
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
    @auto_args
    def __init__(
        self, dur_tot, chord, instruments, nos, section_dur_nCVI, \
        rhythmic_nCVI_max, td_min, td_octaves, dyns, rr_max, rdur_nCVI_max, \
        rspread_nCVI_max, rtemp_density_min, rr_min):
        self.set_weights()
        self.section_durs = icsd(nos, section_dur_nCVI) * self.dur_tot
        self.set_midpoints()
        self.set_partitions()
        self.partitions = dc_alg(list(get_partition(len(instruments))), nos)
        self.rest_delegation()
        self.init_dirs()
        self.init_progressBar()
        # self.range_delegation()
        # self.delegation(self.reg_widths, self.reg_centers)
        self.reg_widths, self.reg_centers = self.delegation()
        self.td_widths, self.td_centers = self.delegation()
        self.set_full_piece_range()
        self.make_sections()
        self.print_rest_params()
        self.print_pitch_params()
        self.plot_piece_phrase_bounds()
        self.plot_section_phrase_ranges()

    def plot_section_phrase_ranges(self):
        fig = plt.figure(figsize = [144/11, 90/22], dpi = 220)
        ax = fig.add_subplot(111)
        for section in self.sections:
            for group in section.groups:
                mins = [min(phrase.register) for phrase in group.phrases]
                extents = [max(phrase.register) - min(phrase.register) for phrase in group.phrases]
                combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
                pb = np.array(group.phrase_bounds)
                pb[:, 0] = pb[:, 0] + section.start_time
                for p_index in range(group.nop):
                    for inst in group.instruments:
                        ax.broken_barh([pb[p_index]], combs[p_index], \
                        color=(inst.color), alpha = (2/3) / len(group.instruments))
        plt.ylim(24, 72)
        plt.yticks(12 * (2 + np.arange(5)), ['C1','C2','C3', 'C4', 'C5'])
        plt.xlim(0, self.dur_tot)
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(0, self.dur_tot)
        plt.tight_layout()
        plt.savefig(f'saves/figures/ranges/phrase_ranges.png')
        plt.close()

    def set_full_piece_range(self):
        min_ = min([inst.mp_min for inst in self.instruments])
        max_ = max([inst.mp_max for inst in self.instruments])
        self.full_piece_range = range(min_, max_)

    def init_progressBar(self):
        nos = self.nos
        printProgressBar(0, nos, prefix='Progress:', suffix='Complete', length=50)

    def make_sections(self):
        sections = []
        c = self.chord
        ins = self.instruments
        gcw = self.global_chord_weights
        nos = self.nos
        sd = self.section_durs
        fpr = self.full_piece_range
        start_times = [sum(sd[:i]) for i in range(nos)]
        tdm = self.td_min
        tdo = self.td_octaves
        for i in range(nos):
            p = self.partitions[i]
            sdi = sd[i]
            rr = self.rest_ratio[i]
            rdn = self.rdur_nCVI[i]
            rsn = self.rspread_nCVI[i]
            rtd = self.rtemp_density[i]
            st = start_times[i]
            rw = self.reg_widths[i]
            rc = self.reg_centers[i]
            tw = self.td_widths[i]
            tc = self.td_centers[i]
            sec = Section(
                    c, ins, p, gcw, i+1, sdi, rr, rdn, rsn, rtd, nos, st, rw, \
                    rc, fpr, tdm, tdo, tw, tc)
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
        path3 = 'saves/figures/ranges'
        if os.path.exists(path3):
            shutil.rmtree(path3)
        os.mkdir(path3)

    def set_partitions(self):
        self.partitions = dc_alg(list(get_partition(len(self.instruments))), self.nos)

    def set_midpoints(self):
        mps = []
        for i in range(self.nos):
            mps.append((sum(self.section_durs[:i]) + self.section_durs[i] / 2) / self.dur_tot)

        self.midpoints = mps

    # original comment was lost, but I think the probs here are 1. stay the same
    # 2. have a trajectory over the course of the piece, 3. new with each section.
    def rest_delegation(self):
        probs = [1/3, 1/3, 1/3]
        loc = functools.partial(np.random.choice, np.arange(3), p=probs)
        rest_ratio = generalized_delegator(loc, get_rest_ratio, self.rr_max, self.midpoints, self.rr_min)
        rdur_nCVI = generalized_delegator(loc, get_rdur_nCVI, self.rdur_nCVI_max, self.midpoints)
        rspread_nCVI = generalized_delegator(loc, get_rspread_nCVI, self.rspread_nCVI_max, self.midpoints)
        rtemp_density = generalized_delegator(loc, get_rtemp_density, self.rtemp_density_min, self.midpoints)
        self.rest_ratio = rest_ratio
        self.rdur_nCVI = rdur_nCVI
        self.rspread_nCVI = rspread_nCVI
        self.rtemp_density = rtemp_density

    # range delegation is happening over course of piece so that it can have variety.
    # factors are width of register  0 - 1 mapping to (octave - full_range),
    # and center position 0 - 1 mapping to (full_register - width_of_register/2)
    # more complex than rest delegation, because can change over course of a section.
    # on the other hand, less complex, because I am deciding for no stability or
    # trajectory over the whole piece. That is, starting fresh with every section,
    # for better or worse. (15 mins is a long time to notice a slowly morphing register shift)
    # bounded_probs: [1 - sectionwise; 2 - groupwise]

    # # This worked fine, but got rid of so I could do exact thing for other p
    # # parameters, such as td
    # def range_delegation(self):
    #     # probabilities deciding if groups within section are tied together or not
    #     bounded_probs = [1/3, 2/3]
    #     # probabilities for if there is a trajectory or not over the course of a section
    #     # [1 - no trajectory, 2 - yes trajectory]
    #     traj_probs = [1/3, 2/3]
    #     #register width locale
    #     #register center locale
    #     reg_wit_loc = np.random.choice(np.arange(2), p = bounded_probs, size = self.nos)
    #     reg_c_loc = np.random.choice(np.arange(2), p = bounded_probs, size = self.nos)
    #     self.reg_widths = self.gen_del(reg_wit_loc, traj_probs)
    #     self.reg_centers = self.gen_del(reg_c_loc, traj_probs)

    def delegation(self):
        # probabilities deciding if groups within section are tied together or not
        bounded_probs = [1/3, 2/3]
        # probabilities for if there is a trajectory or not over the course of a section
        # [1 - no trajectory, 2 - yes trajectory]
        traj_probs = [1/3, 2/3]
        wit_loc = np.random.choice(np.arange(2), p = bounded_probs, size = self.nos)
        c_loc = np.random.choice(np.arange(2), p = bounded_probs, size = self.nos)
        wit = self.gen_del(wit_loc, traj_probs)
        c = self.gen_del(c_loc, traj_probs)
        return wit, c


    # generalized delegaor; helps not repeat code in range_delegation
    def gen_del(self, loc, traj_probs):
        out = []
        for si in range(self.nos):
            sec_out = []
            nog = len(self.partitions[si])
            if loc[si] == 0:
                traj = np.repeat(np.random.choice(np.arange(2), p = traj_probs), nog)
            if loc[si] == 1:
                traj = np.random.choice(np.arange(2), p = traj_probs, size = nog)
            for group in range(nog):
                if traj[group] == 0:
                    g_out = np.repeat(np.random.uniform(), 2)
                else:
                    g_out = np.random.uniform(size = 2)
                sec_out.append(g_out)
            out.append(sec_out)
        return out

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
                rmin = midi_pitch_to_note_name(min(group.full_group_range))
                rmax = midi_pitch_to_note_name(max(group.full_group_range))
                print(f"full range: {str(rmin)} - {str(rmax)}", file=file)
                print('', file=file)
            print('', file=file)
        file.close()

    def plot_piece_phrase_bounds(self):
        fig = plt.figure(figsize=[12, 3.5])
        ax = fig.add_subplot(111)
        for section in self.sections:
            for group in section.groups:
                for inst in group.instruments:
                    pb = np.array(group.phrase_bounds)
                    pb[:, 0] = pb[:, 0] + group.start_time
                    ax.broken_barh(pb, (sum(group.section_partition[:group.group_num - 1]), len(group.instruments)), color=(inst.color), alpha = 1/len(group.instruments))
        plt.xlim(0, self.dur_tot)
        plt.yticks([], [])
        xlocs, xlabels = plt.xticks()
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(0, self.dur_tot)
        plt.tight_layout()
        plt.savefig('saves/figures/phrases/piece.png')
