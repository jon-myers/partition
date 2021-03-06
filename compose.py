import random, os, itertools, shutil, functools
import numpy as np
from matplotlib import pyplot as plt
from funcs import incremental_create_section_durs as icsd
from funcs import juiced_distribution_maker as jdm
from funcs import *
from math import log2, ceil
import json

class Instrument:
    """Object to collect all the relevant details of each individual instrument"""
    @auto_args
    def __init__(self, name, min, max, instnum, color, td_mult, midi_name):
        self.mp_min = nn_to_mp(self.min)
        self.mp_max = nn_to_mp(self.max) + 1
        self.range = np.arange(self.mp_min, self.mp_max)
        self.notes = []

    def plot_range(self):
        fig = plt.figure(figsize=[6, 1.5])
        plt.plot((self.range), np.repeat(1, self.range), marker='|', color=(self.color))
        plt.xlim(24, 84)
        plt.ylim(0, 2)
        plt.xticks([12 * (2 + j) for j in range(6)], ['C' + str(j + 2) for j in range(6)])
        plt.yticks([])
        plt.title(self.name)
        plt.annotate((self.min), (self.mp_min, 1.25), ha='center')
        plt.annotate((self.max), (self.mp_max, 1.25), ha='center')
        plt.tight_layout()
        plt.savefig('saves/figures/ranges/' + str(self.name) + '.png')

        # dyns
    def make_notes(self, note_stream, starts, durs, vels):
        for n_i, note in enumerate(note_stream):
            if note != 0:
                self.notes.append([note, starts[n_i], durs[n_i], vels])

class Phrase:
    """Contains parametric info and generative methods for phrases"""
    @auto_args
    def __init__(
        self, instruments, section_start_time, phrase_start_time, duration,    \
        full_reg, reg_width, reg_center, full_piece_range, td_frame_top,       \
        td_octaves, td_width, td_center, nCVI_width, nCVI_center,              \
        rhythm_nCVI_max, pitch_set, weights, weight_traj, section_duration,    \
        cs_width, cs_center, vel_width, vel_center, vel_max, vel_min
        ):
        self.midpoint = self.phrase_start_time + (self.duration/2)
        self.set_register()
        self.set_td()
        self.set_nCVI()
        self.non = np.int(np.floor(self.duration * self.td))
        if self.non == 0:
            self.non = 1
        self.note_durs = icsd(self.non, self.nCVI) * self.duration
        if False in self.note_durs: print ('got em')
        self.note_starts = np.array([np.sum(self.note_durs[:i]) for i in range(self.non)])
        self.noi = len(instruments)
        self.set_cs_probs()
        self.make_cs_array()
        self.set_weights()
        self.adjust_pitchset()
        self.make_note_streams()
        self.delegate_note_stream()
        self.set_vels()
        self.set_notes()


    def set_vels(self):
        """Sets the vels, first by doing ranges like td or register, then by
        choosing a uniform random from in that range"""
        vel_max = self.vel_max
        vel_min = self.vel_min
        max_extent = self.vel_max - self.vel_min
        extent = lin_interp(self.vel_width, self.vel_min, self.vel_max)
        min_center = self.vel_min + (extent/2)
        max_center = self.vel_max - (extent/2)
        center = lin_interp(self.vel_center, min_center, max_center)
        self.vel_bounds = [center - extent/2, center + extent/2]
        self.vels = np.random.uniform(self.vel_bounds[0], self.vel_bounds[1])


    def set_notes(self):
        for i, row in enumerate(self.note_matrix):
            self.instruments[i].make_notes(row, self.note_starts+self.phrase_start_time, self.note_durs, self.vels)

    # make a big matrix, the width is all the notes, the height is the insts.
    def delegate_note_stream(self):
        ct=0
        note_matrix = np.zeros((len(self.instruments),len(self.cs_array)), dtype=int)
        for i, cs in enumerate(self.cs_array):
            #choose which inst to send chord to
            inst_choices = np.random.choice(np.arange(len(self.instruments)), replace=False, size=cs)
            for j in inst_choices:
                note_matrix[j,i] = self.note_stream[ct]
                ct+=1

        # bring into correct octave, hopefully with some lookback, so that slightly more
        # often it stays in the closer octave to previous note
        for nm_i, inst_pcs in enumerate(note_matrix):
            for pc_i, pc in enumerate(inst_pcs):
                if pc != 0:
                    choices = [pc + 12*i for i in range(10) if pc+12*i in self.register]
                    choices = [i for i in choices if i in self.instruments[nm_i].range]
                    if len(choices) == 0:
                        note_matrix[nm_i,pc_i] = 0
                    elif 'prev' in locals():
                        diffs = np.abs(choices - prev)+6
                        weights = 1/diffs
                        weights /= np.sum(weights)
                        # print(weights)
                        prev = np.random.choice(choices, p = weights)
                        note_matrix[nm_i,pc_i] = prev
                    else:
                        prev = np.random.choice(choices)
                        note_matrix[nm_i,pc_i] = prev
        self.note_matrix = note_matrix

    def make_note_streams(self):
        ns_len = np.sum(self.cs_array)
        if len(self.pitch_set) == 1:
            self.note_stream = np.repeat(self.pitch_set, ns_len)
        else:
            choice_list = [[self.pitch_set[i] for j in range(int(ceil(self.weights_[i] * ns_len)))] for i in range(len(self.pitch_set))]
            choice_list = [i for i in itertools.chain.from_iterable(choice_list)]
            self.note_stream = np.random.choice(choice_list, replace=False, size = ns_len)

    def make_cs_array(self):
        if self.noi == 1:
            self.cs_array = np.repeat(1,self.non)
        else:
            self.cs_array = np.random.choice(np.arange(self.noi)+1, self.non, p=self.cs_probs)


    def set_cs_probs(self):
        self.cs_probs = tunable_distribution_maker(self.noi, self.cs_center, self.cs_width)

    def set_weights(self):
        x  = (self.midpoint - self.section_start_time) / self.section_duration
        wt = self.weight_traj
        self.weights_ = np.array([lin_interp(x, wt[0][i], wt[1][i]) for i in range(len(wt[0]))])

    def adjust_pitchset(self):
        """Make the pitchset and weights align with the assigned register"""
        register_pitch_set = np.array(list(set([i%12 for i in self.register])))
        is_in = np.isin(self.pitch_set, register_pitch_set)
        if not np.all(is_in):
            self.pitch_set = self.pitch_set[is_in]
            self.weights_ = self.weights_[is_in]
            self.weights_ = self.weights_ / np.sum(self.weights_)

    def set_register(self):
        rmin = min(self.full_reg)
        rmax = max(self.full_reg)
        max_extent = rmax - rmin
        min_extent = 12
        extent = int(round(lin_interp(self.reg_width, min_extent, max_extent)))
        min_center = rmin + (extent/2)
        max_center = rmax - (extent/2)
        center = int(round(lin_interp(self.reg_center, min_center, max_center)))
        self.register = range(int(center - (extent/2)), int(center + (extent/2)))

    def set_td(self):
        """Sets the temporal density"""
        td_max = np.log2(self.td_frame_top)
        td_min = td_max - self.td_octaves
        max_extent = td_max - td_min
        extent = lin_interp(self.td_width, 0.125, max_extent)
        min_center = td_min + (extent/2)
        max_center = td_max - (extent/2)
        center = lin_interp(self.td_center, min_center, max_center)
        log_td_bounds = [center - extent/2, center + extent/2]
        # make this log? currently, won't this trend up?
        log_td = np.random.uniform(log_td_bounds[0], log_td_bounds[1])
        self.td_bounds = 2 ** (np.array(log_td_bounds))
        self.td = 2 ** (log_td)

    def set_nCVI(self):
        """Sets the nCVI, first by doing ranges like td or register, then by
        choosing a uniform random from in that range"""
        nCVI_max = self.rhythm_nCVI_max
        nCVI_min = 0
        max_extent = nCVI_max - nCVI_min
        extent = lin_interp(self.nCVI_width, nCVI_min, nCVI_max)
        min_center = nCVI_min + (extent/2)
        max_center = nCVI_max - (extent/2)
        center = lin_interp(self.nCVI_center, min_center, max_center)
        self.nCVI_bounds = [center - extent/2, center + extent/2]
        # should this be log scale?
        self.nCVI = np.random.uniform(self.nCVI_bounds[0], self.nCVI_bounds[1])

class Group:
    """Contains parametric info and generative methods for a given group"""
    @auto_args
    def __init__(
        self, instruments, pitch_set, weights, group_num, rest_ratio,          \
        rest_dur_nCVI, rest_spread_nCVI, rtemp_density, section_num,           \
        start_time, duration, section_partition, reg_width, reg_center,        \
        full_piece_range, td_max, td_octaves, td_width, td_center, nCVI_width, \
        nCVI_center, rhythm_nCVI_max, cs_width, cs_center, stitch, vel_width,  \
        vel_center, vel_max, vel_min
        ):
        self.make_save_dir()
        self.plot_group_ranges()
        self.set_phrase_bounds()
        self.plot_group_phrase_bounds()
        self.set_full_group_range()
        self.nop = len(self.phrase_bounds)
        self.phrase_traj_interp()
        self.set_td_frame()
        self.make_weight_traj()
        self.make_phrases()
        self.plot_group_regs()

    def make_weight_traj(self):
        a = np.array([spread(i, 1.25) for i in self.weights])
        a = a / np.sum(a)
        b = np.array([spread(i, 1.25) for i in self.weights])
        b = b / np.sum(b)
        self.weight_traj = [a, b]
    def set_td_frame(self):
        mults = np.array([inst.td_mult for inst in self.instruments])
        avg_td_mult = 2**np.average(np.log2(mults))
        self.td_frame_top = avg_td_mult * self.td_max

    def make_save_dir(self):
        os.mkdir('saves/figures/sections/section_' + str(self.section_num) + '/group_' + str(self.group_num))

        #interpolate the section-length parameter trajectories for each phrase

        #really ugly code. Concatinate when you get some free time!
    def phrase_traj_interp(self):
        # for each phrase, assess register width and register center at midpoint
        rws = []
        rcs = []
        tdws = []
        tdcs = []
        nws = []
        ncs = []
        csws = []
        cscs = []
        vws = []
        vcs = []

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
            # td width
            tw = lin_interp(mp_x, self.td_width[0], self.td_width[1])
            # td center
            tc = lin_interp(mp_x, self.td_center[0], self.td_center[1])
            # nCVI width
            nw = lin_interp(mp_x, self.nCVI_width[0], self.nCVI_width[1])
            # nCVI center
            nc = lin_interp(mp_x, self.nCVI_center[0], self.nCVI_center[1])
            # chord_size width
            csw = lin_interp(mp_x, self.cs_width[0], self.cs_width[1])
            # chord size center
            csc = lin_interp(mp_x, self.cs_center[0], self.cs_center[1])
            # vel width
            vw = lin_interp(mp_x, self.vel_width[0], self.vel_width[1])
            # vel center
            vc = lin_interp(mp_x, self.vel_center[0], self.vel_center[1])
            rws.append(rw)
            rcs.append(rc)
            tdws.append(tw)
            tdcs.append(tc)
            nws.append(nw)
            ncs.append(nc)
            csws.append(csw)
            cscs.append(csc)
            vws.append(vw)
            vcs.append(vc)
        self.phrase_rws = rws
        self.phrase_rcs = rcs
        self.phrase_tdws = tdws
        self.phrase_tdcs = tdcs
        self.phrase_nws = nws
        self.phrase_ncs = ncs
        self.phrase_csws = csws
        self.phrase_cscs = cscs
        self.phrase_vws = vws
        self.phrase_vcs = vcs

    def make_phrases(self):
        phrases = []
        ins = self.instruments
        fgr = self.full_group_range
        fpr = self.full_piece_range
        sst = self.start_time
        tdft = self.td_frame_top
        tdo = self.td_octaves
        rnm = self.rhythm_nCVI_max
        ps = self.pitch_set
        w = self.weights
        wt = self.weight_traj
        sd = self.duration
        vmax = self.vel_max
        vmin = self.vel_min
        for i in range(self.nop):
            pst = self.phrase_bounds[i][0] + sst
            dur = self.phrase_bounds[i][1]
            rw = self.phrase_rws[i]
            rc = self.phrase_rcs[i]
            tdw = self.phrase_tdws[i]
            tdc = self.phrase_tdcs[i]
            nw = self.phrase_nws[i]
            nc = self.phrase_ncs[i]
            csw = self.phrase_csws[i]
            csc = self.phrase_cscs[i]
            vw = self.phrase_vws[i]
            vc = self.phrase_vcs[i]
            phrase = Phrase(ins, sst, pst, dur, fgr, rw, rc, fpr, tdft, tdo,   \
                tdw, tdc, nw, nc, rnm, ps, w, wt, sd, csw, csc, vw, vc, vmax,  \
                vmin)
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
        if self.stitch == 3:
            num_of_plays = num_of_rests - 1
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
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.tight_layout()
        plt.savefig(f"saves/figures/sections/section_{self.section_num}/group_{self.group_num}/phrase_bounds.png")
        plt.close()

    def plot_group_ranges(self):
        fig = plt.figure(figsize=[8, 1.0 + 0.5 * len(self.instruments)])
        ax = fig.add_subplot(111)
        for i, inst in enumerate(self.instruments[::-1]):
            plt.plot((inst.range), [i + 0.5 for x in inst.range], marker='|',  \
                color=(inst.color), label=(inst.name))
            plt.annotate((inst.min), (inst.mp_min, 0.75 + i), ha='center')
            plt.annotate((inst.max), (inst.mp_max, 0.75 + i), ha='center')
        plt.xlim(24, 84)
        plt.ylim(0, len(self.instruments) + 1)
        plt.xticks([12*(2+j) for j in range(7)],['C'+str(j+2) for j in range(7)])
        plt.yticks([])
        plt.title('Section ' + str(self.section_num) + ' Group ' + str(self.group_num))
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles[::-1], labels[::-1])
        plt.tight_layout()
        path = 'saves/figures/sections/'
        plt.savefig(path+f"section_{self.section_num}/group_{self.group_num}/ranges.png")
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
        self, chord, instruments, partition, global_chord_weights, section_num,\
        duration, rest_ratio, rest_dur_nCVI, rest_spread_nCVI, rtemp_density,  \
        nos, start_time, reg_widths, reg_centers, full_piece_range, td_max,    \
        td_octaves, td_widths, td_centers, nCVI_widths, nCVI_centers,          \
        rhythm_nCVI_max, cs_widths, cs_centers, vel_widths, vel_centers,       \
        vel_max, vel_min
        ):
        # number of groups
        self.nog = len(self.partition)
        self.get_section_chord()
        self.get_weights()
        self.get_pitch_sets()
        self.make_groups()

    def __continue__(self, stitches):
        self.stitches = stitches
        self.set_rparam_spread_for_groups()
        os.mkdir('saves/figures/sections/section_' + str(self.section_num))
        self.instantiate_groups()
        self.plot_section_phrase_bounds()
        self.plot_section_phrase_ranges()
        self.plot_section_td()
        self.plot_section_td_and_range()
        self.plot_section_nCVI_and_range()
        self.plot_section_nCVIs()
        self.progress()

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
        plt.ylim(24, 96)
        plt.yticks(12 * (1 + np.arange(7)), ['C0','C1','C2','C3', 'C4', 'C5', 'C6'])
        plt.xlim(self.start_time, self.start_time + self.duration)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.tight_layout()
        plt.savefig(f'saves/figures/sections/section_{str(self.section_num)}/phrase_ranges.png')
        plt.savefig(f'saves/figures/ranges/phrase_range_{str(self.section_num)}.png')
        plt.close()

    def plot_section_td(self):
        fig = plt.figure(figsize = [10,4])
        ax = fig.add_subplot(111)
        for group in self.groups:
            # print('prase td bounds: '+str([phrase.td_bounds for phrase in group.phrases]))
            mins = [phrase.td_bounds[0] for phrase in group.phrases]
            extents = [phrase.td_bounds[1] - phrase.td_bounds[0] for phrase in group.phrases]
            combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
            # for lines
            pb = np.array(group.phrase_bounds)
            pb[:, 0] = pb[:, 0] + self.start_time
            for p_index in range(group.nop):
                for inst in group.instruments:
                    plt.plot([[sum(pb[p_index][:i+1])] for i in range(2)], [group.phrases[p_index].td for i in range(2)], \
                    color=inst.color, alpha = 1, linewidth=2 )
        plt.ylim(((1/golden)**2) * self.td_max / (2**self.td_octaves), self.td_max)
        plt.xlim(self.start_time, self.start_time + self.duration)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.yscale('log', basey=2)
        plt.tight_layout()
        plt.savefig(f'saves/figures/sections/section_{str(self.section_num)}/phrase_tds.png')
        plt.savefig(f'saves/figures/temporal_densities/td_{str(self.section_num)}.png')
        plt.close()

    def plot_section_td_and_range(self):
        fig = plt.figure(figsize = [10,4])
        ax = fig.add_subplot(111)
        for group in self.groups:
            # print('prase td bounds: '+str([phrase.td_bounds for phrase in group.phrases]))
            mins = [phrase.td_bounds[0] for phrase in group.phrases]
            extents = [phrase.td_bounds[1] - phrase.td_bounds[0] for phrase in group.phrases]
            combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
            # for lines
            pb = np.array(group.phrase_bounds)
            pb[:, 0] = pb[:, 0] + self.start_time
            for p_index in range(group.nop):
                for inst in group.instruments:
                    ax.broken_barh([pb[p_index]], combs[p_index], \
                    color=(inst.color), alpha=(1/3) / len(group.instruments))
                    #plot actual td
                    plt.plot([[sum(pb[p_index][:i+1])] for i in range(2)], [group.phrases[p_index].td for i in range(2)], \
                    color=inst.color, alpha=1, linewidth=2 )
        plt.ylim(((1/golden)**2) * self.td_max / (2**self.td_octaves), self.td_max)
        plt.xlim(self.start_time, self.start_time + self.duration)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.yscale('log', basey=2)
        plt.tight_layout()
        plt.savefig(f'saves/figures/sections/section_{str(self.section_num)}/phrase_td_and_ranges.png')
        plt.savefig(f'saves/figures/temporal_densities/td_{str(self.section_num)}_and_range.png')
        plt.close()

    def plot_section_nCVI_and_range(self):
        fig = plt.figure(figsize = [10,4])
        ax = fig.add_subplot(111)
        for group in self.groups:
            # print('prase td bounds: '+str([phrase.td_bounds for phrase in group.phrases]))
            mins = [phrase.nCVI_bounds[0] for phrase in group.phrases]
            extents = [phrase.nCVI_bounds[1] - phrase.nCVI_bounds[0] for phrase in group.phrases]
            combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
            # for lines
            pb = np.array(group.phrase_bounds)
            pb[:, 0] = pb[:, 0] + self.start_time
            for p_index in range(group.nop):
                for inst in group.instruments:
                    ax.broken_barh([pb[p_index]], combs[p_index], \
                    color=(inst.color), alpha=(1/3) / len(group.instruments))
                    #plot actual nCVI
                    # this should be refactored somehow ... really confusing
                    plt.plot([[sum(pb[p_index][:i+1])] for i in range(2)], [group.phrases[p_index].nCVI for i in range(2)], \
                    color=inst.color, alpha=1, linewidth=2 )
        plt.ylim(0, self.rhythm_nCVI_max)
        plt.xlim(self.start_time, self.start_time + self.duration)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.tight_layout()
        plt.savefig(f'saves/figures/sections/section_{str(self.section_num)}/phrase_nCVI_and_ranges.png')
        plt.savefig(f'saves/figures/nCVIs/nCVI_{str(self.section_num)}_and_range.png')
        plt.close()

    def plot_section_nCVIs(self):
        fig = plt.figure(figsize = [10,4])
        ax = fig.add_subplot(111)
        for group in self.groups:
            # print('prase td bounds: '+str([phrase.td_bounds for phrase in group.phrases]))
            mins = [phrase.nCVI_bounds[0] for phrase in group.phrases]
            extents = [phrase.nCVI_bounds[1] - phrase.nCVI_bounds[0] for phrase in group.phrases]
            combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
            # for lines
            pb = np.array(group.phrase_bounds)
            pb[:, 0] = pb[:, 0] + self.start_time
            for p_index in range(group.nop):
                for inst in group.instruments:
                    # ax.broken_barh([pb[p_index]], combs[p_index], \
                    # color=(inst.color), alpha=(1/3) / len(group.instruments))
                    #plot actual nCVI
                    # this should be refactored somehow ... really confusing
                    plt.plot([[sum(pb[p_index][:i+1])] for i in range(2)], [group.phrases[p_index].nCVI for i in range(2)], \
                    color=inst.color, alpha=1, linewidth=2 )
        plt.ylim(0, self.rhythm_nCVI_max)
        plt.xlim(self.start_time, self.start_time + self.duration)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(self.start_time, self.start_time + self.duration)
        plt.tight_layout()
        plt.savefig(f'saves/figures/sections/section_{str(self.section_num)}/phrase_nCVI.png')
        plt.savefig(f'saves/figures/nCVIs/nCVI_{str(self.section_num)}.png')
        plt.close()

    def progress(self):
        print_progress_bar((self.section_num), (self.nos), prefix='Progress:', suffix='Complete', length=50)

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
        xlocs = plt.xticks()[0]
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
        tm = self.td_max
        to = self.td_octaves
        rnm = self.rhythm_nCVI_max
        vmax = self.vel_max
        vmin = self.vel_min
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
            tw = self.td_widths[i]
            tc = self.td_centers[i]
            nw = self.nCVI_widths[i]
            nc = self.nCVI_centers[i]
            csw = self.cs_widths[i]
            csc = self.cs_centers[i]
            sti = self.stitches[i]
            vw = self.vel_widths[i]
            vc = self.vel_centers[i]
            g = Group(
                gp, ps, sw[psi], i+1, rrs, rds, rss, rts, sn, st, dur, p, rw,  \
                rc, fpr, tm, to, tw, tc, nw, nc, rnm, csw, csc, sti, vw, vc,   \
                vmax, vmin)
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
            s = [[spread(param, max_ratio) for k in range(j)] for j in rp_partition[i]]
            s = [j for j in itertools.chain.from_iterable(s)]
            spread_.append(s)
        self.rr_spread, self.rdnCVI_spread, self.rsnCVI_spread, self.rtd_spread = spread_

    def get_section_chord(self):
        size = np.random.choice((np.arange(len(self.chord) - 2) + 3), p=(jdm(len(self.chord) - 2)))
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
        dist = jdm(len(self.section_chord)-1)
        pitch_sets = []
        indexes = []
        for group in range(len(self.partition)):
            pcs_size = np.random.choice((np.arange(len(self.section_chord)-1) + 2), p=dist)
            ps_index = np.random.choice((np.arange(len(self.section_chord))), p=(self.section_weights), size=pcs_size, replace=False)
            ps = self.section_chord[ps_index]
            pitch_sets.append(ps)
            indexes.append(ps_index)
        self.pitch_sets = pitch_sets
        self.ps_indexes = indexes

class Piece:
    """Object which generates all sections, delegates top level params, etc"""
    @auto_args
    def __init__(
        self, dur_tot, chord, instruments, nos, section_dur_nCVI, \
        rhythm_nCVI_max, td_max, td_octaves, vel_max, rr_max, rdur_nCVI_max, \
        rspread_nCVI_max, rtemp_density_min, rr_min, vel_min, ratios):
        self.set_weights()
        self.section_durs = icsd(nos, section_dur_nCVI) * self.dur_tot
        self.set_midpoints()
        self.set_partitions()
        self.partitions = dc_alg(list(get_partition(len(instruments))), nos)
        self.rest_delegation()
        self.init_dirs()
        self.init_progressBar()
        self.reg_widths, self.reg_centers = self.delegation()
        self.td_widths, self.td_centers = self.delegation()
        self.nCVI_widths, self.nCVI_centers = self.delegation()
        self.cs_widths, self.cs_centers = self.delegation()
        self.vel_widths, self.vel_centers = self.delegation()
        self.set_full_piece_range()
        self.make_sections()
        self.make_stitches()
        for s_i, section in enumerate(self.sections):
            section.__continue__(self.stitches[s_i])
        self.print_rest_params()
        self.print_pitch_params()
        self.plot_piece_phrase_bounds()
        self.plot_piece_phrase_ranges()
        self.plot_piece_td_and_range()
        self.plot_piece_nCVI_and_range()
        self.plot_piece_td()
        # not using midi anymore, but might be useful to have around, just in case!
        self.print_midi()
        # new representation, better for SC playback, with rest or midi note + dur + vel
        self.create_event_dur_score()
        self.save_event_dur_score()

    def create_event_dur_score(self):
        """new representation, better for SC playback, with rest or midi note + dur + vel"""
        for inst in self.instruments:
            #[rest/midipitch, dur, vel]
            inst_score=[]
            running_clock = 0
            for n, note in enumerate(inst.notes):
                freq = mp_to_adjusted_freq(note[0], self.ratios)
                if type(freq) != int: freq = np.asscalar(freq)
                if type(note[0]) != int: inst.notes[n][0] = np.asscalar(note[0])
                if type(note[1]) != int: inst.notes[n][1] = np.asscalar(note[1])
                if type(note[2]) != int: inst.notes[n][2] = np.asscalar(note[2])
                # if type(note[3]) != int: inst.notes[n][3] = np.asscalar(note[3])
                if note[1] != running_clock:
                    inst_score.append(['Rest()', note[1] - running_clock, 0])
                inst_score.append([freq, note[2], note[3]])
                running_clock = note[1] + note[2]
            inst.event_dur_score = inst_score

    def make_stitches(self):
        self.stitches = [[0 for i in range(len(section.grouping))] for section in self.sections]
        for s_i in range(len(self.sections)-1):
            a = self.sections[s_i].grouping
            b = self.sections[s_i+1].grouping
            stitch = [i for i in a if i in b]
            for st in stitch:
                if self.stitches[s_i][a.index(st)] == 1:
                    self.stitches[s_i][a.index(st)] = 3
                else:
                    self.stitches[s_i][a.index(st)] = 2
                self.stitches[s_i+1][b.index(st)] = 1

    def save_event_dur_score(self):
        path = 'saves/json/'
        for inst in self.instruments:
            json_string = json.dumps(inst.event_dur_score)
            f = open(path+'inst_'+str(inst.instnum-1), "w")
            f.write(json_string)
            f.close()

    def print_midi(self):
        path = 'saves/midi/'
        for inst in self.instruments:
            name = fill_space(inst.name) + '.mid'
            easy_midi_generator(inst.notes, path + name, inst.midi_name)

    def plot_piece_phrase_ranges(self):
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
        plt.ylim(24, 96)
        plt.yticks(12 * (1 + np.arange(7)), ['C0','C1','C2','C3', 'C4', 'C5', 'C6'])
        plt.xlim(0, self.dur_tot)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(0, self.dur_tot)
        plt.tight_layout()
        plt.savefig(f'saves/figures/ranges/phrase_ranges.png')
        plt.close()

    def plot_piece_td_and_range(self):
        fig = plt.figure(figsize = [144/11, 90/11], dpi = 220)
        ax = fig.add_subplot(111)
        for section in self.sections:
            for group in section.groups:
                # print('prase td bounds: '+str([phrase.td_bounds for phrase in group.phrases]))
                mins = [phrase.td_bounds[0] for phrase in group.phrases]
                extents = [phrase.td_bounds[1] - phrase.td_bounds[0] for phrase in group.phrases]
                combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
                # for lines
                pb = np.array(group.phrase_bounds)
                pb[:, 0] = pb[:, 0] + section.start_time
                for p_index in range(group.nop):
                    for inst in group.instruments:
                        ax.broken_barh([pb[p_index]], combs[p_index], \
                        color=(inst.color), alpha = (1/3) / len(group.instruments))
                        #plot actual td
                        plt.plot([[sum(pb[p_index][:i+1])] for i in range(2)], [group.phrases[p_index].td for i in range(2)], \
                        color=inst.color, alpha = 1, linewidth=2 )
        plt.ylim(((1/golden)**2) * self.td_max / (2**self.td_octaves), self.td_max)
        plt.xlim(0, self.dur_tot)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(0, self.dur_tot)
        plt.yscale('log', basey=2)
        ylocs = plt.yticks()[0]
        plt.yticks(ylocs, [str(i) for i in ylocs])
        plt.ylim(((1/golden)**2) * self.td_max / (2**self.td_octaves), self.td_max)
        plt.tight_layout()
        plt.savefig(f'saves/figures/temporal_densities/piece_td_and_range.png')
        plt.close()

    def plot_piece_nCVI_and_range(self):
        fig = plt.figure(figsize = [144/11, 90/11], dpi = 220)
        ax = fig.add_subplot(111)
        for section in self.sections:
            for group in section.groups:
                mins = [phrase.nCVI_bounds[0] for phrase in group.phrases]
                extents = [phrase.nCVI_bounds[1] - phrase.nCVI_bounds[0] for phrase in group.phrases]
                combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
                # for lines
                pb = np.array(group.phrase_bounds)
                pb[:, 0] = pb[:, 0] + section.start_time
                for p_index in range(group.nop):
                    for inst in group.instruments:
                        ax.broken_barh([pb[p_index]], combs[p_index], \
                        color=(inst.color), alpha = (1/3) / len(group.instruments))
                        #plot actual td
                        plt.plot([[sum(pb[p_index][:i+1])] for i in range(2)], [group.phrases[p_index].nCVI for i in range(2)], \
                        color=inst.color, alpha = 1, linewidth=2 )
        plt.ylim(0, self.rhythm_nCVI_max)
        plt.xlim(0, self.dur_tot)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(0, self.dur_tot)
        ylocs = plt.yticks()[0]
        plt.ylim(0, self.rhythm_nCVI_max)
        plt.yticks(ylocs, [str(i) for i in ylocs])
        plt.tight_layout()
        plt.savefig(f'saves/figures/temporal_densities/piece_nCVI_and_range.png')
        plt.close()

    def plot_piece_td(self):
        fig = plt.figure(figsize = [144/11, 90/11], dpi = 220)
        ax = fig.add_subplot(111)
        for section in self.sections:
            for group in section.groups:
                # print('prase td bounds: '+str([phrase.td_bounds for phrase in group.phrases]))
                mins = [phrase.td_bounds[0] for phrase in group.phrases]
                extents = [phrase.td_bounds[1] - phrase.td_bounds[0] for phrase in group.phrases]
                combs = np.array([[mins[i], extents[i]] for i in range(group.nop)])
                # for lines
                pb = np.array(group.phrase_bounds)
                pb[:, 0] = pb[:, 0] + section.start_time
                for p_index in range(group.nop):
                    for inst in group.instruments:
                        plt.plot([[sum(pb[p_index][:i+1])] for i in range(2)], [group.phrases[p_index].td for i in range(2)], \
                        color=inst.color, alpha = 1, linewidth=2)
        plt.ylim(((1/golden)**2) * self.td_max / (2**self.td_octaves), self.td_max)
        plt.xlim(0, self.dur_tot)
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(0, self.dur_tot)
        plt.yscale('log', basey=2)
        ylocs = plt.yticks()[0]
        plt.yticks(ylocs, [str(i) for i in ylocs])
        plt.ylim(((1/golden)**2) * self.td_max / (2**self.td_octaves), self.td_max)
        plt.tight_layout()
        plt.savefig(f'saves/figures/temporal_densities/piece_td.png')
        plt.close()

    def set_full_piece_range(self):
        min_ = min([inst.mp_min for inst in self.instruments])
        max_ = max([inst.mp_max for inst in self.instruments])
        self.full_piece_range = range(min_, max_)

    def init_progressBar(self):
        nos = self.nos
        print_progress_bar(0, nos, prefix='Progress:', suffix='Complete', length=50)

    def make_sections(self):
        sections = []
        c = self.chord
        ins = self.instruments
        gcw = self.global_chord_weights
        nos = self.nos
        sd = self.section_durs
        fpr = self.full_piece_range
        start_times = [sum(sd[:i]) for i in range(nos)]
        tdm = self.td_max
        tdo = self.td_octaves
        rnm = self.rhythm_nCVI_max
        vmax = self.vel_max
        vmin = self.vel_min
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
            nw = self.nCVI_widths[i]
            nc = self.nCVI_centers[i]
            csw = self.cs_widths[i]
            csc = self.cs_centers[i]
            vw = self.vel_widths[i]
            vc = self.vel_centers[i]
            sec = Section(
                    c, ins, p, gcw, i+1, sdi, rr, rdn, rsn, rtd, nos, st, rw, \
                    rc, fpr, tdm, tdo, tw, tc, nw, nc, rnm, csw, csc, vw, vc, \
                    vmax, vmin)
            sections.append(sec)
        self.sections = sections

    def init_dirs(self):
        dirs = ['sections', 'phrases', 'ranges', 'temporal_densities']
        paths = ['saves/figures/' + dir for dir in dirs]
        for i in range(len(dirs)):
            if os.path.exists(paths[i]):
                shutil.rmtree(paths[i])
            os.mkdir(paths[i])

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
        rest_ratio = generalized_delegator(loc, get_rr, self.rr_max, self.midpoints, self.rr_min)
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

    def set_weights(self, standard_dist=0.25):
        weights = np.random.normal(0.5, standard_dist, len(self.chord))
        while np.all((weights == np.abs(weights)), axis=0) == False:
                weights = np.random.normal(0.5, standard_dist, len(self.chord))
        self.global_chord_weights = weights / np.sum(weights)
        # print (self.global_chord_weights)

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
                rmin = mp_to_nn(min(group.full_group_range))
                rmax = mp_to_nn(max(group.full_group_range))
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
        xlocs = plt.xticks()[0]
        plt.xticks(xlocs, [secs_to_mins(i) for i in xlocs])
        plt.xlim(0, self.dur_tot)
        plt.tight_layout()
        plt.savefig('saves/figures/phrases/piece.png')
