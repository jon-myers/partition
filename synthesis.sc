
(
(
~notes = Array.fill(12, {arg i;
	f = File.open("/Users/Jon/Documents/2019/partition/saves/json/inst_" ++ i.asString, "r");
	z = f.readAllString.compile.value;
	f.close;
	z;
	});

~midi_pitches = Array.fill(12, {arg i; Array.fill(~notes[i].size, {arg j;
	if (~notes[i][j][0].class == String, {Rest()}, {~notes[i][j][0]});
})});
~durs = Array.fill(12, {arg i; Array.fill(~notes[i].size, {arg j; ~notes[i][j][1]})});
~vels = Array.fill(12, {arg i; Array.fill(~notes[i].size, {arg j; ~notes[i][j][2]})})
);

(
SynthDef(\smooth, { |out, freq = 440, sustain = 1, amp = 0.5|
    var sig;
    sig = LFTri.ar(freq, 0, amp) * EnvGen.kr(Env.linen(0.05, sustain, 0.1), doneAction: Done.freeSelf);
    Out.ar(out, sig ! 2)
}).add;
);

(
~pbinds = Array.fill(~notes.size, {arg i;
	Pbind(
		\instrument, \smooth,
		\midinote, Pseq(~midi_pitches[i]),
		\dur, Pseq(~durs[i])
	)
});
);

Ppar(~pbinds).play;
);

