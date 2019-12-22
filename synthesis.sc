(
s.boot;
~notes = Array.fill(16, {arg i;
	f = File.open("/Users/Jon/Documents/2019/partition/saves/json/inst_" ++ i.asString, "r");
	z = f.readAllString.compile.value;
	f.close;
	z;
	});

~midi_pitches = Array.fill(size(~notes), {arg i; Array.fill(~notes[i].size, {arg j;
	if (~notes[i][j][0].class == String, {Rest()}, {~notes[i][j][0]});
})});
~durs = Array.fill(size(~notes), {arg i; Array.fill(~notes[i].size, {arg j; ~notes[i][j][1]})});
~vels = Array.fill(size(~notes), {arg i; Array.fill(~notes[i].size, {arg j; ~notes[i][j][2]})});

~makeEnvSpec = {arg maxRamps=5;
	var attack, decay, points, dur, durs, curves;
	attack = (2**(4.0.rand)) / (2**6);
	decay = (2**(4.0.rand)) / (2**5);
	points = (maxRamps-1).rand+2;
	points = Array.fill(points, {0.5.rand+0.5});
	points = points / points.maxItem();
	points = [0] ++ points ++ [0];
	curves = Array.fill(size(points)-1, {8.0.rand - 4});
	[attack, decay, points, curves]
};
~envSpecs = Array.fill(size(~notes), {~makeEnvSpec.value()});
~amps = Array.fill(size(~notes), {Array.fill(4, {1.0.rand}).normalizeSum});

~makeEnv = {arg durTot, envSpec;
	var attack, decay, dur, durs, points, curves;
	attack = envSpec[0];
	decay = envSpec[1];
	points = envSpec[2];
	curves = envSpec[3];
	dur = durTot - attack - decay;
	durs = Array.fill(size(curves)-2, {1.0.rand});
	durs = dur * durs / sum(durs);
	durs = [attack] ++ durs ++ [decay];
	Env(points, durs, curves);
};

//Make the SynthDefs
size(~notes).do({arg index;
	SynthDef(("\inst_"++index.asString).asSymbol, { |out, freq = 440, sustain = 1, amp|
		var sig, env, amps = ~amps[index], envSpec = ~envSpecs[index];
		freq = freq * (1 + (0.01.rand - 0.005));
		env = ~makeEnv.value(sustain, envSpec);
		sig = SinOsc.ar(freq, 0, amps[0]);
		sig = sig + LFTri.ar(freq, 0, amps[1]);
		sig = sig + LFSaw.ar(freq, 0, amps[2]);
		sig = sig + LFPulse.ar(freq, 0, amps[3]);
		sig = sig * amp * EnvGen.kr(env, doneAction: Done.freeSelf);
		sig = BLowPass.ar(sig,3000,0.5);
		sig = BHiPass.ar(sig, 100);
		Out.ar(out, sig ! 2)
	}).add;
});

~synthNames = Array.fill(size(~notes), {arg i; ("\inst_"++i.asString).asSymbol});

(
~pbinds = Array.fill(~notes.size, {arg i;
	Pbind(
		\instrument, ~synthNames[i],
		\midinote, Pseq(~midi_pitches[i]),
		\dur, Pseq(~durs[i]),
	)
});
);

Ppar(~pbinds).play;
);

