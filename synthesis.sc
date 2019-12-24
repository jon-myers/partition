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

~makeAmpEnvSpec = {arg maxRamps=5;
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
~ampEnvSpecs = Array.fill(size(~notes), {~makeAmpEnvSpec.value()});
~amps = Array.fill(size(~notes), {Array.fill(4, {1.0.rand}).normalizeSum});

~makeAmpEnv = {arg durTot, envSpec;
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


~makeLpEnvSpec = {arg maxRamps=4;
	var points, durs, curves;
	points = (maxRamps-1).rand + 2;
	points = Array.fill(points, {400.0 * (2.0 ** (5.0.rand))});
	durs = Array.fill(size(points), {1.0.rand}).normalizeSum;
	curves = Array.fill(size(points)-1, {8.0.rand - 4});
	[points, durs, curves];
};


~makeLpEnv = {arg durTot, envSpec;
	var points = envSpec[0], durs = envSpec[1] * durTot, curves = envSpec[2];
	Env(points, durs, curves);
};

~lpFreqSpecs = Array.fill(size(~notes), {~makeLpEnvSpec.value()});

//Make the SynthDefs
size(~notes).do({arg index;
	SynthDef(("\inst_"++index.asString).asSymbol, { |out, freq = 440, sustain = 1, amp|
		var sig, ampEnv, lpFreq, lpFreqEnv, lpFreqSpec, amps, ampEnvSpec;
		lpFreqSpec = ~lpFreqSpecs[index];
		lpFreqEnv = ~makeLpEnv.value(sustain, lpFreqSpec);
		amps = ~amps[index];
		ampEnvSpec = ~ampEnvSpecs[index];
		freq = freq * (1 + (0.01.rand - 0.005));
		ampEnv = ~makeAmpEnv.value(sustain, ampEnvSpec);
		// lpFreq = 400 * (2**(5.0.rand));
		sig = SinOsc.ar(freq, 0, amps[0]);
		sig = sig + LFTri.ar(freq, 0, amps[1]);
		sig = sig + LFSaw.ar(freq, 0, amps[2]);
		sig = sig + LFPulse.ar(freq, 0, amps[3]);
		sig = sig * amp * EnvGen.kr(ampEnv, doneAction: Done.freeSelf);
		sig = BLowPass.ar(sig,EnvGen.kr(lpFreqEnv));
		sig = BHiPass.ar(sig, 100);
		sig = FreeVerb.ar(sig, 0.5, 0.75);
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




//to do:
//make sure filter volume reduction is somehow compensated for. ??
//tuning system
//dynes in python code (different every note?)
