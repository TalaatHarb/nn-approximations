package net.talaatharb.nn.algorithms;

import java.util.List;

import net.talaatharb.nn.visualization.NetworkSnapshot;

/** One SGD update; before/after states use the same input and their own weights. */
public record TrainingUpdate(int epoch, long step, int sampleIndex, float lossBefore, float lossAfter,
		List<Float> expected, NetworkSnapshot before, NetworkSnapshot after) {

	public TrainingUpdate {
		expected = List.copyOf(expected);
	}
}
