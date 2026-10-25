package net.talaatharb.nn.visualization;

public interface InspectableNetwork {

	/** Runs inference and returns detached state, including hidden activations. */
	NetworkSnapshot inspect(float[] input);
}
