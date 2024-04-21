package net.talaatharb.nn.functions;

import java.util.function.UnaryOperator;

public interface ActivationFunction extends UnaryOperator<Float> {

	// https://dustinstansbury.github.io/theclevermachine/derivation-common-neural-network-activation-functions
	float numericalDervative(float input, float output);
}
