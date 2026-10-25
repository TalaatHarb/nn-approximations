package net.talaatharb.nn.examples;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Random;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.metrics.MeanSquareError;

class ExampleProblemTest {

	@ParameterizedTest
	@EnumSource(ExampleProblem.class)
	void everyExampleLearnsItsFunction(ExampleProblem example) {
		final var network = example.newNetwork();
		final var data = example.data();
		final var loss = new MeanSquareError();
		float result = new StochasticGradientDescent(example.learningRate(), new Random(42))
				.train(network, data, example.threshold(), 5000, loss);
		assertTrue(result <= example.threshold(), example + " failed to converge: " + result);
		double actual = 0;
		for (int i = 0; i < data.getInput().length; i++) {
			float[] prediction = network.apply(data.getInput()[i]);
			actual += loss.errorMetric(data.getOutput()[i], prediction);
			if (example.isLogic()) {
				assertEquals((int) data.getOutput()[i][0], prediction[0] >= 0.5f ? 1 : 0);
			}
		}
		assertEquals(result, actual / data.getInput().length, 1e-7);
		if (!example.isLogic()) {
			double unseenError = 0;
			for (int i = 0; i < 40; i++) {
				float x = -0.975f + 0.05f * i;
				float error = network.apply(new float[] { x })[0] - example.target(x);
				unseenError += error * error;
			}
			assertTrue(unseenError / 40 < example.threshold() * 1.5, "Poor interpolation for " + example);
		}
	}
}
