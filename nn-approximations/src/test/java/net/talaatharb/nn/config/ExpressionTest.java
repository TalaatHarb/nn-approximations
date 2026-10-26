package net.talaatharb.nn.config;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import net.talaatharb.nn.functions.ActivationFunction;

class ExpressionTest {

	@ParameterizedTest
	@CsvSource(delimiter = ';', value = { "1 + 2 * 3; 7", "(1 + 2) * 3; 9", "2 ^ 3 ^ 2; 512", "-x^2; -4",
			"-2 * -x; 4", "sin(pi / 2); 1", "max(x, 3); 3", "xor(1, 0); 1", "xor(1, 1); 0", "and(x, y); 1",
			"nand(1, 0.7); 0", "not(0.2); 1", "7 % 4; 3", "abs(-x) + sqrt(16); 6", "atan2(1, 1) * 4; 3.141592653589793",
			"step(y - 0.5) + sign(-3); 0", "e; 2.718281828459045", "pow(x, 0.5) * 2E-1 * 10; 2.8284271247461903",
			"log10(1000) + ln(e); 4" })
	void evaluatesExpressions(String text, double expected) {
		assertEquals(expected, Expression.compile(text, List.of("x", "y")).evaluate(2, 1), 1e-12, text);
	}

	@ParameterizedTest
	@CsvSource(delimiter = ';', value = { "x +", "(x", "foo(x)", "z", "sin()", "1 2", "x $ 1" })
	void reportsErrorsWithPosition(String text) {
		final var exception = assertThrows(IllegalArgumentException.class,
				() -> Expression.compile(text, List.of("x")));
		assertTrue(exception.getMessage().contains("position"), text + " -> " + exception.getMessage());
	}

	@Test
	void reservedNamesAreKnown() {
		assertTrue(Expression.isReserved("sin"));
		assertTrue(Expression.isReserved("pi"));
		assertFalse(Expression.isReserved("x1"));
	}

	@Test
	void everyTrainableActivationHasFiniteDerivatives() {
		for (ActivationType type : ActivationType.values()) {
			final ActivationFunction function = ActivationSpec.of(type).create();
			for (float x : new float[] { -2, -1, 0, 1, 2 }) {
				final float y = function.apply(x);
				assertTrue(Float.isFinite(y), type + " f(" + x + ")");
				assertTrue(Float.isFinite(function.numericalDervative(x, y)), type + " f'(" + x + ")");
			}
			if (type.trainable()) {
				assertTrue(function.numericalDervative(0.25f, function.apply(0.25f)) != 0
						|| function.numericalDervative(-0.25f, function.apply(-0.25f)) != 0, type.id());
			}
		}
	}
}
