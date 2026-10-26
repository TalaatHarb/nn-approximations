package net.talaatharb.nn.config;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;
import java.util.regex.Pattern;

/**
 * Small, dependency-free arithmetic expression compiler for dataset targets.
 * Supports + - * / % ^, parentheses, variables, the constants pi and e, and the
 * functions listed by {@link #functionNames()}. Logic helpers treat values >= 0.5 as true.
 */
public final class Expression {

	public static final Pattern IDENTIFIER = Pattern.compile("[A-Za-z_][A-Za-z0-9_]*");

	private record Fn(int arity, ToDoubleFunction<double[]> body) {
	}

	private static boolean truth(double value) {
		return value >= 0.5;
	}

	private static double bool(boolean value) {
		return value ? 1 : 0;
	}

	private static Fn unary(DoubleUnaryOperator op) {
		return new Fn(1, a -> op.applyAsDouble(a[0]));
	}

	private static Fn binary(DoubleBinaryOperator op) {
		return new Fn(2, a -> op.applyAsDouble(a[0], a[1]));
	}

	private static final Map<String, Fn> FUNCTIONS = Map.ofEntries(
			Map.entry("sin", unary(Math::sin)), Map.entry("cos", unary(Math::cos)),
			Map.entry("tan", unary(Math::tan)), Map.entry("asin", unary(Math::asin)),
			Map.entry("acos", unary(Math::acos)), Map.entry("atan", unary(Math::atan)),
			Map.entry("sinh", unary(Math::sinh)), Map.entry("cosh", unary(Math::cosh)),
			Map.entry("tanh", unary(Math::tanh)), Map.entry("exp", unary(Math::exp)),
			Map.entry("log", unary(Math::log)), Map.entry("ln", unary(Math::log)),
			Map.entry("log10", unary(Math::log10)), Map.entry("sqrt", unary(Math::sqrt)),
			Map.entry("abs", unary(Math::abs)), Map.entry("floor", unary(Math::floor)),
			Map.entry("ceil", unary(Math::ceil)), Map.entry("round", unary(v -> (double) Math.round(v))),
			Map.entry("sign", unary(Math::signum)), Map.entry("sigmoid", unary(v -> 1 / (1 + Math.exp(-v)))),
			Map.entry("step", unary(v -> v >= 0 ? 1 : 0)), Map.entry("min", binary(Math::min)),
			Map.entry("max", binary(Math::max)), Map.entry("pow", binary(Math::pow)),
			Map.entry("atan2", binary(Math::atan2)), Map.entry("not", unary(v -> bool(!truth(v)))),
			Map.entry("and", binary((a, b) -> bool(truth(a) && truth(b)))),
			Map.entry("or", binary((a, b) -> bool(truth(a) || truth(b)))),
			Map.entry("xor", binary((a, b) -> bool(truth(a) != truth(b)))),
			Map.entry("nand", binary((a, b) -> bool(!(truth(a) && truth(b))))),
			Map.entry("nor", binary((a, b) -> bool(!(truth(a) || truth(b))))));

	private static final Map<String, Double> CONSTANTS = Map.of("pi", Math.PI, "e", Math.E);

	public static Set<String> functionNames() {
		return new TreeSet<>(FUNCTIONS.keySet());
	}

	/** Names that cannot be used as input variables. */
	public static boolean isReserved(String name) {
		return FUNCTIONS.containsKey(name) || CONSTANTS.containsKey(name);
	}

	@FunctionalInterface
	private interface Node {
		double eval(double[] variables);
	}

	private final String source;
	private final Node root;

	private Expression(String source, Node root) {
		this.source = source;
		this.root = root;
	}

	public static Expression compile(String text, List<String> variables) {
		if (text == null || text.isBlank()) {
			throw new IllegalArgumentException("Expression is empty");
		}
		final var parser = new Parser(text, variables);
		final var node = parser.expression();
		parser.skipSpaces();
		if (parser.position < text.length()) {
			throw parser.error("Unexpected '" + text.charAt(parser.position) + "'");
		}
		return new Expression(text, node);
	}

	public double evaluate(double... variables) {
		return root.eval(variables);
	}

	@Override
	public String toString() {
		return source;
	}

	private static final class Parser {
		private final String text;
		private final List<String> variables;
		private int position;

		Parser(String text, List<String> variables) {
			this.text = text;
			this.variables = variables;
		}

		IllegalArgumentException error(String message) {
			return new IllegalArgumentException(message + " at position " + (position + 1) + " in '" + text + "'");
		}

		void skipSpaces() {
			while (position < text.length() && Character.isWhitespace(text.charAt(position))) {
				position++;
			}
		}

		boolean accept(char expected) {
			skipSpaces();
			if (position < text.length() && text.charAt(position) == expected) {
				position++;
				return true;
			}
			return false;
		}

		Node expression() {
			Node left = term();
			while (true) {
				if (accept('+')) {
					final Node a = left, b = term();
					left = v -> a.eval(v) + b.eval(v);
				} else if (accept('-')) {
					final Node a = left, b = term();
					left = v -> a.eval(v) - b.eval(v);
				} else {
					return left;
				}
			}
		}

		Node term() {
			Node left = unary();
			while (true) {
				if (accept('*')) {
					final Node a = left, b = unary();
					left = v -> a.eval(v) * b.eval(v);
				} else if (accept('/')) {
					final Node a = left, b = unary();
					left = v -> a.eval(v) / b.eval(v);
				} else if (accept('%')) {
					final Node a = left, b = unary();
					left = v -> a.eval(v) % b.eval(v);
				} else {
					return left;
				}
			}
		}

		Node unary() {
			if (accept('-')) {
				final Node operand = unary();
				return v -> -operand.eval(v);
			}
			if (accept('+')) {
				return unary();
			}
			return power();
		}

		// Right-associative; binds tighter than unary minus, so -x^2 == -(x^2).
		Node power() {
			final Node base = primary();
			if (accept('^')) {
				final Node exponent = unary();
				return v -> Math.pow(base.eval(v), exponent.eval(v));
			}
			return base;
		}

		Node primary() {
			skipSpaces();
			if (position >= text.length()) {
				throw error("Unexpected end of expression");
			}
			if (accept('(')) {
				final Node inner = expression();
				if (!accept(')')) {
					throw error("Missing ')'");
				}
				return inner;
			}
			final char c = text.charAt(position);
			if (Character.isDigit(c) || c == '.') {
				return number();
			}
			if (Character.isLetter(c) || c == '_') {
				return identifier();
			}
			throw error("Unexpected '" + c + "'");
		}

		Node number() {
			final int start = position;
			while (position < text.length()
					&& (Character.isDigit(text.charAt(position)) || text.charAt(position) == '.')) {
				position++;
			}
			if (position < text.length() && (text.charAt(position) == 'e' || text.charAt(position) == 'E')) {
				int look = position + 1;
				if (look < text.length() && (text.charAt(look) == '+' || text.charAt(look) == '-')) {
					look++;
				}
				if (look < text.length() && Character.isDigit(text.charAt(look))) {
					position = look;
					while (position < text.length() && Character.isDigit(text.charAt(position))) {
						position++;
					}
				}
			}
			try {
				final double value = Double.parseDouble(text.substring(start, position));
				return v -> value;
			} catch (NumberFormatException exception) {
				position = start;
				throw error("Invalid number");
			}
		}

		Node identifier() {
			final int start = position;
			while (position < text.length()
					&& (Character.isLetterOrDigit(text.charAt(position)) || text.charAt(position) == '_')) {
				position++;
			}
			final String name = text.substring(start, position);
			if (accept('(')) {
				final Fn function = FUNCTIONS.get(name);
				if (function == null) {
					position = start;
					throw error("Unknown function '" + name + "'");
				}
				final List<Node> args = new ArrayList<>();
				if (!accept(')')) {
					do {
						args.add(expression());
					} while (accept(','));
					if (!accept(')')) {
						throw error("Missing ')' after arguments of " + name);
					}
				}
				if (args.size() != function.arity()) {
					position = start;
					throw error(name + " expects " + function.arity() + " argument(s), got " + args.size());
				}
				final Node[] nodes = args.toArray(Node[]::new);
				return v -> {
					final double[] values = new double[nodes.length];
					for (int i = 0; i < nodes.length; i++) {
						values[i] = nodes[i].eval(v);
					}
					return function.body().applyAsDouble(values);
				};
			}
			final int index = variables.indexOf(name);
			if (index >= 0) {
				return v -> v[index];
			}
			final Double constant = CONSTANTS.get(name);
			if (constant != null) {
				final double value = constant;
				return v -> value;
			}
			position = start;
			throw error("Unknown variable '" + name + "' (inputs: " + String.join(", ", variables) + ")");
		}
	}
}
