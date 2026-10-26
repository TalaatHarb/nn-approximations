package net.talaatharb.nn.ui;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;

import javafx.beans.property.ReadOnlyStringProperty;
import javafx.beans.property.ReadOnlyStringWrapper;
import javafx.geometry.VPos;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;
import net.talaatharb.nn.visualization.NetworkSnapshot;

/**
 * Draws a {@link NetworkSnapshot} on a canvas. Edge width/opacity are normalized to the largest
 * magnitude shown so small networks and large ones remain readable. Hovering a neuron highlights
 * its connections and publishes a textual breakdown through {@link #detailProperty()}.
 */
public class NetworkCanvas extends Region {

	public enum EdgeMode {
		WEIGHTS("Weights"), CHANGES("Last update change");

		private final String label;

		EdgeMode(String label) {
			this.label = label;
		}

		@Override
		public String toString() {
			return label;
		}
	}

	private static final Color POSITIVE = Color.rgb(33, 102, 172);
	private static final Color NEGATIVE = Color.rgb(198, 40, 40);
	private static final double MARGIN_X = 70;
	private static final double MARGIN_TOP = 44;
	private static final double MARGIN_BOTTOM = 22;

	private final Canvas canvas = new Canvas();
	private final ReadOnlyStringWrapper detail = new ReadOnlyStringWrapper(this, "detail", "");
	private NetworkSnapshot current;
	private NetworkSnapshot previous;
	private List<String> captions = List.of();
	private List<String> inputNames = List.of();
	private int revealLayer = Integer.MAX_VALUE;
	private EdgeMode mode = EdgeMode.WEIGHTS;
	private int hoverLayer = -1;
	private int hoverUnit = -1;

	public NetworkCanvas() {
		getChildren().add(canvas);
		setMinSize(260, 200);
		setPrefSize(640, 460);
		canvas.setOnMouseMoved(event -> hover(event.getX(), event.getY()));
		canvas.setOnMouseExited(event -> {
			hoverLayer = -1;
			hoverUnit = -1;
			detail.set("");
			draw();
		});
	}

	public ReadOnlyStringProperty detailProperty() {
		return detail.getReadOnlyProperty();
	}

	/**
	 * @param snapshot   state to draw
	 * @param before     state before the last update (enables the change view), or null
	 * @param reveal     last layer index whose values are shown; later layers are greyed out
	 * @param captions   one caption per layer (falls back to snapshot labels)
	 * @param inputNames names for input units
	 */
	public void show(NetworkSnapshot snapshot, NetworkSnapshot before, int reveal, List<String> captions,
			List<String> inputNames) {
		this.current = snapshot;
		this.previous = before != null && before.connections().size() == snapshot.connections().size() ? before
				: null;
		this.revealLayer = reveal;
		this.captions = captions == null ? List.of() : List.copyOf(captions);
		this.inputNames = inputNames == null ? List.of() : List.copyOf(inputNames);
		if (hoverLayer >= snapshot.layers().size()
				|| hoverLayer >= 0 && hoverUnit >= snapshot.layers().get(hoverLayer).units().size()) {
			hoverLayer = -1;
			hoverUnit = -1;
		}
		draw();
		updateDetail();
	}

	public void setMode(EdgeMode mode) {
		this.mode = mode;
		draw();
	}

	public EdgeMode mode() {
		return mode;
	}

	public boolean showsChanges() {
		return mode == EdgeMode.CHANGES && previous != null;
	}

	@Override
	protected void layoutChildren() {
		final double w = Math.max(1, getWidth());
		final double h = Math.max(1, getHeight());
		if (canvas.getWidth() != w || canvas.getHeight() != h) {
			canvas.setWidth(w);
			canvas.setHeight(h);
			draw();
		}
	}

	private double columnX(int layer) {
		final int count = current.layers().size();
		final double usable = canvas.getWidth() - 2 * MARGIN_X;
		return count == 1 ? canvas.getWidth() / 2 : MARGIN_X + usable * layer / (count - 1);
	}

	private double rowY(int layer, int unit) {
		final int count = current.layers().get(layer).units().size();
		final double usable = canvas.getHeight() - MARGIN_TOP - MARGIN_BOTTOM;
		return MARGIN_TOP + usable * (unit + 0.5) / count;
	}

	private double radius() {
		final int maxUnits = current.layers().stream().mapToInt(layer -> layer.units().size()).max().orElse(1);
		final double vertical = (canvas.getHeight() - MARGIN_TOP - MARGIN_BOTTOM) / maxUnits * 0.38;
		final double horizontal = (canvas.getWidth() - 2 * MARGIN_X) / Math.max(1, current.layers().size() - 1)
				* 0.16;
		return Math.max(2.5, Math.min(22, Math.min(vertical, horizontal)));
	}

	private float edgeValue(int index) {
		final float weight = current.connections().get(index).weight();
		return showsChanges() ? weight - previous.connections().get(index).weight() : weight;
	}

	private void draw() {
		final GraphicsContext g = canvas.getGraphicsContext2D();
		g.setFill(Color.WHITE);
		g.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
		if (current == null) {
			return;
		}
		final double r = radius();
		final var connections = current.connections();
		float max = 1e-12f;
		for (int i = 0; i < connections.size(); i++) {
			max = Math.max(max, Math.abs(edgeValue(i)));
		}
		final boolean hovering = hoverLayer >= 0;
		for (int pass = 0; pass < 2; pass++) {
			for (int i = 0; i < connections.size(); i++) {
				final var c = connections.get(i);
				final boolean touches = touchesHover(c);
				if (hovering && (pass == 0) == touches) {
					continue;
				}
				if (!hovering && pass == 1) {
					continue;
				}
				final float value = edgeValue(i);
				final double strength = Math.abs(value) / max;
				final boolean revealed = c.targetLayer() <= revealLayer && c.sourceLayer() <= revealLayer;
				double alpha = revealed ? 0.12 + 0.78 * strength : 0.06;
				if (hovering) {
					alpha = touches ? Math.max(alpha, 0.35) : 0.04;
				}
				g.setStroke((value >= 0 ? POSITIVE : NEGATIVE).deriveColor(0, 1, 1, alpha));
				g.setLineWidth(revealed ? 0.4 + 4.6 * strength : 0.5);
				g.strokeLine(columnX(c.sourceLayer()), rowY(c.sourceLayer(), c.sourceUnit()),
						columnX(c.targetLayer()), rowY(c.targetLayer(), c.targetUnit()));
			}
		}
		final float biasMax = maxBias();
		g.setTextAlign(TextAlignment.CENTER);
		g.setTextBaseline(VPos.CENTER);
		for (int l = 0; l < current.layers().size(); l++) {
			final var layer = current.layers().get(l);
			final boolean revealed = l <= revealLayer;
			g.setFill(Color.rgb(40, 40, 40));
			g.setFont(Font.font(12));
			g.fillText(caption(l), columnX(l), 16);
			g.setFill(Color.GRAY);
			g.setFont(Font.font(10.5));
			g.fillText(layer.units().size() + (layer.units().size() == 1 ? " unit" : " units"), columnX(l), 31);
			for (int u = 0; u < layer.units().size(); u++) {
				final var unit = layer.units().get(u);
				final double x = columnX(l);
				final double y = rowY(l, u);
				g.setFill(revealed ? activationColor(unit.activation()) : Color.gray(0.93));
				g.fillOval(x - r, y - r, 2 * r, 2 * r);
				final boolean hovered = l == hoverLayer && u == hoverUnit;
				if (l > 0 && revealed) {
					final double biasStrength = Math.abs(unit.bias()) / biasMax;
					g.setStroke((unit.bias() >= 0 ? POSITIVE : NEGATIVE).deriveColor(0, 1, 1, 0.3 + 0.7 * biasStrength));
					g.setLineWidth(hovered ? 3.5 : 1 + 2.5 * biasStrength);
				} else {
					g.setStroke(hovered ? Color.BLACK : Color.gray(0.45));
					g.setLineWidth(hovered ? 3 : 1);
				}
				g.strokeOval(x - r, y - r, 2 * r, 2 * r);
				if (hovered) {
					g.setStroke(Color.BLACK);
					g.setLineWidth(1.5);
					g.strokeOval(x - r - 3, y - r - 3, 2 * r + 6, 2 * r + 6);
				}
				if (revealed && r >= 11) {
					g.setFill(Math.abs(Math.tanh(unit.activation())) > 0.6 ? Color.WHITE : Color.BLACK);
					g.setFont(Font.font(Math.min(12, r * 0.62)));
					g.fillText(format(unit.activation()), x, y);
				}
				if (l == 0 && r >= 6 && u < inputNames.size()) {
					g.setFill(Color.rgb(40, 40, 40));
					g.setTextAlign(TextAlignment.RIGHT);
					g.setFont(Font.font(11));
					g.fillText(inputNames.get(u), x - r - 6, y);
					g.setTextAlign(TextAlignment.CENTER);
				}
			}
		}
		g.setTextAlign(TextAlignment.LEFT);
		g.setFill(Color.GRAY);
		g.setFont(Font.font(10.5));
		g.fillText(String.format(Locale.ROOT, "%s | max |%s| = %.4g", showsChanges() ? "Edges: Δweight of last update"
				: "Edges: weights", showsChanges() ? "Δw" : "w", max), 8, canvas.getHeight() - 10);
	}

	private float maxBias() {
		float max = 1e-12f;
		for (int l = 1; l < current.layers().size(); l++) {
			for (var unit : current.layers().get(l).units()) {
				max = Math.max(max, Math.abs(unit.bias()));
			}
		}
		return max;
	}

	private boolean touchesHover(NetworkSnapshot.Connection c) {
		return c.sourceLayer() == hoverLayer && c.sourceUnit() == hoverUnit
				|| c.targetLayer() == hoverLayer && c.targetUnit() == hoverUnit;
	}

	private String caption(int layer) {
		return layer < captions.size() ? captions.get(layer) : current.layers().get(layer).label();
	}

	static Color activationColor(float value) {
		final double t = Math.tanh(value);
		return t >= 0 ? Color.WHITE.interpolate(POSITIVE, t) : Color.WHITE.interpolate(NEGATIVE, -t);
	}

	private static String format(float value) {
		return Math.abs(value) >= 100 ? String.format(Locale.ROOT, "%.0f", value)
				: String.format(Locale.ROOT, "%.2f", value);
	}

	private void hover(double x, double y) {
		if (current == null) {
			return;
		}
		final double r = Math.max(radius(), 6);
		int layer = -1;
		int unit = -1;
		double best = Double.MAX_VALUE;
		for (int l = 0; l < current.layers().size(); l++) {
			if (Math.abs(columnX(l) - x) > r + 4) {
				continue;
			}
			for (int u = 0; u < current.layers().get(l).units().size(); u++) {
				final double distance = Math.hypot(columnX(l) - x, rowY(l, u) - y);
				if (distance <= r + 4 && distance < best) {
					best = distance;
					layer = l;
					unit = u;
				}
			}
		}
		if (layer != hoverLayer || unit != hoverUnit) {
			hoverLayer = layer;
			hoverUnit = unit;
			draw();
			updateDetail();
		}
	}

	private void updateDetail() {
		if (current == null || hoverLayer < 0) {
			detail.set("");
			return;
		}
		final var unit = current.layers().get(hoverLayer).units().get(hoverUnit);
		final var text = new StringBuilder();
		text.append(caption(hoverLayer)).append(", unit ").append(hoverUnit + 1).append(": ");
		if (hoverLayer == 0) {
			text.append(String.format(Locale.ROOT, "input %s = %.4f", hoverUnit < inputNames.size()
					? inputNames.get(hoverUnit) : "x" + (hoverUnit + 1), unit.activation()));
		} else {
			text.append(String.format(Locale.ROOT, "z = %.4f -> a = %.4f, bias %.4f", unit.preActivation(),
					unit.activation(), unit.bias()));
			if (previous != null) {
				final float biasBefore = previous.layers().get(hoverLayer).units().get(hoverUnit).bias();
				text.append(String.format(Locale.ROOT, " (Δ %+.5f)", unit.bias() - biasBefore));
			}
		}
		final var incoming = new ArrayList<Integer>();
		final var outgoing = new ArrayList<Integer>();
		for (int i = 0; i < current.connections().size(); i++) {
			final var c = current.connections().get(i);
			if (c.targetLayer() == hoverLayer && c.targetUnit() == hoverUnit) {
				incoming.add(i);
			} else if (c.sourceLayer() == hoverLayer && c.sourceUnit() == hoverUnit) {
				outgoing.add(i);
			}
		}
		appendStrongest(text, "\nStrongest incoming", incoming, true);
		appendStrongest(text, "\nStrongest outgoing", outgoing, false);
		detail.set(text.toString());
	}

	private void appendStrongest(StringBuilder text, String title, List<Integer> indices, boolean incoming) {
		if (indices.isEmpty()) {
			return;
		}
		indices.sort(Comparator.comparingDouble(i -> -Math.abs(current.connections().get(i).weight())));
		text.append(title).append(':');
		for (int i : indices.subList(0, Math.min(4, indices.size()))) {
			final var c = current.connections().get(i);
			final int otherLayer = incoming ? c.sourceLayer() : c.targetLayer();
			final int otherUnit = incoming ? c.sourceUnit() : c.targetUnit();
			final String name = otherLayer == 0 && otherUnit < inputNames.size() ? inputNames.get(otherUnit)
					: "L" + otherLayer + "#" + (otherUnit + 1);
			text.append(String.format(Locale.ROOT, "  %s w=%+.3f", name, c.weight()));
			if (previous != null) {
				text.append(String.format(Locale.ROOT, " (Δ%+.4f)", c.weight() - previous.connections().get(i).weight()));
			}
			if (incoming) {
				final float source = current.layers().get(c.sourceLayer()).units().get(c.sourceUnit()).activation();
				text.append(String.format(Locale.ROOT, " · %.3f", source));
			}
		}
	}
}
