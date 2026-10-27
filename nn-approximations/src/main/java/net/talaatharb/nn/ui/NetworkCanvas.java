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
import javafx.scene.image.WritableImage;
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
	private static final double TILE_LEFT = 14;

	/**
	 * Draws the input layer as a width x height picture (channels 1 = grayscale, 3 = RGB interleaved per
	 * pixel) instead of one circle per pixel; {@code classes} label the output units.
	 */
	public record ImageLayout(int width, int height, int channels, List<String> classes) {
		public ImageLayout {
			classes = classes == null ? List.of() : List.copyOf(classes);
		}

		int pixels() {
			return width * height;
		}
	}

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
	private ImageLayout image;

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

	/** Switches the input layer to a picture (null restores one circle per input). */
	public void setImageLayout(ImageLayout layout) {
		this.image = layout;
		hoverLayer = -1;
		hoverUnit = -1;
		draw();
		updateDetail();
	}

	public ImageLayout imageLayout() {
		return image;
	}

	private boolean imageMode() {
		return image != null && current != null && current.layers().size() > 1
				&& current.layers().get(0).units().size() == image.pixels() * image.channels();
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

	/** Side of the square that holds the input picture (and, below it, the receptive field). */
	private double tileBox() {
		final double usable = canvas.getHeight() - MARGIN_TOP - MARGIN_BOTTOM;
		return Math.max(24, Math.min(Math.min((usable - 30) / 2, canvas.getWidth() * 0.26), 190));
	}

	private double tileWidth() {
		return tileBox() * image.width() / Math.max(image.width(), image.height());
	}

	private double tileHeight() {
		return tileBox() * image.height() / Math.max(image.width(), image.height());
	}

	private double tileY() {
		return MARGIN_TOP + 4;
	}

	private double fieldY() {
		return tileY() + tileBox() + 24;
	}

	private double columnX(int layer) {
		final int count = current.layers().size();
		if (imageMode()) {
			if (layer == 0) {
				return TILE_LEFT + tileWidth() / 2;
			}
			final double left = TILE_LEFT + tileBox() + 80;
			final double right = canvas.getWidth() - MARGIN_X;
			return count == 2 ? (left + right) / 2 : left + (right - left) * (layer - 1) / (count - 2);
		}
		final double usable = canvas.getWidth() - 2 * MARGIN_X;
		return count == 1 ? canvas.getWidth() / 2 : MARGIN_X + usable * layer / (count - 1);
	}

	private double rowY(int layer, int unit) {
		final int count = current.layers().get(layer).units().size();
		final double usable = canvas.getHeight() - MARGIN_TOP - MARGIN_BOTTOM;
		return MARGIN_TOP + usable * (unit + 0.5) / count;
	}

	private double radius() {
		final int first = imageMode() ? 1 : 0;
		final int maxUnits = current.layers().stream().skip(first).mapToInt(layer -> layer.units().size()).max()
				.orElse(1);
		final double vertical = (canvas.getHeight() - MARGIN_TOP - MARGIN_BOTTOM) / maxUnits * 0.38;
		final double span = imageMode() ? (canvas.getWidth() - TILE_LEFT - tileBox() - 80 - MARGIN_X)
				/ Math.max(1, current.layers().size() - 2) : (canvas.getWidth() - 2 * MARGIN_X)
						/ Math.max(1, current.layers().size() - 1);
		return Math.max(2.5, Math.min(22, Math.min(vertical, span * 0.16)));
	}

	private float edgeValue(int index) {
		final float weight = current.connections().get(index).weight();
		return showsChanges() ? weight - previous.connections().get(index).weight() : weight;
	}

	static float visibleWeightChange(NetworkSnapshot current, NetworkSnapshot previous, int index, int reveal) {
		if (previous == null || current.connections().size() != previous.connections().size()
				|| current.layers().size() != previous.layers().size()) {
			return 0;
		}
		final var edge = current.connections().get(index);
		final var old = previous.connections().get(index);
		if (edge.sourceLayer() != old.sourceLayer() || edge.sourceUnit() != old.sourceUnit()
				|| edge.targetLayer() != old.targetLayer() || edge.targetUnit() != old.targetUnit()
				|| edge.sourceLayer() > reveal || edge.targetLayer() > reveal) {
			return 0;
		}
		final int sources = current.layers().get(edge.sourceLayer()).units().size();
		final int targets = current.layers().get(edge.targetLayer()).units().size();
		if (sources > 6 || targets > 6 || sources * targets > 24
				|| sources != previous.layers().get(edge.sourceLayer()).units().size()
				|| targets != previous.layers().get(edge.targetLayer()).units().size()) {
			return 0;
		}
		final float delta = edge.weight() - old.weight();
		final double threshold = Math.max(1e-7, 1e-5 * Math.max(Math.abs(edge.weight()), Math.abs(old.weight())));
		return Float.isFinite(delta) && Math.abs(delta) > threshold ? delta : 0;
	}

	private void drawWeightChange(GraphicsContext g, int index, double radius) {
		final float delta = visibleWeightChange(current, previous, index, revealLayer);
		if (delta == 0) {
			return;
		}
		final var edge = current.connections().get(index);
		final double x1 = columnX(edge.sourceLayer());
		final double x2 = columnX(edge.targetLayer());
		final int sources = current.layers().get(edge.sourceLayer()).units().size();
		final int targets = current.layers().get(edge.targetLayer()).units().size();
		final double rowSpacing = (canvas.getHeight() - MARGIN_TOP - MARGIN_BOTTOM) / Math.max(sources, targets);
		if (Math.abs(x2 - x1) < Math.max(90, radius * 5) || rowSpacing < 28) {
			return;
		}
		// Stagger markers by destination to avoid piling them up at crossing midpoints.
		final double fraction = 0.32 + 0.3 * (edge.targetUnit() + 0.5) / targets;
		final double x = x1 + (x2 - x1) * fraction;
		final double y1 = rowY(edge.sourceLayer(), edge.sourceUnit());
		final double y = y1 + (rowY(edge.targetLayer(), edge.targetUnit()) - y1) * fraction;
		final double direction = delta > 0 ? -1 : 1;
		final double tip = y + direction * 5;
		g.setStroke(Color.rgb(255, 255, 255, 0.85));
		g.setLineWidth(4);
		g.strokeLine(x, y - direction * 5, x, tip);
		g.strokeLine(x - 3, tip - direction * 3, x, tip);
		g.strokeLine(x + 3, tip - direction * 3, x, tip);
		g.setStroke(delta > 0 ? Color.rgb(46, 125, 50, 0.8) : Color.rgb(198, 40, 40, 0.8));
		g.setLineWidth(1.4);
		g.strokeLine(x, y - direction * 5, x, tip);
		g.strokeLine(x - 3, tip - direction * 3, x, tip);
		g.strokeLine(x + 3, tip - direction * 3, x, tip);
	}

	private void draw() {
		final GraphicsContext g = canvas.getGraphicsContext2D();
		g.setFill(Color.WHITE);
		g.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
		if (current == null) {
			return;
		}
		final boolean pictures = imageMode();
		final double r = radius();
		final var connections = current.connections();
		float max = 1e-12f;
		for (int i = 0; i < connections.size(); i++) {
			if (!pictures || connections.get(i).sourceLayer() > 0) {
				max = Math.max(max, Math.abs(edgeValue(i)));
			}
		}
		final boolean hovering = hoverLayer >= 0 && !(pictures && hoverLayer == 0);
		if (pictures) {
			drawFan(g);
		}
		for (int pass = 0; pass < 2; pass++) {
			for (int i = 0; i < connections.size(); i++) {
				final var c = connections.get(i);
				if (pictures && c.sourceLayer() == 0) {
					continue;
				}
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
		for (int i = 0; i < connections.size(); i++) {
			final var edge = connections.get(i);
			if ((!pictures || edge.sourceLayer() > 0) && (!hovering || touchesHover(edge))) {
				drawWeightChange(g, i, r);
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
			if (pictures && l == 0) {
				g.fillText(image.width() + "×" + image.height() + (image.channels() == 1 ? " pixels" : " RGB pixels"),
						columnX(l), 31);
				drawInputTile(g);
				continue;
			}
			g.fillText(layer.units().size() + (layer.units().size() == 1 ? " unit" : " units"), columnX(l), 31);
			final boolean last = l == current.layers().size() - 1;
			final int strongest = pictures && last && revealed ? strongest(layer) : -1;
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
				if (pictures && last && u < image.classes().size()) {
					g.setFill(u == strongest ? Color.BLACK : Color.rgb(90, 90, 90));
					g.setTextAlign(TextAlignment.LEFT);
					g.setFont(u == strongest ? Font.font("System", javafx.scene.text.FontWeight.BOLD, 12)
							: Font.font(11));
					final String label = image.classes().get(u);
					g.fillText(label.length() > 10 ? label.substring(0, 9) + "…" : label, x + r + 6, y);
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

	private static int strongest(NetworkSnapshot.Layer layer) {
		int best = 0;
		for (int u = 1; u < layer.units().size(); u++) {
			best = layer.units().get(u).activation() > layer.units().get(best).activation() ? u : best;
		}
		return best;
	}

	/** Pixel values of layer 0 as a picture (dark ink on white for grayscale). */
	private void drawInputTile(GraphicsContext g) {
		final var units = current.layers().get(0).units();
		final var picture = new WritableImage(image.width(), image.height());
		final var writer = picture.getPixelWriter();
		final int channels = image.channels();
		for (int y = 0; y < image.height(); y++) {
			for (int x = 0; x < image.width(); x++) {
				final int base = (y * image.width() + x) * channels;
				if (channels == 1) {
					writer.setColor(x, y, Color.gray(1 - clamp(units.get(base).activation())));
				} else {
					writer.setColor(x, y, Color.color(clamp(units.get(base).activation()),
							clamp(units.get(base + 1).activation()), clamp(units.get(base + 2).activation())));
				}
			}
		}
		final double tw = tileWidth();
		final double th = tileHeight();
		g.setImageSmoothing(false);
		g.drawImage(picture, TILE_LEFT, tileY(), tw, th);
		g.setStroke(hoverLayer == 0 ? Color.BLACK : Color.gray(0.55));
		g.setLineWidth(1);
		g.strokeRect(TILE_LEFT - 0.5, tileY() - 0.5, tw + 1, th + 1);
		drawReceptiveField(g);
	}

	/** Incoming weights (or their last change) of the hovered first hidden unit, laid out as an image. */
	private void drawReceptiveField(GraphicsContext g) {
		final double tw = tileWidth();
		final double th = tileHeight();
		final double top = fieldY();
		g.setTextAlign(TextAlignment.LEFT);
		g.setFont(Font.font(10.5));
		g.setFill(Color.GRAY);
		if (hoverLayer != 1) {
			g.fillText("Hover a " + caption(1) + " unit", TILE_LEFT, top - 9);
			g.fillText("to see its input weights", TILE_LEFT, top + 6);
			g.setTextAlign(TextAlignment.CENTER);
			return;
		}
		final float[] field = new float[image.pixels()];
		for (int i = 0; i < current.connections().size(); i++) {
			final var c = current.connections().get(i);
			if (c.sourceLayer() == 0 && c.targetUnit() == hoverUnit) {
				field[c.sourceUnit() / image.channels()] += edgeValue(i);
			}
		}
		float max = 1e-12f;
		for (float value : field) {
			max = Math.max(max, Math.abs(value));
		}
		final var picture = new WritableImage(image.width(), image.height());
		final var writer = picture.getPixelWriter();
		for (int p = 0; p < field.length; p++) {
			writer.setColor(p % image.width(), p / image.width(),
					field[p] >= 0 ? Color.WHITE.interpolate(POSITIVE, field[p] / max)
							: Color.WHITE.interpolate(NEGATIVE, -field[p] / max));
		}
		g.fillText((showsChanges() ? "Δ input weights of " : "Input weights of ") + caption(1) + " #"
				+ (hoverUnit + 1), TILE_LEFT, top - 9);
		g.setImageSmoothing(false);
		g.drawImage(picture, TILE_LEFT, top, tw, th);
		g.setStroke(Color.gray(0.55));
		g.strokeRect(TILE_LEFT - 0.5, top - 0.5, tw + 1, th + 1);
		g.fillText(String.format(Locale.ROOT, "blue +, red −, max |%s| %.3g", showsChanges() ? "Δw" : "w", max),
				TILE_LEFT, top + th + 10);
		g.setTextAlign(TextAlignment.CENTER);
	}

	/**
	 * One line per first-layer unit instead of thousands of pixel edges: width shows the norm of its
	 * input weights (or of their change), colour the sign of the weighted input it currently receives.
	 */
	private void drawFan(GraphicsContext g) {
		final int hidden = current.layers().get(1).units().size();
		final double[] norm = new double[hidden];
		final double[] drive = new double[hidden];
		final var inputs = current.layers().get(0).units();
		for (int i = 0; i < current.connections().size(); i++) {
			final var c = current.connections().get(i);
			if (c.sourceLayer() == 0) {
				final float value = edgeValue(i);
				norm[c.targetUnit()] += value * value;
				drive[c.targetUnit()] += value * inputs.get(c.sourceUnit()).activation();
			}
		}
		double max = 1e-12;
		for (int u = 0; u < hidden; u++) {
			norm[u] = Math.sqrt(norm[u]);
			max = Math.max(max, norm[u]);
		}
		final double startX = TILE_LEFT + tileWidth() + 2;
		final double startY = tileY() + tileHeight() / 2;
		for (int u = 0; u < hidden; u++) {
			final double strength = norm[u] / max;
			final boolean hovered = hoverLayer == 1 && hoverUnit == u;
			final double alpha = hoverLayer >= 1 && !hovered ? 0.05 : 0.1 + 0.5 * strength;
			g.setStroke((drive[u] >= 0 ? POSITIVE : NEGATIVE).deriveColor(0, 1, 1, hovered ? 0.9 : alpha));
			g.setLineWidth(hovered ? 3 : 0.5 + 3 * strength);
			g.strokeLine(startX, startY, columnX(1), rowY(1, u));
		}
	}

	private static double clamp(float value) {
		return Math.max(0, Math.min(1, value));
	}

	/** Pixel index under (x, y) in the input picture, or -1. */
	private int pixelAt(double x, double y) {
		final double tw = tileWidth();
		final double th = tileHeight();
		if (x < TILE_LEFT || y < tileY() || x >= TILE_LEFT + tw || y >= tileY() + th) {
			return -1;
		}
		final int px = Math.min(image.width() - 1, (int) ((x - TILE_LEFT) / tw * image.width()));
		final int py = Math.min(image.height() - 1, (int) ((y - tileY()) / th * image.height()));
		return py * image.width() + px;
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
		final boolean pictures = imageMode();
		final int pixel = pictures ? pixelAt(x, y) : -1;
		if (pixel >= 0) {
			layer = 0;
			unit = pixel * image.channels();
		}
		for (int l = pictures ? 1 : 0; l < current.layers().size() && pixel < 0; l++) {
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
		if (hoverLayer == 0 && imageMode()) {
			final int pixel = hoverUnit / image.channels();
			text.append(String.format(Locale.ROOT, "Pixel (x %d, y %d) = %.3f", pixel % image.width(),
					pixel / image.width(), unit.activation()));
			if (image.channels() == 3) {
				final var units = current.layers().get(0).units();
				text.append(String.format(Locale.ROOT, " (r), g %.3f, b %.3f", units.get(hoverUnit + 1).activation(),
						units.get(hoverUnit + 2).activation()));
			}
			text.append("\nHover a first hidden unit to see which pixels drive it.");
			detail.set(text.toString());
			return;
		}
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
