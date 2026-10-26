package net.talaatharb.nn.ui;

import java.util.List;
import java.util.Locale;
import java.util.function.UnaryOperator;

import javafx.geometry.VPos;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.image.PixelWriter;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.Region;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.TextAlignment;

/**
 * Heatmap of one network output over a 2-D input plane, with training samples overlaid
 * and (for classification) the 0.5 decision boundary.
 */
public class OutputSurface extends Region {

	public static final int RESOLUTION = 48;

	/** Output values on a RESOLUTION x RESOLUTION grid, row 0 at the bottom (minimum y). */
	public record Grid(float xMin, float xMax, float yMin, float yMax, float[][] values) {
		float min() {
			float min = Float.POSITIVE_INFINITY;
			for (float[] row : values) {
				for (float v : row) {
					min = Math.min(min, v);
				}
			}
			return min;
		}

		float max() {
			float max = Float.NEGATIVE_INFINITY;
			for (float[] row : values) {
				for (float v : row) {
					max = Math.max(max, v);
				}
			}
			return max;
		}
	}

	/** Must run on the thread that owns the network (e.g. the training thread). */
	public static Grid compute(UnaryOperator<float[]> network, float[][] bounds, int output) {
		final float padX = (bounds[0][1] - bounds[0][0]) * 0.1f;
		final float padY = (bounds[1][1] - bounds[1][0]) * 0.1f;
		final float xMin = bounds[0][0] - padX;
		final float xMax = bounds[0][1] + padX;
		final float yMin = bounds[1][0] - padY;
		final float yMax = bounds[1][1] + padY;
		final float[][] values = new float[RESOLUTION][RESOLUTION];
		final float[] input = new float[2];
		for (int row = 0; row < RESOLUTION; row++) {
			input[1] = yMin + (yMax - yMin) * row / (RESOLUTION - 1);
			for (int column = 0; column < RESOLUTION; column++) {
				input[0] = xMin + (xMax - xMin) * column / (RESOLUTION - 1);
				values[row][column] = network.apply(input)[output];
			}
		}
		return new Grid(xMin, xMax, yMin, yMax, values);
	}

	private static final double MARGIN = 34;
	private final Canvas canvas = new Canvas();
	private Grid grid;
	private float[][] inputs = new float[0][];
	private float[] targets = new float[0];
	private boolean classification;
	private List<String> axisNames = List.of("x1", "x2");
	private String outputName = "y";

	public OutputSurface() {
		getChildren().add(canvas);
		setMinSize(200, 200);
		setPrefSize(360, 320);
	}

	public void show(Grid grid, float[][] inputs, float[] targets, boolean classification, List<String> axisNames,
			String outputName) {
		this.grid = grid;
		this.inputs = inputs;
		this.targets = targets;
		this.classification = classification;
		this.axisNames = axisNames;
		this.outputName = outputName;
		draw();
	}

	public void update(Grid grid) {
		this.grid = grid;
		draw();
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

	private Color color(float value, float low, float high) {
		if (classification) {
			final double t = Math.max(0, Math.min(1, value));
			return t < 0.5 ? Color.rgb(198, 40, 40).interpolate(Color.WHITE, t * 2)
					: Color.WHITE.interpolate(Color.rgb(33, 102, 172), (t - 0.5) * 2);
		}
		final double t = high > low ? (value - low) / (high - low) : 0.5;
		return Color.hsb(260 - 220 * Math.max(0, Math.min(1, t)), 0.65, 0.95);
	}

	private void draw() {
		final GraphicsContext g = canvas.getGraphicsContext2D();
		g.setFill(Color.WHITE);
		g.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());
		if (grid == null) {
			return;
		}
		final double size = Math.max(10, Math.min(canvas.getWidth() - 2 * MARGIN, canvas.getHeight() - 2 * MARGIN));
		final double left = (canvas.getWidth() - size) / 2;
		final double top = (canvas.getHeight() - size) / 2;
		float low = grid.min();
		float high = grid.max();
		for (float target : targets) {
			low = Math.min(low, target);
			high = Math.max(high, target);
		}
		final var image = new WritableImage(RESOLUTION, RESOLUTION);
		final PixelWriter pixels = image.getPixelWriter();
		for (int row = 0; row < RESOLUTION; row++) {
			for (int column = 0; column < RESOLUTION; column++) {
				pixels.setColor(column, RESOLUTION - 1 - row, color(grid.values()[row][column], low, high));
			}
		}
		g.setImageSmoothing(true);
		g.drawImage(image, left, top, size, size);
		g.setStroke(Color.gray(0.4));
		g.setLineWidth(1);
		g.strokeRect(left, top, size, size);
		if (classification) {
			drawContour(g, left, top, size, 0.5f);
		}
		for (int i = 0; i < inputs.length && i < targets.length; i++) {
			final double x = left + (inputs[i][0] - grid.xMin()) / (grid.xMax() - grid.xMin()) * size;
			final double y = top + size - (inputs[i][1] - grid.yMin()) / (grid.yMax() - grid.yMin()) * size;
			final double r = inputs.length > 400 ? 2 : 5;
			g.setFill(color(targets[i], low, high));
			g.fillOval(x - r, y - r, 2 * r, 2 * r);
			g.setStroke(Color.BLACK);
			g.setLineWidth(inputs.length > 400 ? 0.5 : 1.5);
			g.strokeOval(x - r, y - r, 2 * r, 2 * r);
		}
		g.setFill(Color.rgb(40, 40, 40));
		g.setFont(Font.font(11));
		g.setTextBaseline(VPos.TOP);
		g.setTextAlign(TextAlignment.LEFT);
		g.fillText(fmt(grid.xMin()), left, top + size + 3);
		g.setTextAlign(TextAlignment.RIGHT);
		g.fillText(fmt(grid.xMax()), left + size, top + size + 3);
		g.setTextAlign(TextAlignment.CENTER);
		g.fillText(axisNames.get(0) + " →", left + size / 2, top + size + 3);
		g.setTextBaseline(VPos.BOTTOM);
		g.fillText(String.format(Locale.ROOT, "%s over (%s, %s)   range %s .. %s%s", outputName, axisNames.get(0),
				axisNames.get(1), fmt(grid.min()), fmt(grid.max()), classification ? "   — = 0.5 boundary" : ""),
				left + size / 2, top - 4);
		g.save();
		g.translate(left - 4, top + size / 2);
		g.rotate(-90);
		g.fillText(axisNames.get(1) + " →", 0, 0);
		g.restore();
		g.setTextAlign(TextAlignment.RIGHT);
		g.setTextBaseline(VPos.BOTTOM);
		g.fillText(fmt(grid.yMin()), left - 4, top + size);
		g.setTextBaseline(VPos.TOP);
		g.fillText(fmt(grid.yMax()), left - 4, top);
	}

	/** Marching squares over the grid cells. */
	private void drawContour(GraphicsContext g, double left, double top, double size, float level) {
		final float[][] v = grid.values();
		final double cell = size / (RESOLUTION - 1);
		g.setStroke(Color.BLACK);
		g.setLineWidth(2);
		for (int row = 0; row < RESOLUTION - 1; row++) {
			for (int column = 0; column < RESOLUTION - 1; column++) {
				final float a = v[row][column];
				final float b = v[row][column + 1];
				final float c = v[row + 1][column + 1];
				final float d = v[row + 1][column];
				final double[] points = new double[8];
				int count = 0;
				// Edges: bottom a-b, right b-c, top d-c, left a-d (row grows upward).
				final float[][] edges = { { a, b, 0, 0, 1, 0 }, { b, c, 1, 0, 1, 1 }, { d, c, 0, 1, 1, 1 },
						{ a, d, 0, 0, 0, 1 } };
				for (float[] e : edges) {
					if ((e[0] < level) != (e[1] < level)) {
						final double t = (level - e[0]) / (e[1] - e[0]);
						final double gx = column + e[2] + (e[4] - e[2]) * t;
						final double gy = row + e[3] + (e[5] - e[3]) * t;
						points[count++] = left + gx * cell;
						points[count++] = top + size - gy * cell;
					}
				}
				if (count >= 4) {
					g.strokeLine(points[0], points[1], points[2], points[3]);
				}
				if (count == 8) {
					g.strokeLine(points[4], points[5], points[6], points[7]);
				}
			}
		}
	}

	private static String fmt(float value) {
		return String.format(Locale.ROOT, "%.3g", value);
	}
}
