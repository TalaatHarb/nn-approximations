package net.talaatharb.nn.ui;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.Consumer;

import javafx.geometry.Pos;
import javafx.geometry.VPos;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Tooltip;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Pane;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.scene.text.TextAlignment;

/**
 * Compact view of an image classifier: the inspected sample with its class scores, a confusion
 * matrix with accuracies, and a paged thumbnail gallery (green = correct, red = wrong) whose
 * thumbnails select the sample to inspect.
 */
public class ImageClassificationView extends VBox {

	/** Image samples of one split; {@code prefix} forms ids such as {@code test:12}. */
	public record Split(String name, String prefix, float[][] pixels, int[] labels) {
	}

	private static final double THUMB = 34;
	private static final double GAP = 4;
	private static final int GALLERY_ROWS = 2;
	private static final Color CORRECT = Color.rgb(46, 125, 50);
	private static final Color WRONG = Color.rgb(198, 40, 40);
	private static final Color BAR = Color.rgb(33, 102, 172);

	private final Canvas sample = new Canvas(84, 84);
	private final Label sampleTitle = new Label();
	private final Label sampleVerdict = new Label();
	private final Canvas scores = new Canvas(170, 150);
	private final Canvas confusion = new Canvas(150, 150);
	private final Label confusionTitle = new Label("Confusion");
	private final Label accuracy = new Label();
	private final ComboBox<String> splitChoice = new ComboBox<>();
	private final CheckBox mistakesOnly = new CheckBox("Mistakes only");
	private final Button previousPage = new Button("◀");
	private final Button nextPage = new Button("▶");
	private final Label pageLabel = new Label();
	private final Canvas gallery = new Canvas(400, GALLERY_ROWS * (THUMB + GAP) + GAP);
	private final Pane galleryHolder = new Pane(gallery);

	private NetworkCanvas.ImageLayout layout;
	private final List<Split> splits = new ArrayList<>();
	private final List<int[]> predicted = new ArrayList<>();
	private final List<int[]> galleryHits = new ArrayList<>();
	private int page;
	private int hoverThumb = -1;
	private Consumer<String> onSelect = id -> {
	};

	public ImageClassificationView() {
		super(6);
		sampleTitle.setStyle("-fx-font-weight: bold;");
		sampleVerdict.setWrapText(true);
		sampleVerdict.setMaxWidth(120);
		accuracy.setWrapText(true);
		accuracy.setMaxWidth(170);
		confusionTitle.setStyle("-fx-font-weight: bold;");
		final var sampleBox = new VBox(4, sampleTitle, sample, sampleVerdict);
		sampleBox.setMinWidth(100);
		final var scoresBox = new VBox(4, label("Class scores"), scores);
		final var confusionBox = new VBox(4, confusionTitle, confusion, accuracy);
		final var top = new HBox(14, sampleBox, scoresBox, confusionBox);

		splitChoice.setTooltip(new Tooltip("Which samples the gallery shows"));
		splitChoice.valueProperty().addListener((observable, oldValue, value) -> {
			page = 0;
			drawGallery();
		});
		mistakesOnly.setTooltip(new Tooltip("Only show misclassified samples"));
		mistakesOnly.selectedProperty().addListener((observable, oldValue, value) -> {
			page = 0;
			drawGallery();
		});
		previousPage.setOnAction(event -> {
			page = Math.max(0, page - 1);
			drawGallery();
		});
		nextPage.setOnAction(event -> {
			page++;
			drawGallery();
		});
		final var controls = new HBox(8, label("Gallery"), splitChoice, mistakesOnly, previousPage, pageLabel,
				nextPage);
		controls.setAlignment(Pos.CENTER_LEFT);
		galleryHolder.setMinHeight(gallery.getHeight());
		galleryHolder.setPrefHeight(gallery.getHeight());
		galleryHolder.setMinWidth(0);
		galleryHolder.widthProperty().addListener((observable, oldValue, value) -> {
			gallery.setWidth(Math.max(THUMB + 2 * GAP, value.doubleValue()));
			drawGallery();
		});
		gallery.setOnMouseMoved(event -> {
			final int index = thumbAt(event.getX(), event.getY());
			if (index != hoverThumb) {
				hoverThumb = index;
				drawGallery();
			}
		});
		gallery.setOnMouseExited(event -> {
			hoverThumb = -1;
			drawGallery();
		});
		gallery.setOnMouseClicked(event -> {
			final int index = thumbAt(event.getX(), event.getY());
			if (index >= 0) {
				onSelect.accept(currentSplit().prefix() + ":" + index);
			}
		});
		final var hint = new Label("Click a thumbnail to run it through the network; hover the input picture or"
				+ " a first hidden unit on the left for pixels and receptive fields.");
		hint.setWrapText(true);
		hint.setStyle("-fx-text-fill: #555; -fx-font-size: 11px;");
		getChildren().addAll(top, controls, galleryHolder, hint);
		VBox.setVgrow(galleryHolder, Priority.NEVER);
	}

	private static Label label(String text) {
		final var label = new Label(text);
		label.setStyle("-fx-font-weight: bold;");
		label.setMinWidth(Label.USE_PREF_SIZE);
		return label;
	}

	public void setOnSampleSelected(Consumer<String> handler) {
		onSelect = handler == null ? id -> {
		} : handler;
	}

	/** Configures the classes, image size and sample splits; clears predictions and the sample. */
	public void setData(NetworkCanvas.ImageLayout layout, List<Split> splits) {
		this.layout = layout;
		this.splits.clear();
		this.splits.addAll(splits);
		predicted.clear();
		splits.forEach(split -> predicted.add(null));
		page = 0;
		final String chosen = splitChoice.getValue();
		splitChoice.getItems().setAll(splits.stream().map(Split::name).toList());
		splitChoice.setValue(splits.stream().map(Split::name).filter(name -> name.equals(chosen)).findFirst()
				.orElse(splits.isEmpty() ? null : splits.get(splits.size() - 1).name()));
		confusionTitle.setText("Confusion · " + (splits.isEmpty() ? "" : splits.get(splits.size() - 1).name()));
		showSample("", null, -1, null);
		update(List.of());
	}

	/** Predicted class per sample for each split (same order as {@link #setData}); null entries are unknown. */
	public void update(List<int[]> predictions) {
		for (int i = 0; i < splits.size(); i++) {
			predicted.set(i, i < predictions.size() ? predictions.get(i) : null);
		}
		final var text = new StringBuilder();
		for (int i = 0; i < splits.size(); i++) {
			final double value = accuracy(i);
			if (!Double.isNaN(value)) {
				text.append(text.isEmpty() ? "" : "\n").append(String.format(Locale.ROOT, "%s accuracy %.1f%% (%d)",
						splits.get(i).name(), 100 * value, splits.get(i).labels().length));
			}
		}
		accuracy.setText(text.toString());
		drawConfusion();
		drawGallery();
	}

	/** Share of correctly classified samples of a split, or NaN when unknown. */
	public double accuracy(int split) {
		final int[] guesses = predicted.get(split);
		final int[] labels = splits.get(split).labels();
		if (guesses == null || labels.length == 0) {
			return Double.NaN;
		}
		int correct = 0;
		for (int i = 0; i < labels.length; i++) {
			correct += guesses[i] == labels[i] ? 1 : 0;
		}
		return (double) correct / labels.length;
	}

	/**
	 * @param title  e.g. "test:12" or a file name
	 * @param pixels image values in 0..1, or null to clear
	 * @param label  true class, or -1 when unknown
	 * @param output network outputs (one score per class)
	 */
	public void showSample(String title, float[] pixels, int label, float[] output) {
		sampleTitle.setText(title);
		final GraphicsContext g = sample.getGraphicsContext2D();
		g.setFill(Color.WHITE);
		g.fillRect(0, 0, sample.getWidth(), sample.getHeight());
		final int guess = output == null ? -1 : argmax(output);
		if (pixels != null && layout != null) {
			drawPicture(g, pixels, 1, 1, sample.getWidth() - 2, sample.getHeight() - 2);
			g.setStroke(label < 0 || guess < 0 ? Color.gray(0.5) : guess == label ? CORRECT : WRONG);
			g.setLineWidth(2);
			g.strokeRect(1, 1, sample.getWidth() - 2, sample.getHeight() - 2);
		}
		final var classes = layout == null ? List.<String>of() : layout.classes();
		sampleVerdict.setText(guess < 0 ? "" : (label >= 0 ? "label " + name(classes, label) + "\n" : "")
				+ "predicted " + name(classes, guess)
				+ (label < 0 ? "" : guess == label ? " ✓" : " ✗"));
		sampleVerdict.setTextFill(label < 0 || guess < 0 ? Color.BLACK : guess == label ? CORRECT : WRONG);
		drawScores(output, label);
	}

	private static String name(List<String> classes, int index) {
		return index < classes.size() ? classes.get(index) : Integer.toString(index);
	}

	private void drawScores(float[] output, int label) {
		final GraphicsContext g = scores.getGraphicsContext2D();
		g.setFill(Color.WHITE);
		g.fillRect(0, 0, scores.getWidth(), scores.getHeight());
		if (output == null || layout == null) {
			return;
		}
		final var order = new ArrayList<Integer>();
		for (int i = 0; i < output.length; i++) {
			order.add(i);
		}
		final int shown = Math.min(10, output.length);
		final List<Integer> rows = output.length <= shown ? order
				: order.stream().sorted((a, b) -> Float.compare(output[b], output[a])).limit(shown).toList();
		final double rowHeight = scores.getHeight() / shown;
		final double labelWidth = 44;
		final double barWidth = scores.getWidth() - labelWidth - 38;
		final int best = argmax(output);
		g.setTextBaseline(VPos.CENTER);
		for (int r = 0; r < rows.size(); r++) {
			final int c = rows.get(r);
			final double y = r * rowHeight;
			final double value = Math.max(0, Math.min(1, output[c]));
			g.setFont(c == best ? Font.font("System", FontWeight.BOLD, 11) : Font.font(11));
			g.setFill(c == label ? CORRECT : Color.rgb(40, 40, 40));
			g.setTextAlign(TextAlignment.RIGHT);
			final String name = name(layout.classes(), c);
			g.fillText(name.length() > 7 ? name.substring(0, 6) + "…" : name, labelWidth - 4, y + rowHeight / 2);
			g.setFill(Color.gray(0.93));
			g.fillRect(labelWidth, y + 2, barWidth, rowHeight - 4);
			g.setFill(c == best ? (label < 0 || c == label ? BAR : WRONG) : BAR.deriveColor(0, 0.6, 1.2, 0.7));
			g.fillRect(labelWidth, y + 2, barWidth * value, rowHeight - 4);
			g.setFill(Color.rgb(60, 60, 60));
			g.setTextAlign(TextAlignment.LEFT);
			g.setFont(Font.font(10));
			g.fillText(String.format(Locale.ROOT, "%.2f", output[c]), labelWidth + barWidth + 4, y + rowHeight / 2);
		}
	}

	private void drawConfusion() {
		final GraphicsContext g = confusion.getGraphicsContext2D();
		g.setFill(Color.WHITE);
		g.fillRect(0, 0, confusion.getWidth(), confusion.getHeight());
		if (layout == null || splits.isEmpty() || predicted.get(splits.size() - 1) == null) {
			g.setFill(Color.GRAY);
			g.setFont(Font.font(11));
			g.setTextAlign(TextAlignment.LEFT);
			g.fillText("No predictions yet", 4, 14);
			return;
		}
		final int n = layout.classes().size();
		final int[][] counts = new int[n][n];
		final var split = splits.get(splits.size() - 1);
		final int[] guesses = predicted.get(splits.size() - 1);
		final int[] rowTotals = new int[n];
		for (int i = 0; i < split.labels().length; i++) {
			counts[split.labels()[i]][guesses[i]]++;
			rowTotals[split.labels()[i]]++;
		}
		final double axis = n <= 20 ? 14 : 0;
		final double cell = (confusion.getWidth() - axis) / n;
		g.setFont(Font.font(Math.max(7, Math.min(10, cell * 0.7))));
		g.setTextAlign(TextAlignment.CENTER);
		g.setTextBaseline(VPos.CENTER);
		for (int t = 0; t < n; t++) {
			for (int p = 0; p < n; p++) {
				final double share = rowTotals[t] == 0 ? 0 : (double) counts[t][p] / rowTotals[t];
				final Color base = t == p ? CORRECT : WRONG;
				g.setFill(counts[t][p] == 0 ? Color.gray(0.97) : Color.WHITE.interpolate(base, 0.15 + 0.85 * share));
				g.fillRect(axis + p * cell, axis + t * cell, cell - (cell > 4 ? 1 : 0), cell - (cell > 4 ? 1 : 0));
				if (cell >= 13 && counts[t][p] > 0) {
					g.setFill(share > 0.55 ? Color.WHITE : Color.BLACK);
					g.fillText(Integer.toString(counts[t][p]), axis + (p + 0.5) * cell, axis + (t + 0.5) * cell);
				}
			}
			if (axis > 0) {
				g.setFill(Color.rgb(70, 70, 70));
				final String label = name(layout.classes(), t);
				final String shortLabel = label.length() > 2 ? label.substring(0, 2) : label;
				g.fillText(shortLabel, axis / 2, axis + (t + 0.5) * cell);
				g.fillText(shortLabel, axis + (t + 0.5) * cell, axis / 2);
			}
		}
		Tooltip.install(confusion, new Tooltip("Rows: true class, columns: predicted class ("
				+ split.name() + " samples). Diagonal = correct."));
	}

	private Split currentSplit() {
		final String name = splitChoice.getValue();
		return splits.stream().filter(split -> split.name().equals(name)).findFirst()
				.orElse(splits.isEmpty() ? null : splits.get(0));
	}

	private int[] currentPredictions() {
		final int index = splits.indexOf(currentSplit());
		return index < 0 ? null : predicted.get(index);
	}

	private void drawGallery() {
		final GraphicsContext g = gallery.getGraphicsContext2D();
		g.setFill(Color.WHITE);
		g.fillRect(0, 0, gallery.getWidth(), gallery.getHeight());
		galleryHits.clear();
		final var split = currentSplit();
		if (split == null || layout == null) {
			pageLabel.setText("");
			return;
		}
		final int[] guesses = currentPredictions();
		final var indices = new ArrayList<Integer>();
		for (int i = 0; i < split.labels().length; i++) {
			if (!mistakesOnly.isSelected() || guesses != null && guesses[i] != split.labels()[i]) {
				indices.add(i);
			}
		}
		final int columns = Math.max(1, (int) ((gallery.getWidth() - GAP) / (THUMB + GAP)));
		final int perPage = columns * GALLERY_ROWS;
		final int pages = Math.max(1, (indices.size() + perPage - 1) / perPage);
		page = Math.min(page, pages - 1);
		pageLabel.setText(indices.isEmpty() ? (mistakesOnly.isSelected() ? "no mistakes" : "empty")
				: String.format(Locale.ROOT, "%d / %d  (%d samples)", page + 1, pages, indices.size()));
		previousPage.setDisable(page == 0);
		nextPage.setDisable(page >= pages - 1);
		for (int k = 0; k < perPage && page * perPage + k < indices.size(); k++) {
			final int index = indices.get(page * perPage + k);
			final double x = GAP + (k % columns) * (THUMB + GAP);
			final double y = GAP + (k / columns) * (THUMB + GAP);
			drawPicture(g, split.pixels()[index], x, y, THUMB, THUMB);
			final boolean known = guesses != null;
			g.setStroke(!known ? Color.gray(0.6) : guesses[index] == split.labels()[index] ? CORRECT : WRONG);
			g.setLineWidth(index == hoverThumb ? 3 : 1.5);
			g.strokeRect(x, y, THUMB, THUMB);
			if (known && guesses[index] != split.labels()[index]) {
				g.setFill(WRONG);
				g.setFont(Font.font("System", FontWeight.BOLD, 9));
				g.setTextAlign(TextAlignment.RIGHT);
				g.setTextBaseline(VPos.BOTTOM);
				final String name = name(layout.classes(), guesses[index]);
				g.fillText(name.length() > 3 ? name.substring(0, 3) : name, x + THUMB - 2, y + THUMB - 1);
			}
			galleryHits.add(new int[] { (int) x, (int) y, index });
		}
		Tooltip.install(gallery, new Tooltip("Border: green = correct, red = wrong (red text = predicted class)."
				+ " Click to inspect."));
	}

	private int thumbAt(double x, double y) {
		for (int[] hit : galleryHits) {
			if (x >= hit[0] && x < hit[0] + THUMB && y >= hit[1] && y < hit[1] + THUMB) {
				return hit[2];
			}
		}
		return -1;
	}

	private void drawPicture(GraphicsContext g, float[] pixels, double x, double y, double w, double h) {
		final var picture = new WritableImage(layout.width(), layout.height());
		final var writer = picture.getPixelWriter();
		final int channels = layout.channels();
		for (int py = 0; py < layout.height(); py++) {
			for (int px = 0; px < layout.width(); px++) {
				final int base = (py * layout.width() + px) * channels;
				if (base + channels > pixels.length) {
					continue;
				}
				writer.setColor(px, py, channels == 1 ? Color.gray(1 - clamp(pixels[base]))
						: Color.color(clamp(pixels[base]), clamp(pixels[base + 1]), clamp(pixels[base + 2])));
			}
		}
		g.setImageSmoothing(false);
		g.drawImage(picture, x, y, w, h);
	}

	private static double clamp(float value) {
		return Math.max(0, Math.min(1, value));
	}

	public static int argmax(float[] values) {
		int best = 0;
		for (int i = 1; i < values.length; i++) {
			best = values[i] > values[best] ? i : best;
		}
		return best;
	}

	/** For tests: number of thumbnails currently drawn. */
	int visibleThumbnails() {
		return galleryHits.size();
	}

	/** For tests: the gallery canvas. */
	Canvas galleryCanvas() {
		return gallery;
	}
}
