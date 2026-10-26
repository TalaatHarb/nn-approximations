package net.talaatharb.nn.config;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import net.talaatharb.nn.data.DatasetCache;
import net.talaatharb.nn.data.DatasetCache.ImageSample;
import net.talaatharb.nn.data.HuggingFaceClient;
import net.talaatharb.nn.data.ImageLoader;
import net.talaatharb.nn.structures.TrainingData;

/**
 * Image classification samples from a Hugging Face dataset (e.g. {@code ylecun/mnist}). Each sample
 * is an image file in the local {@link DatasetCache}; inputs are its pixels scaled to 0..1 and the
 * target is a one-hot vector over {@link #classes()}. The class names are stored so the model can
 * be validated (and its layer sizes derived) without network access.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({ "repository", "config", "split", "imageColumn", "labelColumn", "classes", "samples",
		"testSplit", "testSamples", "image" })
public record HuggingFaceDataset(String repository, String config, String split, String imageColumn,
		String labelColumn, List<String> classes, int samples, String testSplit, int testSamples, ImageSpec image)
		implements DatasetSpec {

	public static final int MAX_IMAGE_SIDE = 64;
	public static final int MAX_CLASSES = 100;

	/** Every image is resized to width x height; grayscale gives 1 input per pixel, colour 3. */
	public record ImageSpec(int width, int height, boolean grayscale) {
		@JsonIgnore
		public int channels() {
			return grayscale ? 1 : 3;
		}
	}

	/** Raised when the samples have not been downloaded yet; fetch them with {@link #fetch}. */
	public static final class NotDownloadedException extends IllegalStateException {
		private static final long serialVersionUID = 1L;

		NotDownloadedException(String message) {
			super(message);
		}
	}

	private static final Map<String, List<String>> PIXEL_NAMES = new ConcurrentHashMap<>();

	public HuggingFaceDataset {
		config = blankTo(config, "default");
		split = blankTo(split, "train");
		imageColumn = blankTo(imageColumn, "image");
		labelColumn = blankTo(labelColumn, "label");
		classes = classes == null ? List.of() : List.copyOf(classes);
		testSplit = testSplit == null || testSplit.isBlank() ? null : testSplit.trim();
		image = image == null ? new ImageSpec(28, 28, true) : image;
	}

	private static String blankTo(String value, String fallback) {
		return value == null || value.isBlank() ? fallback : value.trim();
	}

	@Override
	public int inputSize() {
		return image.width() * image.height() * image.channels();
	}

	@Override
	public int outputSize() {
		return classes.size();
	}

	@Override
	public List<String> inputNames() {
		final String key = image.width() + "x" + image.height() + (image.grayscale() ? "g" : "c");
		return PIXEL_NAMES.computeIfAbsent(key, ignored -> {
			final var names = new ArrayList<String>(inputSize());
			for (int y = 0; y < image.height(); y++) {
				for (int x = 0; x < image.width(); x++) {
					if (image.grayscale()) {
						names.add("px_" + y + "_" + x);
					} else {
						names.add("r_" + y + "_" + x);
						names.add("g_" + y + "_" + x);
						names.add("b_" + y + "_" + x);
					}
				}
			}
			return List.copyOf(names);
		});
	}

	@Override
	public List<String> outputNames() {
		return classes;
	}

	@Override
	public void validate(List<String> errors) {
		if (repository == null || !HuggingFaceClient.REPOSITORY.matcher(repository).matches()) {
			errors.add("dataset.repository must look like 'owner/name', e.g. ylecun/mnist");
		}
		for (var entry : Map.of("config", config, "split", split, "imageColumn", imageColumn, "labelColumn",
				labelColumn).entrySet()) {
			if (!HuggingFaceClient.NAME.matcher(entry.getValue()).matches() || entry.getValue().startsWith(".")) {
				errors.add("dataset." + entry.getKey() + " may only use letters, digits, '_', '.' and '-'");
			}
		}
		if (testSplit != null && (!HuggingFaceClient.NAME.matcher(testSplit).matches() || testSplit.startsWith("."))) {
			errors.add("dataset.testSplit may only use letters, digits, '_', '.' and '-'");
		}
		if (classes.size() < 2 || classes.size() > MAX_CLASSES) {
			errors.add("dataset.classes needs between 2 and " + MAX_CLASSES + " class names (use Load info)");
		}
		if (classes.stream().anyMatch(name -> name == null || name.isBlank())
				|| new HashSet<>(classes).size() != classes.size()) {
			errors.add("dataset.classes must be unique and non-blank");
		}
		if (samples <= 0 || samples > MAX_SAMPLES) {
			errors.add("dataset.samples must be between 1 and " + MAX_SAMPLES);
		}
		if (testSamples < 0 || testSamples > MAX_SAMPLES) {
			errors.add("dataset.testSamples must be between 0 and " + MAX_SAMPLES);
		}
		if (testSplit != null && testSplit.equals(split)) {
			errors.add("dataset.testSplit must differ from dataset.split");
		}
		if (image.width() <= 0 || image.height() <= 0 || image.width() > MAX_IMAGE_SIDE
				|| image.height() > MAX_IMAGE_SIDE) {
			errors.add("dataset.image width and height must be between 1 and " + MAX_IMAGE_SIDE);
		}
	}

	@JsonIgnore
	public DatasetCache.Request trainingRequest() {
		return new DatasetCache.Request(repository, config, split, imageColumn, labelColumn, classes, samples);
	}

	/** Held-out split request, or null when no test split is configured. */
	@JsonIgnore
	public DatasetCache.Request testRequest() {
		return hasTestSplit()
				? new DatasetCache.Request(repository, config, testSplit, imageColumn, labelColumn, classes, testSamples)
				: null;
	}

	@JsonIgnore
	public boolean hasTestSplit() {
		return testSplit != null && testSamples > 0;
	}

	/** True when every configured split is in the default cache. */
	@JsonIgnore
	public boolean isDownloaded() {
		final var cache = DatasetCache.openDefault();
		return cache.cached(trainingRequest()).isPresent()
				&& (!hasTestSplit() || cache.cached(testRequest()).isPresent());
	}

	/** Cached training samples; throws {@link NotDownloadedException} if they are missing. */
	@JsonIgnore
	public List<ImageSample> trainingSamples() {
		return cachedOrThrow(trainingRequest());
	}

	@JsonIgnore
	public List<ImageSample> heldOutSamples() {
		return hasTestSplit() ? cachedOrThrow(testRequest()) : List.of();
	}

	private List<ImageSample> cachedOrThrow(DatasetCache.Request request) {
		return DatasetCache.openDefault().cached(request).orElseThrow(() -> new NotDownloadedException(
				"Samples of " + repository + " " + config + "/" + request.split() + " are not downloaded yet"
						+ " (they are fetched from Hugging Face into " + DatasetCache.defaultDirectory() + ")"));
	}

	/** Downloads (if needed) the training and test splits into the default cache. */
	public void fetch(HuggingFaceClient.Progress progress) throws java.io.IOException, InterruptedException {
		final var cache = DatasetCache.openDefault();
		cache.fetch(trainingRequest(), progress);
		if (hasTestSplit()) {
			cache.fetch(testRequest(), progress);
		}
	}

	@Override
	public TrainingData toTrainingData() {
		return toTrainingData(trainingSamples());
	}

	public TrainingData toTrainingData(List<ImageSample> list) {
		final float[][] inputs = new float[list.size()][];
		final float[][] outputs = new float[list.size()][];
		for (int i = 0; i < list.size(); i++) {
			inputs[i] = pixels(list.get(i).image());
			outputs[i] = new float[classes.size()];
			outputs[i][list.get(i).label()] = 1;
		}
		return new TrainingData(inputs, outputs);
	}

	public float[] pixels(Path file) {
		return ImageLoader.load(file, image.width(), image.height(), image.grayscale());
	}

	@Override
	public float[][] inputBounds() {
		final float[][] bounds = new float[inputSize()][];
		for (int i = 0; i < bounds.length; i++) {
			bounds[i] = new float[] { 0, 1 };
		}
		return bounds;
	}
}
