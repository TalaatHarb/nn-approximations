package net.talaatharb.nn.config;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

import net.talaatharb.nn.data.DatasetCache;
import net.talaatharb.nn.data.HuggingFaceClient;
import net.talaatharb.nn.structures.TrainingData;

/**
 * Rows of a tabular Hugging Face dataset (e.g. {@code scikit-learn/iris}): numeric {@link #inputColumns()}
 * predict {@link #targetColumn()}. With {@link #classes()} the target is categorical (one output per
 * class, one-hot); otherwise it is a number. Rows are shuffled with {@link #seed()}; without a
 * {@link #testSplit()} the first {@link #testSamples()} shuffled rows are held out for testing.
 * Inputs (and numeric targets) are scaled with statistics of the training rows, so the network sees
 * well-conditioned values while the UI and exports work in raw column units.
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@JsonPropertyOrder({ "repository", "config", "split", "inputColumns", "targetColumn", "classes", "samples",
		"testSplit", "testSamples", "seed", "inputScaling", "targetScaling" })
public record HuggingFaceTableDataset(String repository, String config, String split, List<String> inputColumns,
		String targetColumn, List<String> classes, int samples, String testSplit, int testSamples, long seed,
		DataScaling.Mode inputScaling, DataScaling.Mode targetScaling) implements DatasetSpec {

	public static final int MAX_COLUMNS = 500;

	/** Scaled training/test samples and the scaling that maps raw values to them. */
	public record Prepared(TrainingData train, TrainingData test, DataScaling scaling, int rowsRead,
			int rowsSkipped) {
	}

	private static final Map<List<Object>, Prepared> PREPARED = new ConcurrentHashMap<>();

	public HuggingFaceTableDataset {
		config = blankTo(config, "default");
		split = blankTo(split, "train");
		inputColumns = inputColumns == null ? List.of()
				: inputColumns.stream().map(column -> column == null ? "" : column.trim()).toList();
		targetColumn = targetColumn == null ? "" : targetColumn.trim();
		classes = classes == null ? List.of() : List.copyOf(classes);
		testSplit = testSplit == null || testSplit.isBlank() ? null : testSplit.trim();
		inputScaling = inputScaling == null ? DataScaling.Mode.STANDARDIZE : inputScaling;
		targetScaling = targetScaling == null ? DataScaling.Mode.STANDARDIZE : targetScaling;
	}

	private static String blankTo(String value, String fallback) {
		return value == null || value.isBlank() ? fallback : value.trim();
	}

	@Override
	public int inputSize() {
		return inputColumns.size();
	}

	@Override
	public int outputSize() {
		return isCategorical() ? classes.size() : 1;
	}

	@Override
	public List<String> inputNames() {
		return inputColumns;
	}

	@Override
	public List<String> outputNames() {
		return isCategorical() ? classes : List.of(targetColumn);
	}

	/** True when the target column holds class names/indices listed in {@link #classes()}. */
	@JsonIgnore
	public boolean isCategorical() {
		return !classes.isEmpty();
	}

	/** Target scaling actually applied: never for categorical targets. */
	@JsonIgnore
	public DataScaling.Mode effectiveTargetScaling() {
		return isCategorical() ? DataScaling.Mode.NONE : targetScaling;
	}

	@Override
	public void validate(List<String> errors) {
		if (repository == null || !HuggingFaceClient.REPOSITORY.matcher(repository).matches()) {
			errors.add("dataset.repository must look like 'owner/name', e.g. scikit-learn/iris");
		}
		for (var entry : Map.of("config", config, "split", split).entrySet()) {
			if (!validName(entry.getValue())) {
				errors.add("dataset." + entry.getKey() + " may only use letters, digits, '_', '.' and '-'");
			}
		}
		if (testSplit != null && !validName(testSplit)) {
			errors.add("dataset.testSplit may only use letters, digits, '_', '.' and '-'");
		}
		if (testSplit != null && testSplit.equals(split)) {
			errors.add("dataset.testSplit must differ from dataset.split (leave it empty to hold out rows instead)");
		}
		if (inputColumns.isEmpty() || inputColumns.size() > MAX_COLUMNS) {
			errors.add("dataset.inputColumns needs between 1 and " + MAX_COLUMNS + " columns");
		}
		if (inputColumns.stream().anyMatch(String::isEmpty) || new HashSet<>(inputColumns).size() != inputColumns
				.size()) {
			errors.add("dataset.inputColumns must be unique and non-blank");
		}
		if (targetColumn.isEmpty()) {
			errors.add("dataset.targetColumn is required");
		} else if (inputColumns.contains(targetColumn)) {
			errors.add("dataset.targetColumn '" + targetColumn + "' cannot also be an input column");
		}
		if (isCategorical()) {
			if (classes.size() < 2 || classes.size() > HuggingFaceDataset.MAX_CLASSES) {
				errors.add("dataset.classes needs between 2 and " + HuggingFaceDataset.MAX_CLASSES
						+ " class names (or none for a numeric target)");
			}
			if (classes.stream().anyMatch(name -> name == null || name.isBlank())
					|| new HashSet<>(classes).size() != classes.size()) {
				errors.add("dataset.classes must be unique and non-blank");
			}
		}
		if (samples <= 0 || samples > MAX_SAMPLES) {
			errors.add("dataset.samples must be between 1 and " + MAX_SAMPLES);
		}
		if (testSamples < 0 || testSamples > MAX_SAMPLES) {
			errors.add("dataset.testSamples must be between 0 and " + MAX_SAMPLES);
		}
	}

	private static boolean validName(String name) {
		return HuggingFaceClient.NAME.matcher(name).matches() && !name.startsWith(".");
	}

	@JsonIgnore
	public boolean hasTestSplit() {
		return testSplit != null && testSamples > 0;
	}

	/** True when every configured split is in the default cache. */
	@JsonIgnore
	public boolean isDownloaded() {
		final var cache = DatasetCache.openDefault();
		return cache.cachedTable(repository, config, split, List.of()).isPresent()
				&& (!hasTestSplit() || cache.cachedTable(repository, config, testSplit, List.of()).isPresent());
	}

	/** Downloads (if needed) the Parquet shards of the configured splits into the default cache. */
	public void fetch(HuggingFaceClient.Progress progress) throws java.io.IOException, InterruptedException {
		final var cache = DatasetCache.openDefault();
		cache.fetchTable(repository, config, split, progress);
		if (hasTestSplit()) {
			cache.fetchTable(repository, config, testSplit, progress);
		}
	}

	/**
	 * Reads, shuffles, splits and scales the cached rows (memoized). Throws
	 * {@link HuggingFaceDataset.NotDownloadedException} when the data is not cached.
	 */
	public Prepared prepare() {
		final List<Object> key = List.of(this, DatasetCache.defaultDirectory());
		final var known = PREPARED.get(key);
		if (known != null) {
			return known;
		}
		final var cache = DatasetCache.openDefault();
		final var columns = new ArrayList<>(inputColumns);
		columns.add(targetColumn);
		final var trainTable = table(cache, split, columns);
		final var skipped = new int[1];
		final var trainRows = rows(trainTable, skipped);
		final List<float[][]> testRows;
		final List<float[][]> fitRows;
		if (testSplit != null) {
			fitRows = trainRows.subList(0, Math.min(samples, trainRows.size()));
			testRows = hasTestSplit() ? rows(table(cache, testSplit, columns), skipped) : List.of();
		} else {
			final int held = Math.min(testSamples, Math.max(0, trainRows.size() - 1));
			testRows = trainRows.subList(0, held);
			fitRows = trainRows.subList(held, Math.min(trainRows.size(), held + samples));
		}
		if (fitRows.isEmpty()) {
			throw new IllegalStateException("No usable rows in " + repository + " " + config + "/" + split
					+ " (inputs must be numeric and the target " + (isCategorical() ? "one of the classes" : "numeric")
					+ "; " + skipped[0] + " rows skipped)");
		}
		final float[][] rawInputs = fitRows.stream().map(row -> row[0]).toArray(float[][]::new);
		final float[][] rawTargets = fitRows.stream().map(row -> row[1]).toArray(float[][]::new);
		final float[][] in = DataScaling.fit(rawInputs, inputSize(), inputScaling);
		final float[][] out = DataScaling.fit(rawTargets, outputSize(), effectiveTargetScaling());
		final var scaling = new DataScaling(in[0], in[1], out[0], out[1]);
		final var testCount = Math.min(testRows.size(), testSplit != null ? testSamples : testRows.size());
		final var prepared = new Prepared(scaled(fitRows, scaling),
				testCount == 0 ? null : scaled(testRows.subList(0, testCount), scaling), scaling,
				trainTable.rows().size(), skipped[0]);
		if (PREPARED.size() > 16) {
			PREPARED.clear();
		}
		PREPARED.put(key, prepared);
		return prepared;
	}

	private DatasetCache.Table table(DatasetCache cache, String name, List<String> columns) {
		return cache.cachedTable(repository, config, name, columns).orElseThrow(
				() -> new HuggingFaceDataset.NotDownloadedException("Rows of " + repository + " " + config + "/" + name
						+ " are not downloaded yet (they are fetched from Hugging Face into "
						+ DatasetCache.defaultDirectory() + ")"));
	}

	/** Usable rows as {raw inputs, raw target}, in seeded random order. */
	private List<float[][]> rows(DatasetCache.Table table, int[] skipped) {
		final var rows = new ArrayList<float[][]>();
		for (Object[] values : table.rows()) {
			final float[] inputs = new float[inputSize()];
			boolean usable = true;
			for (int c = 0; c < inputs.length && usable; c++) {
				final Double number = number(values[c]);
				usable = number != null;
				inputs[c] = usable ? number.floatValue() : 0;
			}
			final float[] target = usable ? target(values[inputs.length]) : null;
			if (target == null) {
				skipped[0]++;
				continue;
			}
			rows.add(new float[][] { inputs, target });
		}
		Collections.shuffle(rows, new Random(seed));
		return rows;
	}

	private float[] target(Object value) {
		if (isCategorical()) {
			final int index = classIndex(value, classes);
			if (index < 0) {
				return null;
			}
			final float[] oneHot = new float[classes.size()];
			oneHot[index] = 1;
			return oneHot;
		}
		final Double number = number(value);
		return number == null ? null : new float[] { number.floatValue() };
	}

	/** Finite number of a cell (booleans as 0/1), or null. */
	static Double number(Object value) {
		final double number;
		if (value instanceof Number n) {
			number = n.doubleValue();
		} else if (value instanceof Boolean b) {
			number = b ? 1 : 0;
		} else {
			return null;
		}
		return Double.isFinite(number) && Math.abs(number) <= Float.MAX_VALUE ? number : null;
	}

	/** Class of a cell: its text among {@code classes}, else an integral value used as an index. */
	public static int classIndex(Object value, List<String> classes) {
		if (value == null) {
			return -1;
		}
		final Double number = value instanceof String ? null : number(value);
		final String text = number != null && number == Math.rint(number) ? Long.toString(number.longValue())
				: value.toString();
		final int named = classes.indexOf(text);
		if (named >= 0) {
			return named;
		}
		return number != null && number == Math.rint(number) && number >= 0 && number < classes.size()
				? number.intValue() : -1;
	}

	private static TrainingData scaled(List<float[][]> rows, DataScaling scaling) {
		final float[][] in = new float[rows.size()][];
		final float[][] out = new float[rows.size()][];
		for (int i = 0; i < in.length; i++) {
			in[i] = scaling.scaleInput(rows.get(i)[0]);
			out[i] = scaling.scaleOutput(rows.get(i)[1]);
		}
		return new TrainingData(in, out);
	}

	@Override
	public TrainingData toTrainingData() {
		return prepare().train();
	}

	@Override
	public DataScaling scaling() {
		return prepare().scaling();
	}

	/** Bounds of the scaled training inputs (what the network sees). */
	@Override
	public float[][] inputBounds() {
		final float[][] inputs = prepare().train().getInput();
		final float[][] bounds = new float[inputSize()][];
		for (int c = 0; c < inputSize(); c++) {
			float min = Float.POSITIVE_INFINITY;
			float max = Float.NEGATIVE_INFINITY;
			for (float[] row : inputs) {
				min = Math.min(min, row[c]);
				max = Math.max(max, row[c]);
			}
			if (!(min < max)) {
				min -= 0.5f;
				max += 0.5f;
			}
			bounds[c] = new float[] { min, max };
		}
		return bounds;
	}
}
