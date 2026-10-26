package net.talaatharb.nn.data;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

/**
 * Local copy of Hugging Face image classification splits: the Parquet shards plus one image file per
 * extracted sample and an {@code index.json} of {@code image path -> label} rows. Layout:
 * {@code <dir>/<owner>__<name>/<config>/<split>/{parquet/, images/, index.json}}.
 */
public final class DatasetCache {

	public static final String DIRECTORY_PROPERTY = "nn.datasets.dir";
	private static final String INDEX = "index.json";
	private static final ObjectMapper JSON = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT)
			.setSerializationInclusion(JsonInclude.Include.NON_NULL);
	private static final Object LOCK = new Object();

	/** One labelled image on disk. */
	public record ImageSample(Path image, int label) {
	}

	/** What to extract: the first {@code count} labelled rows of a split. */
	public record Request(String repository, String config, String split, String imageColumn, String labelColumn,
			List<String> classes, int count) {
		public Request {
			HuggingFaceClient.requireRepository(repository);
			Objects.requireNonNull(config);
			Objects.requireNonNull(split);
			Objects.requireNonNull(imageColumn);
			Objects.requireNonNull(labelColumn);
			classes = List.copyOf(classes);
		}

		String describe() {
			return repository + " " + config + "/" + split;
		}
	}

	record Source(String repository, String config, String split, String imageColumn, String labelColumn,
			List<String> classes) {
	}

	record Row(String image, int label) {
	}

	record Index(Source source, boolean complete, List<Row> samples) {
	}

	private final Path directory;
	private final HuggingFaceClient client;

	public DatasetCache(Path directory, HuggingFaceClient client) {
		this.directory = directory;
		this.client = client;
	}

	public static Path defaultDirectory() {
		final String configured = System.getProperty(DIRECTORY_PROPERTY);
		return configured != null && !configured.isBlank() ? Path.of(configured)
				: Path.of(System.getProperty("user.home"), ".nn-approximations", "datasets");
	}

	public static DatasetCache openDefault() {
		return new DatasetCache(defaultDirectory(), HuggingFaceClient.fromEnvironment());
	}

	public Path directory() {
		return directory;
	}

	public Path splitDirectory(String repository, String config, String split) {
		HuggingFaceClient.requireRepository(repository);
		return directory.resolve(repository.replace("/", "__")).resolve(safe(config)).resolve(safe(split));
	}

	private static String safe(String name) {
		if (!HuggingFaceClient.NAME.matcher(name).matches() || name.startsWith(".")) {
			throw new IllegalArgumentException("'" + name + "' may only use letters, digits, '_', '.', '-'");
		}
		return name;
	}

	/**
	 * Samples already on disk, without network access. Empty when fewer than requested are cached
	 * (unless the split itself is smaller).
	 */
	public Optional<List<ImageSample>> cached(Request request) {
		final Path folder = splitDirectory(request.repository(), request.config(), request.split());
		final var index = readIndex(folder);
		if (index == null || !index.source().equals(source(request))
				|| index.samples().size() < request.count() && !index.complete()) {
			return Optional.empty();
		}
		final var samples = new ArrayList<ImageSample>();
		for (var row : index.samples().subList(0, Math.min(request.count(), index.samples().size()))) {
			final Path image = folder.resolve(row.image()).normalize();
			if (!image.startsWith(folder) || !Files.isRegularFile(image)) {
				return Optional.empty();
			}
			samples.add(new ImageSample(image, row.label()));
		}
		return Optional.of(List.copyOf(samples));
	}

	/** Returns cached samples or downloads and extracts the missing ones. */
	public List<ImageSample> fetch(Request request, HuggingFaceClient.Progress progress)
			throws IOException, InterruptedException {
		synchronized (LOCK) {
			final var existing = cached(request);
			if (existing.isPresent()) {
				return existing.get();
			}
			final Path folder = splitDirectory(request.repository(), request.config(), request.split());
			progress.update("Listing Parquet files of " + request.describe(), -1);
			final List<URI> shards = client.parquetFiles(request.repository(), request.config(), request.split());
			final var rows = new ArrayList<Row>();
			boolean complete = true;
			Files.createDirectories(folder.resolve("images"));
			long rowNumber = 0;
			shards: for (int s = 0; s < shards.size(); s++) {
				final Path shard = folder.resolve("parquet").resolve(s + ".parquet");
				if (!Files.isRegularFile(shard)) {
					final int number = s + 1;
					client.download(shards.get(s), shard, (message, fraction) -> progress.update(
							"Shard " + number + "/" + shards.size() + ": " + message, fraction));
				}
				progress.update("Extracting images from " + request.describe(), -1);
				final var reader = ParquetReader.open(shard);
				final String imagePath = reader.column(request.imageColumn() + ".bytes").isPresent()
						? request.imageColumn() + ".bytes" : request.imageColumn();
				final var paths = List.of(imagePath, request.labelColumn());
				for (int g = 0; g < reader.rowGroupCount(); g++) {
					final var group = reader.readRowGroup(g, paths);
					final var images = group.get(imagePath);
					final var labels = group.get(request.labelColumn());
					for (int r = 0; r < images.size(); r++, rowNumber++) {
						final int label = label(labels.get(r), request.classes());
						if (label < 0 || !(images.get(r) instanceof byte[] bytes) || bytes.length == 0) {
							continue;
						}
						final String name = String.format(Locale.ROOT, "images/%06d.%s", rowNumber, extension(bytes));
						final Path file = folder.resolve(name);
						if (!Files.isRegularFile(file)) {
							Files.write(file, bytes);
						}
						rows.add(new Row(name, label));
						if (rows.size() >= request.count()) {
							complete = s == shards.size() - 1 && g == reader.rowGroupCount() - 1
									&& r == images.size() - 1;
							break shards;
						}
					}
					if (Thread.currentThread().isInterrupted()) {
						throw new InterruptedException("Dataset download cancelled");
					}
				}
				progress.update(String.format(Locale.ROOT, "Extracted %d of %d samples from %s", rows.size(),
						request.count(), request.describe()), (double) rows.size() / request.count());
			}
			if (rows.isEmpty()) {
				throw new IOException("No labelled images found in " + request.describe() + " (column '"
						+ request.imageColumn() + "', label '" + request.labelColumn() + "')");
			}
			writeIndex(folder, new Index(source(request), complete, rows));
			return cached(request).orElseThrow(() -> new IOException("Cache for " + request.describe()
					+ " could not be read back"));
		}
	}

	private static Source source(Request request) {
		return new Source(request.repository(), request.config(), request.split(), request.imageColumn(),
				request.labelColumn(), request.classes());
	}

	// ---------------------------------------------------------------- tables

	/** At most this many rows of a tabular split are downloaded and read. */
	public static final int MAX_TABLE_ROWS = 100_000;
	private static final String TABLE = "table.json";

	/** Which Parquet shards of a split are on disk ({@code complete} = every shard of the split). */
	record TableIndex(String repository, String config, String split, int shards, long rows, boolean complete) {
	}

	/** Rows of selected columns; {@code rows.get(r)[c]} is the value of {@code columns.get(c)} (may be null). */
	public record Table(List<String> columns, List<Object[]> rows, boolean complete) {
		public Table {
			columns = List.copyOf(columns);
			rows = List.copyOf(rows);
		}
	}

	/** Columns of a cached split, without network access; empty when the split is not downloaded. */
	public Optional<Table> cachedTable(String repository, String config, String split, List<String> columns) {
		final Path folder = splitDirectory(repository, config, split);
		final var index = readTableIndex(folder);
		if (index == null || !index.repository().equals(repository) || !index.config().equals(config)
				|| !index.split().equals(split)) {
			return Optional.empty();
		}
		final var rows = new ArrayList<Object[]>();
		for (int s = 0; s < index.shards() && rows.size() < MAX_TABLE_ROWS; s++) {
			final Path shard = folder.resolve("parquet").resolve(s + ".parquet");
			if (!Files.isRegularFile(shard)) {
				return Optional.empty();
			}
			final var reader = ParquetReader.open(shard);
			for (String column : columns) {
				if (reader.column(column).isEmpty()) {
					throw new IllegalArgumentException("Column '" + column + "' is not in " + repository + " " + config
							+ "/" + split + " (available: " + String.join(", ", reader.columns().stream()
									.map(ParquetReader.Column::path).toList()) + ")");
				}
			}
			final var values = reader.read(columns, MAX_TABLE_ROWS - rows.size());
			final int count = columns.isEmpty() ? 0 : values.get(columns.get(0)).size();
			for (int r = 0; r < count; r++) {
				final Object[] row = new Object[columns.size()];
				for (int c = 0; c < row.length; c++) {
					row[c] = values.get(columns.get(c)).get(r);
				}
				rows.add(row);
			}
		}
		return Optional.of(new Table(columns, rows, index.complete() && rows.size() < MAX_TABLE_ROWS));
	}

	/** Downloads the Parquet shards of a split (until {@link #MAX_TABLE_ROWS}) unless already cached. */
	public void fetchTable(String repository, String config, String split, HuggingFaceClient.Progress progress)
			throws IOException, InterruptedException {
		synchronized (LOCK) {
			final Path folder = splitDirectory(repository, config, split);
			final var existing = readTableIndex(folder);
			if (existing != null && existing.repository().equals(repository) && existing.config().equals(config)
					&& existing.split().equals(split) && allShards(folder, existing.shards())) {
				return;
			}
			final String name = repository + " " + config + "/" + split;
			progress.update("Listing Parquet files of " + name, -1);
			final List<URI> shards = client.parquetFiles(repository, config, split);
			Files.createDirectories(folder.resolve("parquet"));
			long rows = 0;
			int downloaded = 0;
			while (downloaded < shards.size() && rows < MAX_TABLE_ROWS) {
				final Path shard = folder.resolve("parquet").resolve(downloaded + ".parquet");
				if (!Files.isRegularFile(shard)) {
					final int number = downloaded + 1;
					client.download(shards.get(downloaded), shard, (message, fraction) -> progress.update(
							"Shard " + number + "/" + shards.size() + " of " + name + ": " + message, fraction));
				}
				rows += ParquetReader.open(shard).rowCount();
				downloaded++;
				if (Thread.currentThread().isInterrupted()) {
					throw new InterruptedException("Dataset download cancelled");
				}
			}
			final var index = new TableIndex(repository, config, split, downloaded, rows,
					downloaded == shards.size());
			writeJson(folder, TABLE, index);
		}
	}

	private static boolean allShards(Path folder, int count) {
		for (int s = 0; s < count; s++) {
			if (!Files.isRegularFile(folder.resolve("parquet").resolve(s + ".parquet"))) {
				return false;
			}
		}
		return true;
	}

	private static TableIndex readTableIndex(Path folder) {
		final Path file = folder.resolve(TABLE);
		if (!Files.isRegularFile(file)) {
			return null;
		}
		try {
			return JSON.readValue(file.toFile(), TableIndex.class);
		} catch (IOException exception) {
			return null;
		}
	}

	static int label(Object value, List<String> classes) {
		final long index;
		if (value instanceof Long number) {
			index = number;
		} else if (value instanceof Integer number) {
			index = number;
		} else if (value instanceof String text) {
			index = classes.indexOf(text);
		} else {
			return -1;
		}
		return index >= 0 && index < classes.size() ? (int) index : -1;
	}

	static String extension(byte[] bytes) {
		if (bytes.length > 3 && (bytes[0] & 0xFF) == 0x89 && bytes[1] == 'P' && bytes[2] == 'N' && bytes[3] == 'G') {
			return "png";
		}
		if (bytes.length > 2 && (bytes[0] & 0xFF) == 0xFF && (bytes[1] & 0xFF) == 0xD8) {
			return "jpg";
		}
		if (bytes.length > 2 && bytes[0] == 'G' && bytes[1] == 'I' && bytes[2] == 'F') {
			return "gif";
		}
		if (bytes.length > 1 && bytes[0] == 'B' && bytes[1] == 'M') {
			return "bmp";
		}
		return "img";
	}

	private static Index readIndex(Path folder) {
		final Path file = folder.resolve(INDEX);
		if (!Files.isRegularFile(file)) {
			return null;
		}
		try {
			return JSON.readValue(file.toFile(), Index.class);
		} catch (IOException exception) {
			return null;
		}
	}

	private static void writeIndex(Path folder, Index index) {
		writeJson(folder, INDEX, index);
	}

	private static void writeJson(Path folder, String name, Object value) {
		try {
			final Path temporary = Files.createTempFile(folder, "index", ".tmp");
			try {
				JSON.writeValue(temporary.toFile(), value);
				Files.move(temporary, folder.resolve(name), StandardCopyOption.REPLACE_EXISTING,
						StandardCopyOption.ATOMIC_MOVE);
			} finally {
				Files.deleteIfExists(temporary);
			}
		} catch (IOException exception) {
			throw new UncheckedIOException("Could not write the dataset index in " + folder, exception);
		}
	}
}
