package net.talaatharb.nn.data;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.zip.GZIPInputStream;

/**
 * Small dependency-free Parquet reader covering what dataset exports typically use: flat or nested
 * (non-repeated) columns, data pages v1/v2, PLAIN and dictionary encodings, and uncompressed,
 * Snappy or GZIP pages. Values decode to {@link Boolean}, {@link Integer}, {@link Long},
 * {@link Float}, {@link Double}, {@link String} (UTF-8 annotated byte arrays) or {@code byte[]};
 * missing values are {@code null}.
 */
public final class ParquetReader {

	private static final byte[] MAGIC = "PAR1".getBytes(StandardCharsets.US_ASCII);

	private static final int BOOLEAN = 0;
	private static final int INT32 = 1;
	private static final int INT64 = 2;
	private static final int INT96 = 3;
	private static final int FLOAT = 4;
	private static final int DOUBLE = 5;
	private static final int BYTE_ARRAY = 6;
	private static final int FIXED_LEN_BYTE_ARRAY = 7;

	/** A leaf column; nested group names are joined with dots, e.g. {@code image.bytes}. */
	public record Column(String path, int physicalType, int typeLength, int maxDefinitionLevel,
			int maxRepetitionLevel, boolean utf8) {
	}

	private record RowGroup(long rows, Map<String, Map<Integer, Object>> chunks) {
	}

	private final byte[] file;
	private final Map<String, Column> columns = new LinkedHashMap<>();
	private final List<RowGroup> rowGroups = new ArrayList<>();
	private final long rowCount;

	public ParquetReader(byte[] file) {
		this.file = file;
		if (file.length < 12 || !startsWith(0) || !startsWith(file.length - 4)) {
			throw new IllegalArgumentException("Not a Parquet file (missing PAR1 magic)");
		}
		final int footerLength = readInt(file, file.length - 8);
		final int footerStart = file.length - 8 - footerLength;
		if (footerLength <= 0 || footerStart < 4) {
			throw new IllegalArgumentException("Corrupt Parquet footer length");
		}
		final var metadata = new ThriftCompact(file, footerStart).readStruct();
		rowCount = ThriftCompact.longField(metadata, 3, 0);
		final var schema = ThriftCompact.listField(metadata, 2);
		if (schema.isEmpty()) {
			throw new IllegalArgumentException("Parquet file has no schema");
		}
		final int[] cursor = { 1 };
		final var root = element(schema.get(0));
		for (int i = 0; i < ThriftCompact.intField(root, 5, 0); i++) {
			walk(schema, cursor, "", 0, 0);
		}
		for (Object item : ThriftCompact.listField(metadata, 4)) {
			final var group = element(item);
			final var chunks = new LinkedHashMap<String, Map<Integer, Object>>();
			for (Object chunk : ThriftCompact.listField(group, 1)) {
				final var meta = ThriftCompact.structField(element(chunk), 3);
				if (meta == null) {
					throw new IllegalArgumentException("Column chunks stored in external files are not supported");
				}
				final var parts = new ArrayList<String>();
				for (Object part : ThriftCompact.listField(meta, 3)) {
					parts.add(new String((byte[]) part, StandardCharsets.UTF_8));
				}
				chunks.put(String.join(".", parts), meta);
			}
			rowGroups.add(new RowGroup(ThriftCompact.longField(group, 3, 0), chunks));
		}
	}

	public static ParquetReader open(Path path) {
		try {
			return new ParquetReader(Files.readAllBytes(path));
		} catch (IOException exception) {
			throw new UncheckedIOException("Could not read " + path, exception);
		}
	}

	@SuppressWarnings("unchecked")
	private static Map<Integer, Object> element(Object value) {
		if (!(value instanceof Map<?, ?> map)) {
			throw new IllegalArgumentException("Corrupt Parquet metadata");
		}
		return (Map<Integer, Object>) map;
	}

	private void walk(List<?> schema, int[] cursor, String prefix, int definition, int repetition) {
		if (cursor[0] >= schema.size()) {
			throw new IllegalArgumentException("Parquet schema is shorter than declared");
		}
		final var node = element(schema.get(cursor[0]++));
		final String name = ThriftCompact.stringField(node, 4);
		final String path = prefix.isEmpty() ? name : prefix + "." + name;
		final int repetitionType = ThriftCompact.intField(node, 3, 0);
		final int def = definition + (repetitionType != 0 ? 1 : 0);
		final int rep = repetition + (repetitionType == 2 ? 1 : 0);
		final int children = ThriftCompact.intField(node, 5, 0);
		if (children > 0) {
			for (int i = 0; i < children; i++) {
				walk(schema, cursor, path, def, rep);
			}
			return;
		}
		final var logical = ThriftCompact.structField(node, 10);
		final boolean utf8 = ThriftCompact.intField(node, 6, -1) == 0 || logical != null && logical.containsKey(1);
		columns.put(path, new Column(path, ThriftCompact.intField(node, 1, -1), ThriftCompact.intField(node, 2, 0),
				def, rep, utf8));
	}

	public long rowCount() {
		return rowCount;
	}

	public List<Column> columns() {
		return List.copyOf(columns.values());
	}

	public int rowGroupCount() {
		return rowGroups.size();
	}

	/**
	 * Reads whole row groups until at least {@code maxRows} rows (or the file) are read.
	 *
	 * @return requested column path to one value per row (trimmed to {@code maxRows})
	 */
	public Map<String, List<Object>> read(List<String> paths, long maxRows) {
		final var result = new LinkedHashMap<String, List<Object>>();
		paths.forEach(path -> result.put(path, new ArrayList<>()));
		long read = 0;
		for (int g = 0; g < rowGroups.size() && read < maxRows; g++) {
			final var group = readRowGroup(g, paths);
			paths.forEach(path -> result.get(path).addAll(group.get(path)));
			read += rowGroups.get(g).rows();
		}
		if (read > maxRows) {
			result.replaceAll((path, values) -> new ArrayList<>(values.subList(0, (int) maxRows)));
		}
		return result;
	}

	/** Values of the requested columns in one row group. */
	public Map<String, List<Object>> readRowGroup(int index, List<String> paths) {
		for (String path : paths) {
			final var column = columns.get(path);
			if (column == null) {
				throw new IllegalArgumentException("Parquet file has no column '" + path + "' (columns: "
						+ String.join(", ", columns.keySet()) + ")");
			}
			if (column.maxRepetitionLevel() > 0) {
				throw new IllegalArgumentException("Repeated column '" + path + "' is not supported");
			}
		}
		final var group = rowGroups.get(index);
		final var result = new LinkedHashMap<String, List<Object>>();
		for (String path : paths) {
			final var meta = group.chunks().get(path);
			if (meta == null) {
				throw new IllegalArgumentException("Row group " + index + " lacks column '" + path + "'");
			}
			final var values = readChunk(columns.get(path), meta);
			if (values.size() != group.rows()) {
				throw new IllegalArgumentException("Column '" + path + "' has " + values.size()
						+ " values in a row group of " + group.rows() + " rows");
			}
			result.put(path, values);
		}
		return result;
	}

	public Optional<Column> column(String path) {
		return Optional.ofNullable(columns.get(path));
	}

	private List<Object> readChunk(Column column, Map<Integer, Object> meta) {
		final int codec = ThriftCompact.intField(meta, 4, 0);
		final long numValues = ThriftCompact.longField(meta, 5, 0);
		final long dataOffset = ThriftCompact.longField(meta, 9, -1);
		final long dictionaryOffset = ThriftCompact.longField(meta, 11, -1);
		long position = dictionaryOffset > 0 && dictionaryOffset < dataOffset ? dictionaryOffset : dataOffset;
		final var values = new ArrayList<Object>((int) Math.min(numValues, 1 << 20));
		Object[] dictionary = null;
		while (values.size() < numValues) {
			if (position < 4 || position >= file.length - 8) {
				throw new IllegalArgumentException("Column '" + column.path() + "' page offset out of range");
			}
			final var reader = new ThriftCompact(file, (int) position);
			final var header = reader.readStruct();
			final int pageType = ThriftCompact.intField(header, 1, -1);
			final int uncompressedSize = ThriftCompact.intField(header, 2, 0);
			final int compressedSize = ThriftCompact.intField(header, 3, 0);
			final int start = reader.position();
			if (compressedSize < 0 || start + (long) compressedSize > file.length) {
				throw new IllegalArgumentException("Column '" + column.path() + "' page exceeds the file");
			}
			switch (pageType) {
				case 2 -> {
					final var dictionaryHeader = ThriftCompact.structField(header, 7);
					final byte[] page = decompress(codec, start, compressedSize, uncompressedSize);
					final int count = ThriftCompact.intField(dictionaryHeader, 1, 0);
					dictionary = new Object[count];
					decodePlain(column, page, new int[] { 0 }, page.length, count, dictionary, 0);
				}
				case 0 -> {
					final var dataHeader = ThriftCompact.structField(header, 5);
					final byte[] page = decompress(codec, start, compressedSize, uncompressedSize);
					final int count = ThriftCompact.intField(dataHeader, 1, 0);
					final int[] cursor = { 0 };
					int[] levels = null;
					if (column.maxDefinitionLevel() > 0) {
						final int length = readInt(page, cursor[0]);
						cursor[0] += 4;
						levels = RleHybrid.decode(page, cursor[0], cursor[0] + length,
								bitWidth(column.maxDefinitionLevel()), count);
						cursor[0] += length;
					}
					decodePage(column, page, cursor, page.length, count, levels,
							ThriftCompact.intField(dataHeader, 2, 0), dictionary, values);
				}
				case 3 -> {
					final var v2 = ThriftCompact.structField(header, 8);
					final int count = ThriftCompact.intField(v2, 1, 0);
					final int definitionLength = ThriftCompact.intField(v2, 5, 0);
					final int repetitionLength = ThriftCompact.intField(v2, 6, 0);
					final boolean compressed = !Boolean.FALSE.equals(v2.get(7));
					int[] levels = null;
					if (column.maxDefinitionLevel() > 0) {
						final int from = start + repetitionLength;
						levels = RleHybrid.decode(file, from, from + definitionLength,
								bitWidth(column.maxDefinitionLevel()), count);
					}
					final int valuesStart = start + repetitionLength + definitionLength;
					final int valuesLength = compressedSize - repetitionLength - definitionLength;
					final byte[] page = compressed
							? decompress(codec, valuesStart, valuesLength,
									uncompressedSize - repetitionLength - definitionLength)
							: java.util.Arrays.copyOfRange(file, valuesStart, valuesStart + valuesLength);
					decodePage(column, page, new int[] { 0 }, page.length, count, levels,
							ThriftCompact.intField(v2, 4, 0), dictionary, values);
				}
				default -> {
					// index pages and unknown pages carry no values
				}
			}
			position = start + (long) compressedSize;
		}
		return values;
	}

	private void decodePage(Column column, byte[] page, int[] cursor, int end, int count, int[] levels,
			int encoding, Object[] dictionary, List<Object> values) {
		int present = 0;
		if (levels == null) {
			present = count;
		} else {
			for (int level : levels) {
				if (level == column.maxDefinitionLevel()) {
					present++;
				}
			}
		}
		final Object[] decoded = new Object[present];
		switch (encoding) {
			case 0 -> decodePlain(column, page, cursor, end, present, decoded, 0);
			case 2, 8 -> {
				if (dictionary == null) {
					throw new IllegalArgumentException("Column '" + column.path() + "' uses a missing dictionary");
				}
				if (present > 0) {
					final int width = page[cursor[0]++] & 0xFF;
					final int[] indices = RleHybrid.decode(page, cursor[0], end, width, present);
					for (int i = 0; i < present; i++) {
						if (indices[i] < 0 || indices[i] >= dictionary.length) {
							throw new IllegalArgumentException("Dictionary index out of range in '" + column.path() + "'");
						}
						decoded[i] = dictionary[indices[i]];
					}
				}
			}
			case 3 -> {
				if (column.physicalType() != BOOLEAN) {
					throw new IllegalArgumentException("RLE values are only supported for booleans");
				}
				final int length = readInt(page, cursor[0]);
				final int[] bits = RleHybrid.decode(page, cursor[0] + 4, cursor[0] + 4 + length, 1, present);
				for (int i = 0; i < present; i++) {
					decoded[i] = bits[i] == 1;
				}
			}
			default -> throw new IllegalArgumentException("Parquet encoding " + encoding + " in column '"
					+ column.path() + "' is not supported (PLAIN and dictionary encodings are)");
		}
		int next = 0;
		for (int i = 0; i < count; i++) {
			values.add(levels == null || levels[i] == column.maxDefinitionLevel() ? decoded[next++] : null);
		}
	}

	private static void decodePlain(Column column, byte[] page, int[] cursor, int end, int count, Object[] target,
			int offset) {
		int p = cursor[0];
		try {
			for (int i = 0; i < count; i++) {
				final Object value;
				switch (column.physicalType()) {
					case BOOLEAN -> {
						value = (page[cursor[0] + i / 8] >> (i % 8) & 1) == 1;
						p = cursor[0] + (i + 8) / 8;
					}
					case INT32 -> {
						value = readInt(page, p);
						p += 4;
					}
					case INT64 -> {
						value = readLong(page, p);
						p += 8;
					}
					case FLOAT -> {
						value = Float.intBitsToFloat(readInt(page, p));
						p += 4;
					}
					case DOUBLE -> {
						value = Double.longBitsToDouble(readLong(page, p));
						p += 8;
					}
					case BYTE_ARRAY -> {
						final int length = readInt(page, p);
						if (length < 0 || p + 4 + length > end) {
							throw new IllegalArgumentException("Byte array exceeds the page");
						}
						value = bytes(column, page, p + 4, length);
						p += 4 + length;
					}
					case FIXED_LEN_BYTE_ARRAY, INT96 -> {
						final int length = column.physicalType() == INT96 ? 12 : column.typeLength();
						value = bytes(column, page, p, length);
						p += length;
					}
					default -> throw new IllegalArgumentException("Unsupported Parquet type " + column.physicalType());
				}
				target[offset + i] = value;
			}
		} catch (ArrayIndexOutOfBoundsException exception) {
			throw new IllegalArgumentException("Truncated values in column '" + column.path() + "'", exception);
		}
		cursor[0] = p;
	}

	private static Object bytes(Column column, byte[] page, int from, int length) {
		if (column.utf8()) {
			return new String(page, from, length, StandardCharsets.UTF_8);
		}
		final byte[] copy = new byte[length];
		System.arraycopy(page, from, copy, 0, length);
		return copy;
	}

	private byte[] decompress(int codec, int start, int length, int uncompressedSize) {
		return switch (codec) {
			case 0 -> java.util.Arrays.copyOfRange(file, start, start + length);
			case 1 -> {
				final byte[] output = Snappy.decompress(file, start, length);
				if (output.length != uncompressedSize) {
					throw new IllegalArgumentException("Snappy page size mismatch");
				}
				yield output;
			}
			case 2 -> {
				try (var input = new GZIPInputStream(new ByteArrayInputStream(file, start, length))) {
					yield input.readAllBytes();
				} catch (IOException exception) {
					throw new UncheckedIOException("Corrupt GZIP page", exception);
				}
			}
			default -> throw new IllegalArgumentException("Parquet compression codec " + codecName(codec)
					+ " is not supported (uncompressed, SNAPPY and GZIP are)");
		};
	}

	private static String codecName(int codec) {
		return switch (codec) {
			case 3 -> "LZO";
			case 4 -> "BROTLI";
			case 5, 7 -> "LZ4";
			case 6 -> "ZSTD";
			default -> Integer.toString(codec);
		};
	}

	private boolean startsWith(int offset) {
		for (int i = 0; i < MAGIC.length; i++) {
			if (file[offset + i] != MAGIC[i]) {
				return false;
			}
		}
		return true;
	}

	static int bitWidth(int maxValue) {
		return 32 - Integer.numberOfLeadingZeros(maxValue);
	}

	static int readInt(byte[] data, int offset) {
		return (data[offset] & 0xFF) | (data[offset + 1] & 0xFF) << 8 | (data[offset + 2] & 0xFF) << 16
				| (data[offset + 3] & 0xFF) << 24;
	}

	private static long readLong(byte[] data, int offset) {
		return (readInt(data, offset) & 0xFFFFFFFFL) | (long) readInt(data, offset + 4) << 32;
	}

	/** Parquet's RLE / bit-packing hybrid encoding. */
	static final class RleHybrid {
		private RleHybrid() {
		}

		static int[] decode(byte[] data, int position, int end, int bitWidth, int count) {
			final int[] values = new int[count];
			int produced = 0;
			final int byteWidth = (bitWidth + 7) / 8;
			while (produced < count) {
				if (position >= end) {
					throw new IllegalArgumentException("RLE data ended early");
				}
				long header = 0;
				for (int shift = 0;; shift += 7) {
					final int b = data[position++] & 0xFF;
					header |= (long) (b & 0x7F) << shift;
					if ((b & 0x80) == 0) {
						break;
					}
					if (shift > 28) {
						throw new IllegalArgumentException("RLE header too long");
					}
				}
				if ((header & 1) == 0) {
					final long run = header >>> 1;
					int value = 0;
					for (int i = 0; i < byteWidth; i++) {
						value |= (data[position++] & 0xFF) << (8 * i);
					}
					for (long i = 0; i < run && produced < count; i++) {
						values[produced++] = value;
					}
				} else {
					final long groups = header >>> 1;
					final long bytes = groups * bitWidth;
					if (position + bytes > data.length) {
						throw new IllegalArgumentException("Bit-packed run exceeds the data");
					}
					final long total = groups * 8;
					for (long i = 0; i < total && produced < count; i++) {
						final long bit = i * bitWidth;
						int value = 0;
						for (int b = 0; b < bitWidth; b++) {
							final long at = bit + b;
							if ((data[position + (int) (at >>> 3)] >> (at & 7) & 1) != 0) {
								value |= 1 << b;
							}
						}
						values[produced++] = value;
					}
					position += (int) bytes;
				}
			}
			return values;
		}
	}
}
