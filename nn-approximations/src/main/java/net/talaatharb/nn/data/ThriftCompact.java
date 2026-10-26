package net.talaatharb.nn.data;

import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal, schema-less reader for the Thrift compact protocol (used by Parquet metadata). Structs
 * decode to {@code Map<Integer, Object>} keyed by field id; integers become {@link Long}, binaries
 * {@code byte[]}, lists {@link List}.
 */
final class ThriftCompact {

	private static final int MAX_DEPTH = 64;

	private final byte[] data;
	private int position;

	ThriftCompact(byte[] data, int offset) {
		this.data = data;
		this.position = offset;
	}

	int position() {
		return position;
	}

	Map<Integer, Object> readStruct() {
		return readStruct(0);
	}

	private Map<Integer, Object> readStruct(int depth) {
		if (depth > MAX_DEPTH) {
			throw new IllegalArgumentException("Thrift structure nested too deeply");
		}
		final var fields = new HashMap<Integer, Object>();
		int lastId = 0;
		while (true) {
			final int header = readByte();
			if (header == 0) {
				return fields;
			}
			final int type = header & 0x0F;
			final int delta = header >>> 4;
			final int id = delta == 0 ? (int) zigzag(readVarint()) : lastId + delta;
			lastId = id;
			final Object value = switch (type) {
				case 1 -> Boolean.TRUE;
				case 2 -> Boolean.FALSE;
				default -> readValue(type, depth);
			};
			fields.put(id, value);
		}
	}

	private Object readValue(int type, int depth) {
		return switch (type) {
			case 1, 2 -> readByte() == 1;
			case 3 -> (long) (byte) readByte();
			case 4, 5, 6 -> zigzag(readVarint());
			case 7 -> {
				long bits = 0;
				for (int i = 0; i < 8; i++) {
					bits |= (long) readByte() << (8 * i);
				}
				yield Double.longBitsToDouble(bits);
			}
			case 8 -> {
				final long length = readVarint();
				if (length < 0 || position + length > data.length) {
					throw new IllegalArgumentException("Thrift binary exceeds the buffer");
				}
				final byte[] bytes = new byte[(int) length];
				System.arraycopy(data, position, bytes, 0, bytes.length);
				position += bytes.length;
				yield bytes;
			}
			case 9, 10 -> {
				final int header = readByte();
				long size = header >>> 4;
				if (size == 15) {
					size = readVarint();
				}
				checkSize(size);
				final var list = new ArrayList<Object>((int) size);
				for (int i = 0; i < size; i++) {
					list.add(readValue(header & 0x0F, depth + 1));
				}
				yield list;
			}
			case 11 -> {
				final long size = readVarint();
				checkSize(size);
				final var map = new HashMap<Object, Object>();
				if (size > 0) {
					final int types = readByte();
					for (int i = 0; i < size; i++) {
						final Object key = readValue(types >>> 4, depth + 1);
						map.put(key instanceof byte[] bytes ? new String(bytes, StandardCharsets.UTF_8) : key,
								readValue(types & 0x0F, depth + 1));
					}
				}
				yield map;
			}
			case 12 -> readStruct(depth + 1);
			default -> throw new IllegalArgumentException("Unknown Thrift compact type " + type);
		};
	}

	private void checkSize(long size) {
		if (size < 0 || size > data.length - position) {
			throw new IllegalArgumentException("Thrift collection size " + size + " exceeds the buffer");
		}
	}

	private int readByte() {
		if (position >= data.length) {
			throw new IllegalArgumentException("Unexpected end of Thrift data");
		}
		return data[position++] & 0xFF;
	}

	private long readVarint() {
		long result = 0;
		for (int shift = 0; shift < 64; shift += 7) {
			final int b = readByte();
			result |= (long) (b & 0x7F) << shift;
			if ((b & 0x80) == 0) {
				return result;
			}
		}
		throw new IllegalArgumentException("Thrift varint too long");
	}

	private static long zigzag(long value) {
		return (value >>> 1) ^ -(value & 1);
	}

	static long longField(Map<Integer, Object> struct, int id, long fallback) {
		return struct.get(id) instanceof Long value ? value : fallback;
	}

	static int intField(Map<Integer, Object> struct, int id, int fallback) {
		return (int) longField(struct, id, fallback);
	}

	static String stringField(Map<Integer, Object> struct, int id) {
		return struct.get(id) instanceof byte[] bytes ? new String(bytes, StandardCharsets.UTF_8) : null;
	}

	@SuppressWarnings("unchecked")
	static Map<Integer, Object> structField(Map<Integer, Object> struct, int id) {
		return struct.get(id) instanceof Map<?, ?> map ? (Map<Integer, Object>) map : null;
	}

	static List<?> listField(Map<Integer, Object> struct, int id) {
		return struct.get(id) instanceof List<?> list ? list : List.of();
	}
}
