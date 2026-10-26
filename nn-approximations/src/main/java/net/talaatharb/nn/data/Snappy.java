package net.talaatharb.nn.data;

/** Pure-Java decoder for the raw (block) Snappy format used inside Parquet pages. */
public final class Snappy {

	private Snappy() {
	}

	public static byte[] decompress(byte[] input, int offset, int length) {
		final int end = offset + length;
		int position = offset;
		long size = 0;
		int shift = 0;
		while (true) {
			if (position >= end || shift > 28) {
				throw new IllegalArgumentException("Corrupt Snappy header");
			}
			final int b = input[position++] & 0xFF;
			size |= (long) (b & 0x7F) << shift;
			if ((b & 0x80) == 0) {
				break;
			}
			shift += 7;
		}
		if (size > Integer.MAX_VALUE - 8) {
			throw new IllegalArgumentException("Snappy block too large");
		}
		final byte[] output = new byte[(int) size];
		int out = 0;
		try {
			while (position < end) {
				final int tag = input[position++] & 0xFF;
				switch (tag & 3) {
					case 0 -> {
						int literal = tag >>> 2;
						if (literal >= 60) {
							final int bytes = literal - 59;
							literal = 0;
							for (int i = 0; i < bytes; i++) {
								literal |= (input[position++] & 0xFF) << (8 * i);
							}
						}
						literal++;
						if (literal <= 0 || position + literal > end || out + literal > output.length) {
							throw new IllegalArgumentException("Corrupt Snappy literal");
						}
						System.arraycopy(input, position, output, out, literal);
						position += literal;
						out += literal;
					}
					case 1 -> {
						final int copy = ((tag >>> 2) & 7) + 4;
						final int distance = ((tag >>> 5) << 8) | (input[position++] & 0xFF);
						out = copy(output, out, distance, copy);
					}
					case 2 -> {
						final int copy = (tag >>> 2) + 1;
						final int distance = (input[position] & 0xFF) | (input[position + 1] & 0xFF) << 8;
						position += 2;
						out = copy(output, out, distance, copy);
					}
					default -> {
						final int copy = (tag >>> 2) + 1;
						final int distance = (input[position] & 0xFF) | (input[position + 1] & 0xFF) << 8
								| (input[position + 2] & 0xFF) << 16 | (input[position + 3] & 0xFF) << 24;
						position += 4;
						out = copy(output, out, distance, copy);
					}
				}
			}
		} catch (ArrayIndexOutOfBoundsException exception) {
			throw new IllegalArgumentException("Truncated Snappy block", exception);
		}
		if (out != output.length) {
			throw new IllegalArgumentException("Snappy block shorter than its declared size");
		}
		return output;
	}

	private static int copy(byte[] output, int out, int distance, int length) {
		if (distance <= 0 || distance > out || out + length > output.length) {
			throw new IllegalArgumentException("Corrupt Snappy copy");
		}
		// Byte by byte: copies may overlap their own output (run-length style).
		for (int i = 0; i < length; i++) {
			output[out + i] = output[out - distance + i];
		}
		return out + length;
	}
}
