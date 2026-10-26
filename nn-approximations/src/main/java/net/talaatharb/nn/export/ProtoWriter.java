package net.talaatharb.nn.export;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.function.Consumer;

/** Minimal protocol buffers (proto3 wire format) writer, enough to emit ONNX messages without a runtime library. */
final class ProtoWriter {

	private static final int VARINT = 0;
	private static final int FIXED32 = 5;
	private static final int LENGTH_DELIMITED = 2;

	private final ByteArrayOutputStream out = new ByteArrayOutputStream();

	ProtoWriter varint(int field, long value) {
		tag(field, VARINT);
		rawVarint(value);
		return this;
	}

	ProtoWriter float32(int field, float value) {
		tag(field, FIXED32);
		final int bits = Float.floatToIntBits(value);
		for (int i = 0; i < 4; i++) {
			out.write(bits >>> (8 * i) & 0xFF);
		}
		return this;
	}

	ProtoWriter string(int field, String value) {
		return bytes(field, value.getBytes(StandardCharsets.UTF_8));
	}

	ProtoWriter bytes(int field, byte[] value) {
		tag(field, LENGTH_DELIMITED);
		rawVarint(value.length);
		out.writeBytes(value);
		return this;
	}

	ProtoWriter message(int field, Consumer<ProtoWriter> body) {
		final var nested = new ProtoWriter();
		body.accept(nested);
		return bytes(field, nested.toByteArray());
	}

	/** Packed little-endian floats, the layout ONNX expects in tensor raw_data. */
	static byte[] littleEndian(float[] values) {
		final var buffer = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
		for (float value : values) {
			buffer.putFloat(value);
		}
		return buffer.array();
	}

	byte[] toByteArray() {
		return out.toByteArray();
	}

	private void tag(int field, int wireType) {
		rawVarint((long) field << 3 | wireType);
	}

	private void rawVarint(long value) {
		while ((value & ~0x7FL) != 0) {
			out.write((int) (value & 0x7F | 0x80));
			value >>>= 7;
		}
		out.write((int) value);
	}
}
