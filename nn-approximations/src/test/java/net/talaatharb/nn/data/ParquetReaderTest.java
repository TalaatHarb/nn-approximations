package net.talaatharb.nn.data;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.junit.jupiter.api.Test;

class ParquetReaderTest {

	static byte[] fixture(String name) throws IOException {
		try (InputStream input = ParquetReaderTest.class.getResourceAsStream("/data/" + name)) {
			assertNotNull(input, name);
			return input.readAllBytes();
		}
	}

	@ParameterizedTest
	@ValueSource(strings = { "types-snappy-v1.parquet", "types-gzip-v2.parquet", "types-plain-v2.parquet" })
	void decodesEveryPrimitiveTypeWithNulls(String name) throws IOException {
		final var reader = new ParquetReader(fixture(name));
		assertEquals(50, reader.rowCount());
		assertEquals(List.of("id", "big", "ratio", "small", "name", "flag"),
				reader.columns().stream().map(ParquetReader.Column::path).toList());
		final var columns = reader.read(List.of("id", "big", "ratio", "small", "name", "flag"), Long.MAX_VALUE);
		for (int i = 0; i < 50; i++) {
			final boolean missing = i % 5 == 0;
			assertEquals(i, columns.get("id").get(i));
			assertEquals(missing ? null : i * 1_000_000_007L, columns.get("big").get(i), name + " big " + i);
			assertEquals(missing ? null : i / 4.0, columns.get("ratio").get(i));
			assertEquals(i * 0.5f, columns.get("small").get(i));
			assertEquals(missing ? null : "row" + i % 7, columns.get("name").get(i));
			assertEquals(i % 2 == 0, columns.get("flag").get(i));
		}
	}

	@Test
	void readsNestedImageStructsAcrossRowGroupsAndStopsEarly() throws IOException {
		final var reader = new ParquetReader(fixture("tiny-train.parquet"));
		assertEquals(90, reader.rowCount());
		assertEquals(3, reader.rowGroupCount());
		final var image = reader.column("image.bytes").orElseThrow();
		assertEquals(2, image.maxDefinitionLevel());
		final var first = reader.read(List.of("image.bytes", "label"), 40);
		assertEquals(40, first.get("label").size());
		for (int i = 0; i < 40; i++) {
			assertEquals((long) (i % 3), first.get("label").get(i));
			final byte[] png = (byte[]) first.get("image.bytes").get(i);
			assertEquals("png", DatasetCache.extension(png));
		}
		assertNull(reader.read(List.of("image.path"), 1).get("image.path").get(0));
	}

	@Test
	void rejectsUnknownColumnsAndNonParquetData() throws IOException {
		final var reader = new ParquetReader(fixture("tiny-test.parquet"));
		final var error = assertThrows(IllegalArgumentException.class, () -> reader.read(List.of("pixels"), 1));
		assertTrue(error.getMessage().contains("image.bytes"), error.getMessage());
		assertThrows(IllegalArgumentException.class, () -> new ParquetReader("not a parquet file".getBytes()));
		final byte[] truncated = fixture("tiny-test.parquet");
		truncated[truncated.length - 6] = 0x7F;
		assertThrows(IllegalArgumentException.class, () -> new ParquetReader(truncated));
	}

	@Test
	void snappyHandlesLiteralsAndOverlappingCopies() {
		// "abcd" literal, then copy length 8 at distance 4 -> "abcdabcdabcd", then literal "!"
		final byte[] block = { 13, 3 << 2, 'a', 'b', 'c', 'd', (byte) (1 | (8 - 4) << 2), 4, 0, '!' };
		assertEquals("abcdabcdabcd!", new String(Snappy.decompress(block, 0, block.length)));
		final byte[] badCopy = { 8, 1, 0 };
		assertThrows(IllegalArgumentException.class, () -> Snappy.decompress(badCopy, 0, badCopy.length));
		final byte[] truncated = { 10, 5 << 2, 'a' };
		assertThrows(IllegalArgumentException.class, () -> Snappy.decompress(truncated, 0, truncated.length));
	}
}
