package net.talaatharb.nn.ui;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.Collections;
import java.util.List;

import org.junit.jupiter.api.Test;

import net.talaatharb.nn.visualization.NetworkSnapshot;

class NetworkCanvasTest {

	@Test
	void markersFollowTheChangeNotTheWeightSign() {
		assertEquals(0.25f, change(-0.75f, -1f));
		assertEquals(-0.25f, change(0.75f, 1f));
		assertEquals(0, change(1f, 1f));
		assertEquals(0, change(1.000001f, 1f));
		assertEquals(0, change(Float.NaN, 1f));
		assertEquals(0, change(Float.POSITIVE_INFINITY, 1f));
	}

	@Test
	void markersNeedCompatibleRevealedUpdateData() {
		final var current = snapshot(2, 4, 1f);
		final var previous = snapshot(2, 4, 0.5f);
		assertEquals(0.5f, NetworkCanvas.visibleWeightChange(current, previous, 0, 1));
		assertEquals(0, NetworkCanvas.visibleWeightChange(current, null, 0, 1));
		assertEquals(0, NetworkCanvas.visibleWeightChange(current, previous, 0, 0));
		assertEquals(0, NetworkCanvas.visibleWeightChange(current, snapshot(3, 4, 0.5f), 0, 1));
		final var differentEdge = new NetworkSnapshot("test", previous.layers(),
				List.of(new NetworkSnapshot.Connection(0, 1, 1, 0, 0.5f)));
		assertEquals(0, NetworkCanvas.visibleWeightChange(current, differentEdge, 0, 1));
		final var noEdges = new NetworkSnapshot("test", previous.layers(), List.of());
		assertEquals(0, NetworkCanvas.visibleWeightChange(current, noEdges, 0, 1));
	}

	@Test
	void markersAreLimitedToSmallSparseLayerPairs() {
		assertEquals(0.5f, NetworkCanvas.visibleWeightChange(snapshot(4, 6, 1f), snapshot(4, 6, 0.5f), 0, 1));
		assertEquals(0, NetworkCanvas.visibleWeightChange(snapshot(5, 5, 1f), snapshot(5, 5, 0.5f), 0, 1));
		assertEquals(0, NetworkCanvas.visibleWeightChange(snapshot(7, 1, 1f), snapshot(7, 1, 0.5f), 0, 1));
		assertEquals(0, NetworkCanvas.visibleWeightChange(snapshot(1, 7, 1f), snapshot(1, 7, 0.5f), 0, 1));
	}

	private static float change(float weight, float previous) {
		return NetworkCanvas.visibleWeightChange(snapshot(2, 4, weight), snapshot(2, 4, previous), 0, 1);
	}

	private static NetworkSnapshot snapshot(int inputs, int outputs, float weight) {
		final var unit = new NetworkSnapshot.Unit(0, 0, 0);
		return new NetworkSnapshot("test",
				List.of(new NetworkSnapshot.Layer("input", Collections.nCopies(inputs, unit)),
						new NetworkSnapshot.Layer("output", Collections.nCopies(outputs, unit))),
				List.of(new NetworkSnapshot.Connection(0, 0, 1, 0, weight)));
	}
}
