package net.talaatharb.nn.visualization;

import static org.junit.jupiter.api.Assertions.*;

import java.util.List;

import org.junit.jupiter.api.Test;

import net.talaatharb.nn.functions.Functions;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.NeuronLayer;

class NetworkSnapshotTest {

	@Test
	void capturesInputsHiddenValuesBiasesAndConnectionsWithoutLiveReferences() {
		final var network = new FeedForwardNN(new int[] { 2, 2, 1 });
		network.setWeights(new float[][][] { { { 1, 2, 3 }, { -1, 4, 5 } }, { { 0.5f, 2, -1 } } });
		float[] input = { 1, 2 };
		final var snapshot = network.inspect(input);
		assertEquals(3, snapshot.layers().size());
		assertEquals(List.of(1.0f, 2.0f), snapshot.layers().getFirst().units().stream()
				.map(NetworkSnapshot.Unit::activation).toList());
		assertEquals(9, snapshot.layers().get(1).units().get(0).activation());
		assertEquals(13, snapshot.layers().get(1).units().get(1).preActivation());
		assertEquals(List.of(5.5f), snapshot.output());
		assertEquals(6, snapshot.connections().size());
		assertEquals(new NetworkSnapshot.Connection(0, 0, 1, 0, 2), snapshot.connections().getFirst());
		input[0] = 100;
		network.getLayers()[0].getNeurons()[0].updateBias(500);
		network.apply(input);
		assertEquals(1, snapshot.layers().get(1).units().getFirst().bias());
		assertEquals(9, snapshot.layers().get(1).units().getFirst().activation());
		assertThrows(UnsupportedOperationException.class, () -> snapshot.layers().clear());
		assertThrows(UnsupportedOperationException.class, () -> snapshot.layers().getFirst().units().clear());
		assertThrows(UnsupportedOperationException.class, () -> snapshot.connections().clear());
	}

	@Test
	void switchedActivationAndWeightConstructedLayerRemainConsistent() {
		final var layer = new NeuronLayer(new float[][] { { 0, 1, 2 } });
		assertEquals(2, layer.inputSize());
		assertArrayEquals(new float[] { 5 }, layer.apply(new float[] { 1, 2 }));
		layer.switchFunction(Functions.sigmoidFunction());
		assertSame(Functions.sigmoidFunction(), layer.getFunction());
		assertEquals(Functions.sigmoidFunction().apply(5f), layer.apply(new float[] { 1, 2 })[0]);
	}

	@Test
	void validatesShapeAndKeepsTopologyDefensive() {
		assertThrows(IllegalArgumentException.class, () -> new FeedForwardNN(new int[] { 2 }));
		assertThrows(IllegalArgumentException.class, () -> new FeedForwardNN(new int[] { 2, 0 }));
		assertThrows(IllegalArgumentException.class, () -> new FeedForwardNN(new int[] { 2, 1 }, List.of()));
		int[] shape = { 2, 1 };
		final var network = new FeedForwardNN(shape);
		shape[0] = 99;
		network.shape()[0] = 42;
		assertArrayEquals(new int[] { 2, 1 }, network.shape());
		assertThrows(IllegalArgumentException.class, () -> network.apply(new float[] { 1 }));
		assertThrows(IllegalArgumentException.class,
				() -> network.setWeights(new float[][][] { { { 1, 2 } } }));
	}
}
