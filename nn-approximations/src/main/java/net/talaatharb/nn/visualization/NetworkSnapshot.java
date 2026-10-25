package net.talaatharb.nn.visualization;

import java.util.List;

/**
 * Immutable, UI-independent graph state. Connections may run in either direction,
 * allowing future adapters for models such as restricted Boltzmann machines.
 */
public record NetworkSnapshot(String model, List<Layer> layers, List<Connection> connections) {

	public NetworkSnapshot {
		layers = List.copyOf(layers);
		connections = List.copyOf(connections);
	}

	public List<Float> output() {
		return layers.get(layers.size() - 1).units().stream().map(Unit::activation).toList();
	}

	public record Layer(String label, List<Unit> units) {
		public Layer {
			units = List.copyOf(units);
		}
	}

	public record Unit(float activation, float preActivation, float bias) {
	}

	public record Connection(int sourceLayer, int sourceUnit, int targetLayer, int targetUnit, float weight) {
	}
}
