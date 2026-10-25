package net.talaatharb.nn.ui;

import java.util.Locale;

import javafx.scene.control.Label;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.Pane;
import javafx.scene.paint.Color;
import javafx.scene.shape.Circle;
import javafx.scene.shape.Line;
import net.talaatharb.nn.visualization.NetworkSnapshot;

/** A renderer only: it never reads or mutates a live network. */
public class NetworkView extends Pane {

	private NetworkSnapshot snapshot;
	private NetworkSnapshot previous;
	private int visibleLayer = Integer.MAX_VALUE;

	public NetworkView() {
		setMinSize(420, 400);
		setPrefSize(680, 580);
	}

	public void show(NetworkSnapshot snapshot, NetworkSnapshot previous, int visibleLayer) {
		this.snapshot = snapshot;
		this.previous = previous;
		this.visibleLayer = visibleLayer;
		requestLayout();
	}

	@Override
	protected void layoutChildren() {
		getChildren().clear();
		if (snapshot == null) {
			return;
		}
		for (int e = 0; e < snapshot.connections().size(); e++) {
			final var edge = snapshot.connections().get(e);
			if (edge.sourceLayer() > visibleLayer || edge.targetLayer() > visibleLayer) {
				continue;
			}
			final var line = new Line(x(edge.sourceLayer()), y(edge.sourceLayer(), edge.sourceUnit()),
					x(edge.targetLayer()), y(edge.targetLayer(), edge.targetUnit()));
			line.setStroke(edge.weight() >= 0 ? Color.STEELBLUE : Color.INDIANRED);
			line.setStrokeWidth(0.5 + Math.min(4, Math.abs(edge.weight())));
			line.setOpacity(0.7);
			float delta = previous == null ? 0 : edge.weight() - previous.connections().get(e).weight();
			Tooltip.install(line, new Tooltip(String.format(Locale.ROOT,
					"L%d[%d] -> L%d[%d]%nweight = %.6f%nchange = %+.6f",
					edge.sourceLayer(), edge.sourceUnit(), edge.targetLayer(), edge.targetUnit(),
					edge.weight(), delta)));
			getChildren().add(line);
		}
		for (int l = 0; l < snapshot.layers().size(); l++) {
			final var layer = snapshot.layers().get(l);
			final var title = new Label(layer.label().replace(" / ", "\n"));
			title.setStyle("-fx-font-weight: bold;");
			title.relocate(x(l) - 65, 8);
			getChildren().add(title);
			title.autosize();
			for (int n = 0; n < layer.units().size(); n++) {
				final var unit = layer.units().get(n);
				boolean visible = l <= visibleLayer;
				final var circle = new Circle(x(l), y(l, n), 24);
				circle.setFill(visible ? color(unit.activation()) : Color.LIGHTGRAY);
				circle.setStroke(Color.DARKSLATEGRAY);
				final var label = new Label(visible ? String.format(Locale.ROOT, "%.3f", unit.activation()) : "?");
				label.setMouseTransparent(true);
				label.setStyle("-fx-font-weight: bold; -fx-text-fill: #172b3a;");
				label.relocate(x(l) - 21, y(l, n) - 9);
				if (visible) {
					Tooltip.install(circle, new Tooltip(String.format(Locale.ROOT,
							"Layer %d, unit %d%nactivation = %.6f%npre-activation = %.6f%nbias = %.6f",
							l, n, unit.activation(), unit.preActivation(), unit.bias())));
				}
				getChildren().addAll(circle, label);
				label.autosize();
			}
		}
	}

	private double x(int layer) {
		return 85 + layer * (getWidth() - 170) / Math.max(1, snapshot.layers().size() - 1);
	}

	private double y(int layer, int unit) {
		return 65 + (unit + 1) * (getHeight() - 90) / (snapshot.layers().get(layer).units().size() + 1);
	}

	private static Color color(float activation) {
		double intensity = Math.tanh(Math.abs(activation));
		return Color.WHITE.interpolate(activation >= 0 ? Color.LIGHTSKYBLUE : Color.LIGHTSALMON, intensity);
	}
}
