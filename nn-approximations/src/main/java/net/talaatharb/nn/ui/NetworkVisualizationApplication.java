package net.talaatharb.nn.ui;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Slider;
import javafx.scene.control.SplitPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;
import javafx.stage.Stage;
import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.algorithms.TrainingListener;
import net.talaatharb.nn.algorithms.TrainingUpdate;
import net.talaatharb.nn.examples.ExampleProblem;
import net.talaatharb.nn.metrics.MeanSquareError;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.visualization.NetworkSnapshot;

public class NetworkVisualizationApplication extends Application {

	private final ComboBox<ExampleProblem> examples = new ComboBox<>();
	private final TextField rate = new TextField();
	private final TextField epochs = new TextField("5000");
	private final TextField threshold = new TextField();
	private final TextField input = new TextField("0, 1");
	private final Button train = new Button("Train");
	private final Button pause = new Button("Pause");
	private final Button step = new Button("Single update");
	private final Button stop = new Button("Stop");
	private final Button reset = new Button("Reset");
	private final Button infer = new Button("Infer");
	private final Label status = new Label("Ready");
	private final Label output = new Label();
	private final Label layerLabel = new Label();
	private final Slider layer = new Slider(0, 2, 2);
	private final CheckBox before = new CheckBox("Show before update");
	private final TextArea results = new TextArea();
	private final NetworkView graph = new NetworkView();
	private final XYChart.Series<Number, Number> losses = new XYChart.Series<>();
	private final XYChart.Series<Number, Number> targets = new XYChart.Series<>();
	private final XYChart.Series<Number, Number> predictions = new XYChart.Series<>();
	private final AtomicReference<TrainingUpdate> latestUpdate = new AtomicReference<>();
	private final AtomicReference<EpochState> latestEpoch = new AtomicReference<>();
	private final AtomicBoolean stopping = new AtomicBoolean();
	private final Object control = new Object();

	private FeedForwardNN network;
	private NetworkSnapshot displayed;
	private TrainingUpdate lastUpdate;
	private Thread worker;
	private AnimationTimer timer;
	private boolean active;
	private boolean paused;
	private int permits;
	private volatile int delayMillis = 15;
	private int epochOffset;
	private int lastEpoch;
	private String progress = "";

	private record EpochState(int epoch, float loss, List<Float> predictions) { }

	@Override
	public void start(Stage stage) {
		examples.getItems().setAll(ExampleProblem.values());
		examples.setValue(ExampleProblem.XOR);
		rate.setPrefColumnCount(5);
		epochs.setPrefColumnCount(5);
		threshold.setPrefColumnCount(6);
		input.setPrefColumnCount(10);
		results.setEditable(false);
		results.setPrefRowCount(8);
		results.setStyle("-fx-font-family: monospace;");
		output.setWrapText(true);
		status.setWrapText(true);

		final var speed = new Slider(0, 200, delayMillis);
		speed.setPrefWidth(160);
		final var speedLabel = new Label("15 ms/update");
		speed.valueProperty().addListener((observable, oldValue, value) -> {
			delayMillis = value.intValue();
			speedLabel.setText(delayMillis + " ms/update");
		});
		final var settings = new HBox(10, new Label("Example"), examples, new Label("Rate"), rate,
				new Label("Max epochs"), epochs, new Label("Target MSE"), threshold);
		final var buttons = new HBox(10, train, pause, step, stop, reset, speed, speedLabel, before);
		final var header = new VBox(10, settings, buttons, status);
		header.setPadding(new Insets(12));

		layer.setMajorTickUnit(1);
		layer.setMinorTickCount(0);
		layer.setSnapToTicks(true);
		layer.setShowTickLabels(true);
		layer.setShowTickMarks(true);
		layer.valueProperty().addListener((observable, oldValue, value) -> render());
		final var next = new Button("Next layer");
		next.setOnAction(event -> layer.setValue(Math.min(layer.getMax(), layer.getValue() + 1)));
		final var all = new Button("All layers");
		all.setOnAction(event -> layer.setValue(layer.getMax()));
		final var inference = new HBox(10, new Label("Input (comma separated)"), input, infer);
		final var graphPane = new VBox(8, graph,
				new Label("Blue = positive; red = negative. Edge width = |weight|. Hover for bias, z and weight changes."),
				new HBox(10, layerLabel, layer, next, all), inference, output);
		VBox.setVgrow(graph, Priority.ALWAYS);
		graphPane.setPadding(new Insets(12));

		final var lossChart = chart("Epoch", "Dataset MSE", "Training loss");
		lossChart.setLegendVisible(false);
		lossChart.getData().add(losses);
		final var functionChart = chart("x (logic: sample index)", "Output", "Target vs prediction");
		targets.setName("Target");
		predictions.setName("Prediction");
		functionChart.getData().add(targets);
		functionChart.getData().add(predictions);
		final var charts = new VBox(8, lossChart, functionChart, results);
		VBox.setVgrow(lossChart, Priority.ALWAYS);
		VBox.setVgrow(functionChart, Priority.ALWAYS);
		charts.setPadding(new Insets(12));
		charts.setMinWidth(350);
		final var split = new SplitPane(graphPane, charts);
		split.setDividerPositions(0.57);
		final var root = new BorderPane(split, header, null, null, null);

		train.setOnAction(event -> startTraining(false));
		step.setOnAction(event -> {
			if (!active) {
				startTraining(true);
			} else {
				synchronized (control) {
					paused = true;
					permits++;
					control.notifyAll();
				}
				pause.setText("Resume");
				status.setText("Single update requested");
			}
		});
		pause.setOnAction(event -> {
			synchronized (control) {
				paused = !paused;
				permits = 0;
				control.notifyAll();
				pause.setText(paused ? "Resume" : "Pause");
				status.setText(paused ? "Paused after current update" : "Training");
			}
		});
		stop.setOnAction(event -> requestStop());
		reset.setOnAction(event -> resetExample());
		examples.setOnAction(event -> resetExample());
		infer.setOnAction(event -> infer());
		input.setOnAction(event -> {
			if (!active) {
				infer();
			}
		});
		before.selectedProperty().addListener((observable, oldValue, value) -> showLastUpdate());
		resetExample();

		timer = new AnimationTimer() {
			private long lastFrame;

			@Override
			public void handle(long now) {
				if (now - lastFrame >= 33_000_000) {
					flushUpdates();
					lastFrame = now;
				}
			}
		};
		timer.start();
		stage.setTitle("Neural network training and inference");
		stage.setScene(new Scene(root, 1280, 850));
		stage.show();
	}

	private static LineChart<Number, Number> chart(String xLabel, String yLabel, String title) {
		final var x = new NumberAxis();
		final var y = new NumberAxis();
		x.setLabel(xLabel);
		y.setLabel(yLabel);
		x.setForceZeroInRange(false);
		final var chart = new LineChart<Number, Number>(x, y);
		chart.setTitle(title);
		chart.setAnimated(false);
		chart.setCreateSymbols(false);
		chart.setMinHeight(180);
		chart.setPrefHeight(240);
		return chart;
	}

	private void resetExample() {
		final var example = examples.getValue();
		network = example.newNetwork();
		rate.setText(Float.toString(example.learningRate()));
		threshold.setText(Float.toString(example.threshold()));
		input.setText(example.isLogic() ? "0, 1" : "0.5");
		layer.setMax(network.getDepth());
		layer.setValue(network.getDepth());
		lastUpdate = null;
		latestUpdate.set(null);
		latestEpoch.set(null);
		lastEpoch = 0;
		losses.getData().clear();
		progress = "";
		setActive(false);
		updatePredictions(predict(example));
		infer();
		status.setText("Ready. Reset uses seed 42. Train continues from the current weights.");
	}

	private void startTraining(boolean singleStep) {
		final float learningRate;
		final float target;
		final int maxEpochs;
		try {
			learningRate = Float.parseFloat(rate.getText().trim());
			target = Float.parseFloat(threshold.getText().trim());
			maxEpochs = Integer.parseInt(epochs.getText().trim());
			if (!Float.isFinite(learningRate) || learningRate <= 0
					|| !Float.isFinite(target) || target < 0 || maxEpochs <= 0) {
				throw new IllegalArgumentException("Use a positive finite rate, positive epoch count and non-negative finite MSE.");
			}
		} catch (IllegalArgumentException exception) {
			showError("Invalid training settings", exception);
			return;
		}
		final var example = examples.getValue();
		final var data = example.data();
		epochOffset = lastEpoch;
		stopping.set(false);
		synchronized (control) {
			paused = singleStep;
			permits = singleStep ? 1 : 0;
		}
		pause.setText(singleStep ? "Resume" : "Pause");
		setActive(true);
		status.setText(singleStep ? "Single update requested" : "Training");
		final var task = new Task<Float>() {
			@Override
			protected Float call() {
				final var listener = new TrainingListener() {
					@Override
					public void beforeUpdate(int epoch, long step, int sampleIndex) {
						awaitPermission();
					}

					@Override
					public void onUpdate(TrainingUpdate update) {
						latestUpdate.set(update);
					}

					@Override
					public void onEpoch(int epoch, float loss) {
						latestEpoch.set(new EpochState(epochOffset + epoch, loss, predict(example)));
					}
				};
				return new StochasticGradientDescent(learningRate, new Random(42)).train(
						network, data, target, maxEpochs, new MeanSquareError(), listener, stopping::get);
			}
		};
		task.setOnSucceeded(event -> finish(String.format(Locale.ROOT,
				"%s: MSE %.6f", task.getValue() <= target ? "Target reached" : "Epoch limit reached", task.getValue())));
		task.setOnFailed(event -> {
			final var error = task.getException();
			if (error instanceof CancellationException) {
				finish("Stopped; completed weight updates retained");
			} else {
				finish("Training failed");
				showError("Training failed", error);
			}
		});
		worker = new Thread(task, "network-training");
		worker.setDaemon(true);
		worker.start();
	}

	private void awaitPermission() {
		try {
			synchronized (control) {
				while (paused && permits == 0 && !stopping.get()) {
					control.wait();
				}
				if (permits > 0) {
					permits--;
				}
			}
			if (!stopping.get() && delayMillis > 0) {
				Thread.sleep(delayMillis);
			}
		} catch (InterruptedException exception) {
			Thread.currentThread().interrupt();
			throw new CancellationException("Training stopped");
		}
	}

	private void requestStop() {
		stopping.set(true);
		synchronized (control) {
			control.notifyAll();
		}
		if (worker != null) {
			worker.interrupt();
		}
		status.setText("Stopping...");
	}

	private void finish(String message) {
		flushUpdates();
		worker = null;
		setActive(false);
		updatePredictions(predict(examples.getValue()));
		status.setText(message);
	}

	private void setActive(boolean value) {
		active = value;
		train.setDisable(value);
		reset.setDisable(value);
		examples.setDisable(value);
		rate.setDisable(value);
		epochs.setDisable(value);
		threshold.setDisable(value);
		input.setDisable(value);
		infer.setDisable(value);
		pause.setDisable(!value);
		stop.setDisable(!value);
	}

	private void flushUpdates() {
		final var update = latestUpdate.getAndSet(null);
		if (update != null) {
			lastUpdate = update;
			progress = String.format(Locale.ROOT,
					"Epoch %d, update %d, sample %d | sample MSE %.6f -> %.6f | target %s",
					epochOffset + update.epoch(), update.step(), update.sampleIndex(),
					update.lossBefore(), update.lossAfter(), update.expected());
			showLastUpdate();
		}
		final var epoch = latestEpoch.getAndSet(null);
		if (epoch != null) {
			lastEpoch = epoch.epoch();
			losses.getData().add(new XYChart.Data<>(epoch.epoch(), epoch.loss()));
			if (losses.getData().size() > 600) {
				losses.getData().remove(0);
			}
			updatePredictions(epoch.predictions());
		}
	}

	private void showLastUpdate() {
		if (lastUpdate != null) {
			displayed = before.isSelected() ? lastUpdate.before() : lastUpdate.after();
			output.setText(progress + "\nDisplayed " + (before.isSelected() ? "before" : "after")
					+ " update output: " + displayed.output());
			render();
		}
	}

	private void infer() {
		try {
			final var pieces = input.getText().trim().split("[,\\s]+");
			final var values = new float[pieces.length];
			for (int i = 0; i < pieces.length; i++) {
				values[i] = Float.parseFloat(pieces[i]);
				if (!Float.isFinite(values[i])) {
					throw new IllegalArgumentException("Inputs must be finite");
				}
			}
			displayed = network.inspect(values);
			lastUpdate = null;
			before.setSelected(false);
			output.setText("Input " + Arrays.toString(values) + " -> output " + displayed.output()
					+ (examples.getValue().isLogic() ? " (classify at 0.5)" : ""));
			render();
		} catch (IllegalArgumentException exception) {
			showError("Invalid inference input", exception);
		}
	}

	private void render() {
		if (displayed != null) {
			int index = (int) Math.round(layer.getValue());
			layerLabel.setText("Reveal through layer " + index);
			graph.show(displayed, lastUpdate != null && !before.isSelected() ? lastUpdate.before() : null, index);
		}
	}

	private List<Float> predict(ExampleProblem example) {
		final var values = new ArrayList<Float>();
		for (float[] row : example.data().getInput()) {
			values.add(network.apply(row)[0]);
		}
		return List.copyOf(values);
	}

	private void updatePredictions(List<Float> values) {
		final var example = examples.getValue();
		final var data = example.data();
		targets.getData().clear();
		predictions.getData().clear();
		final var text = new StringBuilder("Input              Target   Predicted");
		if (example.isLogic()) {
			text.append("   Class");
		}
		text.append('\n');
		for (int i = 0; i < values.size(); i++) {
			float x = example.isLogic() ? i : data.getInput()[i][0];
			float expected = data.getOutput()[i][0];
			targets.getData().add(new XYChart.Data<>(x, expected));
			predictions.getData().add(new XYChart.Data<>(x, values.get(i)));
			text.append(String.format(Locale.ROOT, "%-18s %.3f    % .4f", Arrays.toString(data.getInput()[i]),
					expected, values.get(i)));
			if (example.isLogic()) {
				text.append(values.get(i) >= 0.5f ? "      1" : "      0");
			}
			text.append('\n');
		}
		results.setText(text.toString());
	}

	private void showError(String title, Throwable error) {
		status.setText(title + ": " + error.getMessage());
		final var alert = new Alert(Alert.AlertType.ERROR);
		alert.setHeaderText(title);
		alert.setContentText(error.getMessage() == null ? error.toString() : error.getMessage());
		alert.show();
	}

	@Override
	public void stop() {
		if (timer != null) {
			timer.stop();
		}
		requestStop();
	}
}
