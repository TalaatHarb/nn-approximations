package net.talaatharb.nn.ui;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import javafx.animation.AnimationTimer;
import javafx.application.Application;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.Alert;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.CheckBox;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Hyperlink;
import javafx.scene.control.Label;
import javafx.scene.control.MenuButton;
import javafx.scene.control.MenuItem;
import javafx.scene.control.Separator;
import javafx.scene.control.SeparatorMenuItem;
import javafx.scene.control.Slider;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.SplitPane;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.BorderPane;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.StackPane;
import javafx.scene.layout.VBox;
import javafx.stage.FileChooser;
import javafx.stage.Stage;
import net.talaatharb.nn.algorithms.StochasticGradientDescent;
import net.talaatharb.nn.algorithms.TrainingListener;
import net.talaatharb.nn.algorithms.TrainingUpdate;
import net.talaatharb.nn.config.DataScaling;
import net.talaatharb.nn.config.ExampleCatalog;
import net.talaatharb.nn.config.HuggingFaceDataset;
import net.talaatharb.nn.config.HuggingFaceTableDataset;
import net.talaatharb.nn.config.InitializationSpec;
import net.talaatharb.nn.config.ModelDefinition;
import net.talaatharb.nn.config.ModelJson;
import net.talaatharb.nn.data.DatasetCache;
import net.talaatharb.nn.export.OnnxExporter;
import net.talaatharb.nn.export.PmmlExporter;
import net.talaatharb.nn.metrics.MeanSquareError;
import net.talaatharb.nn.structures.FeedForwardNN;
import net.talaatharb.nn.structures.TrainingData;
import net.talaatharb.nn.tracking.MlflowClient;
import net.talaatharb.nn.tracking.MlflowRunLogger;
import net.talaatharb.nn.tracking.MlflowSettings;
import net.talaatharb.nn.tracking.RunArtifacts;
import net.talaatharb.nn.visualization.NetworkSnapshot;

public class NetworkVisualizationApplication extends Application {

	enum ExportFormat {
		CONFIGURATION("Model configuration", "Model JSON", "json", ""),
		WITH_WEIGHTS("Model with current weights", "Model JSON", "json", "-trained"),
		ONNX("ONNX model (current weights)", "ONNX", "onnx", ""),
		PMML("PMML neural network (current weights)", "PMML", "pmml", "");

		private final String title;
		private final String filter;
		private final String extension;
		private final String suffix;

		ExportFormat(String title, String filter, String extension, String suffix) {
			this.title = title;
			this.filter = filter;
			this.extension = extension;
			this.suffix = suffix;
		}
	}

	private static final int MAX_LOSS_POINTS = 800;
	private static final long SURFACE_INTERVAL_NANOS = 120_000_000L;
	/** Image models update far faster than the screen; only every ~30 ms one update is captured for display. */
	private static final long IMAGE_OBSERVE_INTERVAL_NANOS = 30_000_000L;
	private static final int MAX_RESULT_ROWS = 200;
	private static final String NUMERIC_INPUT = "Input (comma separated)";
	private static final String IMAGE_INPUT = "Sample (train:N, test:N or image file)";
	private static final String TABLE_INPUT = "Raw column values or train:N / test:N";
	/** Datasets larger than this train at full speed with throttled visual updates. */
	private static final int LARGE_DATASET = 500;
	private static final int MAX_CHART_POINTS = 400;
	private static final java.util.regex.Pattern SAMPLE_ID = java.util.regex.Pattern.compile("(train|test):(\\d+)");

	private final ComboBox<ExampleCatalog.Entry> models = new ComboBox<>();
	private final Button newModel = new Button("New");
	private final Button edit = new Button("Edit");
	private final Button duplicate = new Button("Duplicate");
	private final Button delete = new Button("Delete");
	private final Button importModel = new Button("Import");
	private final MenuButton export = new MenuButton("Export");
	private final Button viewJson = new Button("View JSON");
	private final Label summary = new Label();

	private final TextField rate = new TextField();
	private final TextField epochs = new TextField();
	private final TextField threshold = new TextField();
	private final Button train = new Button("Train");
	private final Button pause = new Button("Pause");
	private final Button step = new Button("Single update");
	private final Button stop = new Button("Stop");

	private final ComboBox<InitializationSpec.Scheme> scheme = new ComboBox<>();
	private final TextField initMin = new TextField();
	private final TextField initMax = new TextField();
	private final TextField seed = new TextField();
	private final Button randomize = new Button("Randomize");
	private final Button reset = new Button("Reset to model");
	private final Button storeWeights = new Button("Store weights in model");
	private final Button clearWeights = new Button("Clear stored weights");
	private final MenuItem pmmlExport = menuItem(ExportFormat.PMML.title + "…", () -> exportModel(ExportFormat.PMML));

	private final CheckBox trackRuns = new CheckBox("Log runs to MLflow");
	private final Button trackingSettings = new Button("MLflow settings…");
	private final Label trackingStatus = new Label();
	private final Hyperlink lastRun = new Hyperlink("Open last run");

	private final TextField input = new TextField();
	private final Button infer = new Button("Infer");
	private final Label inputLabel = new Label(NUMERIC_INPUT);
	private final Button browse = new Button("Browse…");
	private final Slider speed = new Slider(0, 200, 15);
	private final ImageClassificationView imageView = new ImageClassificationView();
	private final Label status = new Label("Ready");
	private final Label output = new Label();
	private final Label detail = new Label();
	private final Label layerLabel = new Label();
	private final Slider layer = new Slider(0, 2, 2);
	private final CheckBox before = new CheckBox("Show before update");
	private final ComboBox<NetworkCanvas.EdgeMode> edgeMode = new ComboBox<>();
	private final CheckBox logLoss = new CheckBox("log10 scale");
	private final ComboBox<String> outputSelector = new ComboBox<>();
	private final TextArea results = new TextArea();
	private final NetworkCanvas graph = new NetworkCanvas();
	private final OutputSurface surface = new OutputSurface();
	private final LineChart<Number, Number> lossChart = chart("Epoch", "Dataset MSE", "Training loss");
	private final LineChart<Number, Number> functionChart = chart("x", "Output", "Target vs prediction");
	private final XYChart.Series<Number, Number> losses = new XYChart.Series<>();
	private final XYChart.Series<Number, Number> targets = new XYChart.Series<>();
	private final XYChart.Series<Number, Number> predictions = new XYChart.Series<>();
	private final XYChart.Series<Number, Number> decision = new XYChart.Series<>();
	private final List<float[]> lossHistory = new ArrayList<>();
	private final AtomicReference<TrainingUpdate> latestUpdate = new AtomicReference<>();
	private final AtomicReference<EpochState> latestEpoch = new AtomicReference<>();
	private final AtomicBoolean stopping = new AtomicBoolean();
	private final Object control = new Object();
	private final List<javafx.scene.Node> lockedWhileTraining = new ArrayList<>();

	private Stage stage;
	private ExampleCatalog catalog;
	private ModelDefinition model;
	private TrainingData data;
	/** Held-out samples of image models (null when there is no test split). */
	private TrainingData testData;
	private List<DatasetCache.ImageSample> trainImages = List.of();
	/** Raw column values = scaling applied in reverse to what the network sees (identity except for tables). */
	private DataScaling scaling = DataScaling.identity(1, 1);
	private boolean autoFastSpeed;
	private List<DatasetCache.ImageSample> testImages = List.of();
	private List<float[]> lastTestPredictions = List.of();
	private HBox outputRow;
	private int loadGeneration;
	private FeedForwardNN network;
	private List<float[]> lastPredictions = List.of();
	private OutputSurface.Grid lastGrid;
	private NetworkSnapshot displayed;
	private TrainingUpdate lastUpdate;
	private Thread worker;
	private AnimationTimer timer;
	private boolean active;
	private boolean paused;
	private boolean switchingModels;
	private int permits;
	private volatile int delayMillis = 15;
	private volatile int selectedOutput;
	private int epochOffset;
	private int lastEpoch;
	private String progress = "";
	private String weightOrigin = "";
	private boolean continuedTraining;
	private MlflowSettings tracking;
	private MlflowRunLogger runLogger;
	private String lastRunUrl;

	private record EpochState(int epoch, float loss, List<float[]> predictions, List<float[]> testPredictions,
			OutputSurface.Grid grid) {
	}

	private record PreparedData(TrainingData train, TrainingData test, List<DatasetCache.ImageSample> trainSamples,
			List<DatasetCache.ImageSample> testSamples) {
	}

	@Override
	public void start(Stage primaryStage) {
		this.stage = primaryStage;
		catalog = ExampleCatalog.openDefault();
		configureControls();
		final var root = new BorderPane(body(), header(), null, null, null);
		wireActions();
		refreshModels(ExampleCatalog.BUILT_IN_IDS.get(0));
		if (!catalog.problems().isEmpty()) {
			status.setText("Skipped invalid model files in " + catalog.directory() + ": "
					+ String.join(" | ", catalog.problems()));
		}

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
		final var screen = javafx.stage.Screen.getPrimary().getVisualBounds();
		stage.setScene(new Scene(root, Math.min(1380, screen.getWidth() * 0.95),
				Math.min(900, screen.getHeight() * 0.92)));
		stage.show();
	}

	private void configureControls() {
		models.setPrefWidth(230);
		for (var field : List.of(rate, epochs, threshold, initMin, initMax)) {
			field.setPrefColumnCount(6);
		}
		seed.setPrefColumnCount(10);
		seed.setPromptText("random");
		seed.setTooltip(new Tooltip("Leave blank to draw a new random seed on every Randomize"));
		input.setPrefColumnCount(12);
		scheme.getItems().setAll(InitializationSpec.Scheme.values());
		scheme.valueProperty().addListener((observable, oldValue, value) -> {
			initMin.setDisable(active || value != InitializationSpec.Scheme.UNIFORM);
			initMax.setDisable(active || value != InitializationSpec.Scheme.UNIFORM);
		});
		edgeMode.getItems().setAll(NetworkCanvas.EdgeMode.values());
		edgeMode.setValue(NetworkCanvas.EdgeMode.WEIGHTS);
		edgeMode.setTooltip(new Tooltip("Colour/width edges by weight or by the change made in the last update"));
		results.setEditable(false);
		results.setPrefRowCount(8);
		results.setMinHeight(160);
		results.setStyle("-fx-font-family: monospace;");
		output.setWrapText(true);
		status.setWrapText(true);
		summary.setWrapText(true);
		summary.setStyle("-fx-text-fill: #444;");
		detail.setWrapText(true);
		detail.setMinHeight(48);
		detail.setStyle("-fx-font-family: monospace; -fx-font-size: 11px;");
		graph.detailProperty().addListener((observable, oldValue, value) -> detail
				.setText(value.isEmpty() ? "Hover a neuron for its pre-activation, bias and strongest connections." : value));
		detail.setText("Hover a neuron for its pre-activation, bias and strongest connections.");
		export.getItems().addAll(
				menuItem(ExportFormat.CONFIGURATION.title + "…", () -> exportModel(ExportFormat.CONFIGURATION)),
				menuItem(ExportFormat.WITH_WEIGHTS.title + "…", () -> exportModel(ExportFormat.WITH_WEIGHTS)),
				new SeparatorMenuItem(),
				menuItem(ExportFormat.ONNX.title + "…", () -> exportModel(ExportFormat.ONNX)), pmmlExport);
		export.setOnShowing(event -> {
			final var reason = PmmlExporter.unsupportedReason(model);
			pmmlExport.setDisable(reason.isPresent());
			pmmlExport.setText(ExportFormat.PMML.title + reason.map(text -> " - unavailable: " + text).orElse("…"));
		});
		tracking = MlflowSettings.load();
		trackRuns.setSelected(tracking.enabled());
		trackRuns.selectedProperty().addListener((observable, oldValue, value) -> {
			if (value != tracking.enabled()) {
				applyTracking(tracking.withEnabled(value));
			}
		});
		trackingSettings.setOnAction(event -> new MlflowSettingsDialog(stage, tracking).showAndWait()
				.ifPresent(this::applyTracking));
		trackingStatus.setStyle("-fx-text-fill: #444;");
		lastRun.setVisible(false);
		lastRun.setOnAction(event -> getHostServices().showDocument(lastRunUrl));
		showTrackingTarget();
		lockedWhileTraining.addAll(List.of(models, newModel, edit, duplicate, delete, importModel, export, viewJson,
				rate, epochs,
				threshold, train, scheme, initMin, initMax, seed, randomize, reset, storeWeights, clearWeights, input,
				infer, browse, outputSelector, trackRuns, trackingSettings));
		browse.setVisible(false);
		browse.setManaged(false);
		browse.setTooltip(new Tooltip("Classify an image file from disk (resized to the model's input size)"));
		imageView.setVisible(false);
		imageView.setOnSampleSelected(id -> {
			if (active) {
				status.setText("Pause is not enough: stop training to inspect " + id);
				return;
			}
			input.setText(id);
			infer();
		});
	}

	private static MenuItem menuItem(String text, Runnable action) {
		final var item = new MenuItem(text);
		item.setOnAction(event -> action.run());
		return item;
	}

	private VBox header() {
		speed.setPrefWidth(140);
		final var speedLabel = new Label("15 ms/update");
		speedLabel.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
		speed.valueProperty().addListener((observable, oldValue, value) -> {
			delayMillis = value.intValue();
			speedLabel.setText(delayMillis + " ms/update");
		});
		final var modelRow = row(new Label("Model"), models, newModel, edit, duplicate, delete, importModel, export,
				viewJson);
		final var trainingRow = row(new Label("Rate"), rate, new Label("Max epochs"), epochs, new Label("Target MSE"),
				threshold, new Separator(javafx.geometry.Orientation.VERTICAL), train, pause, step, stop, speed,
				speedLabel);
		final var weightRow = row(new Label("Weights"), scheme, new Label("min"), initMin, new Label("max"), initMax,
				new Label("seed"), seed, randomize, reset, storeWeights, clearWeights);
		final var trackingRow = row(trackRuns, trackingSettings, trackingStatus, lastRun);
		trackingStatus.setMinWidth(0);
		HBox.setHgrow(trackingStatus, Priority.SOMETIMES);
		final var header = new VBox(8, modelRow, summary, trainingRow, weightRow, trackingRow, status);
		header.setPadding(new Insets(10, 12, 6, 12));
		return header;
	}

	private static HBox row(javafx.scene.Node... nodes) {
		final var row = new HBox(8, nodes);
		row.setAlignment(Pos.CENTER_LEFT);
		for (var node : nodes) {
			if (node instanceof javafx.scene.control.Labeled labeled) {
				labeled.setMinWidth(javafx.scene.layout.Region.USE_PREF_SIZE);
			}
		}
		return row;
	}

	private SplitPane body() {
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
		final var legend = new Label("Blue = positive, red = negative. Edge width/opacity relative to the largest shown;"
				+ " node fill = activation, ring = bias. Small-layer arrows: green ↑ weight increased,"
				+ " red ↓ decreased (last update).");
		legend.setWrapText(true);
		legend.setStyle("-fx-text-fill: #555;");
		final var graphPane = new VBox(8, graph, detail, legend,
				row(layerLabel, layer, next, all, new Label("Edges"), edgeMode, before),
				row(inputLabel, input, infer, browse), output);
		VBox.setVgrow(graph, Priority.ALWAYS);
		graphPane.setPadding(new Insets(10));

		lossChart.setLegendVisible(false);
		lossChart.getData().add(losses);
		targets.setName("Target");
		predictions.setName("Prediction");
		decision.setName("0.5 threshold");
		functionChart.getData().addAll(List.of(targets, predictions));
		final var outputs = new StackPane(functionChart, surface, imageView);
		// Hidden views must not count toward the panel's minimum height.
		for (var view : outputs.getChildren()) {
			view.managedProperty().bind(view.visibleProperty());
		}
		StackPane.setAlignment(imageView, Pos.TOP_LEFT);
		VBox.setVgrow(outputs, Priority.ALWAYS);
		outputRow = row(new Label("Output"), outputSelector);
		final var charts = new VBox(6, row(new Label("Loss"), logLoss), lossChart, outputRow, outputs, results);
		VBox.setVgrow(lossChart, Priority.SOMETIMES);
		charts.setPadding(new Insets(10));
		charts.setMinWidth(360);
		// Fills the panel when there is room; scrolls once the charts and results reach their minimum heights.
		final var chartsScroll = new ScrollPane(charts);
		chartsScroll.setFitToWidth(true);
		chartsScroll.setFitToHeight(true);
		chartsScroll.setHbarPolicy(ScrollPane.ScrollBarPolicy.NEVER);
		chartsScroll.setMinWidth(380);
		final var split = new SplitPane(graphPane, chartsScroll);
		split.setDividerPositions(0.56);
		return split;
	}

	private static LineChart<Number, Number> chart(String xLabel, String yLabel, String title) {
		final var x = new NumberAxis();
		final var y = new NumberAxis();
		x.setLabel(xLabel);
		y.setLabel(yLabel);
		x.setForceZeroInRange(false);
		y.setForceZeroInRange(false);
		final var chart = new LineChart<Number, Number>(x, y);
		chart.setTitle(title);
		chart.setAnimated(false);
		chart.setCreateSymbols(false);
		chart.setMinHeight(170);
		chart.setPrefHeight(230);
		return chart;
	}

	private void wireActions() {
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
		models.valueProperty().addListener((observable, oldValue, value) -> {
			if (!switchingModels && value != null) {
				loadModel(value);
			}
		});
		newModel.setOnAction(event -> editModel(null, template(), "New model"));
		viewJson.setOnAction(event -> new JsonViewer(stage, model, network.copyWeights()).show());
		// The selected entry (not the displayed model) so a model whose dataset failed to load can be fixed.
		edit.setOnAction(event -> editModel(models.getValue(), models.getValue().model(),
				"Edit " + models.getValue().model().name()));
		duplicate.setOnAction(event -> editModel(null,
				models.getValue().model().withName(models.getValue().model().name() + " copy"), "Duplicate model"));
		browse.setOnAction(event -> browseImage());
		delete.setOnAction(event -> deleteModel());
		importModel.setOnAction(event -> importModel());
		randomize.setOnAction(event -> randomizeWeights());
		reset.setOnAction(event -> resetNetwork());
		storeWeights.setOnAction(event -> saveCurrent(model.withWeights(network.copyWeights()),
				"Stored current weights in the model; Reset to model now restores them"));
		clearWeights.setOnAction(event -> saveCurrent(model.withWeights(null),
				"Cleared stored weights; Reset to model now uses the initialization settings"));
		infer.setOnAction(event -> infer());
		input.setOnAction(event -> {
			if (!active) {
				infer();
			}
		});
		before.selectedProperty().addListener((observable, oldValue, value) -> showLastUpdate());
		edgeMode.valueProperty().addListener((observable, oldValue, value) -> {
			graph.setMode(value);
			render();
		});
		logLoss.selectedProperty().addListener((observable, oldValue, value) -> redrawLosses());
		outputSelector.getSelectionModel().selectedIndexProperty().addListener((observable, oldValue, value) -> {
			if (value.intValue() >= 0 && !switchingModels) {
				selectedOutput = value.intValue();
				if (!active) {
					lastGrid = computeGrid();
				}
				updatePredictions(lastPredictions, lastGrid);
			}
		});
	}

	// ---------------------------------------------------------------- catalog

	private void refreshModels(String selectId) {
		switchingModels = true;
		try {
			models.getItems().setAll(catalog.entries());
		} finally {
			switchingModels = false;
		}
		final var selected = models.getItems().stream().filter(entry -> entry.id().equals(selectId)).findFirst()
				.orElse(models.getItems().get(0));
		if (selected.equals(models.getValue())) {
			loadModel(selected);
		} else {
			models.setValue(selected);
		}
	}

	private void loadModel(ExampleCatalog.Entry entry) {
		final int generation = ++loadGeneration;
		if (entry.model().dataset() instanceof HuggingFaceDataset images) {
			prepareDataset(entry, images.repository() + " (" + images.config() + "/" + images.split() + ")",
					generation, report -> {
						images.fetch((message, fraction) -> report.accept(message
								+ (fraction >= 0 ? String.format(Locale.ROOT, " (%.0f%%)", 100 * fraction) : "")));
						final var train = images.trainingSamples();
						final var test = images.heldOutSamples();
						report.accept("Decoding " + (train.size() + test.size()) + " images of " + images.repository()
								+ "…");
						return new PreparedData(images.toTrainingData(train),
								test.isEmpty() ? null : images.toTrainingData(test), train, test);
					});
		} else if (entry.model().dataset() instanceof HuggingFaceTableDataset table) {
			prepareDataset(entry, table.repository() + " (" + table.config() + "/" + table.split() + ")", generation,
					report -> {
						table.fetch((message, fraction) -> report.accept(message
								+ (fraction >= 0 ? String.format(Locale.ROOT, " (%.0f%%)", 100 * fraction) : "")));
						report.accept("Reading rows of " + table.repository() + "…");
						final var prepared = table.prepare();
						return new PreparedData(prepared.train(), prepared.test(), List.of(), List.of());
					});
		} else {
			applyModel(entry, entry.model().dataset().toTrainingData(), null, List.of(), List.of());
		}
	}

	private interface DatasetWork {
		PreparedData load(java.util.function.Consumer<String> report) throws Exception;
	}

	/** Downloads (first time only) and reads a Hugging Face dataset off the FX thread, then shows the model. */
	private void prepareDataset(ExampleCatalog.Entry entry, String what, int generation, DatasetWork work) {
		setDatasetPending(true);
		status.setText("Preparing " + what + " for " + entry.model().name() + "…");
		final var pendingStatus = new AtomicReference<String>();
		final java.util.function.Consumer<String> report = text -> {
			if (pendingStatus.getAndSet(text) == null) {
				Platform.runLater(() -> {
					final String latest = pendingStatus.getAndSet(null);
					if (latest != null && generation == loadGeneration) {
						status.setText(latest);
					}
				});
			}
		};
		final var task = new Task<PreparedData>() {
			@Override
			protected PreparedData call() throws Exception {
				return work.load(report);
			}
		};
		task.setOnSucceeded(event -> {
			if (generation == loadGeneration) {
				final var prepared = task.getValue();
				applyModel(entry, prepared.train(), prepared.test(), prepared.trainSamples(), prepared.testSamples());
			}
		});
		task.setOnFailed(event -> {
			if (generation == loadGeneration) {
				final var error = task.getException();
				status.setText("Could not prepare the dataset of " + entry.model().name() + ": "
						+ (error.getMessage() == null ? error.toString() : error.getMessage())
						+ ". Fix it with Edit, or select another model and back to retry.");
			}
		});
		final var thread = new Thread(task, "dataset-loader");
		thread.setDaemon(true);
		thread.start();
	}

	/**
	 * While a dataset loads (or after it failed) the previous network stays on screen but only model
	 * management is possible.
	 */
	private void setDatasetPending(boolean pending) {
		lockedWhileTraining.forEach(node -> node.setDisable(pending));
		step.setDisable(pending);
		if (pending) {
			for (var node : List.of(models, newModel, edit, duplicate, importModel)) {
				node.setDisable(false);
			}
			final var entry = models.getValue();
			delete.setDisable(entry == null || entry.source() == ExampleCatalog.Source.BUILT_IN);
		}
	}

	private void applyModel(ExampleCatalog.Entry entry, TrainingData trainingData, TrainingData heldOut,
			List<DatasetCache.ImageSample> trainSamples, List<DatasetCache.ImageSample> testSamples) {
		model = entry.model();
		scaling = model.dataset().scaling();
		data = trainingData;
		testData = heldOut;
		trainImages = trainSamples;
		testImages = testSamples;
		lastTestPredictions = List.of();
		setDatasetPending(false);
		final var images = model.dataset() instanceof HuggingFaceDataset hf ? hf : null;
		final var table = model.dataset() instanceof HuggingFaceTableDataset rows ? rows : null;
		final var training = model.training();
		rate.setText(Float.toString(training.learningRate()));
		threshold.setText(Float.toString(training.targetLoss()));
		epochs.setText(Integer.toString(training.maxEpochs()));
		final var init = model.initialization();
		scheme.setValue(init.scheme());
		initMin.setText(Float.toString(init.min()));
		initMax.setText(Float.toString(init.max()));
		seed.setText(init.seed() == null ? "" : Long.toString(init.seed()));
		input.setText(images != null ? (testData != null ? "test:0" : "train:0")
				: table != null ? rawText(scaling.unscaleInput(data.getInput()[0]))
						: Arrays.toString(data.getInput()[0]).replaceAll("[\\[\\]]", ""));
		inputLabel.setText(images != null ? IMAGE_INPUT : table != null ? TABLE_INPUT : NUMERIC_INPUT);
		input.setPrefColumnCount(images != null ? 18 : 12);
		browse.setVisible(images != null);
		browse.setManaged(images != null);
		outputRow.setVisible(images == null);
		outputRow.setManaged(images == null);
		// Large datasets would take minutes per epoch with a per-update delay.
		if (images != null || data.getInput().length > LARGE_DATASET) {
			speed.setValue(0);
			autoFastSpeed = true;
		} else if (autoFastSpeed) {
			speed.setValue(15);
			autoFastSpeed = false;
		}
		switchingModels = true;
		try {
			outputSelector.getItems().setAll(model.dataset().outputNames());
			outputSelector.getSelectionModel().select(0);
			selectedOutput = 0;
		} finally {
			switchingModels = false;
		}
		outputSelector.setDisable(model.dataset().outputSize() < 2);
		delete.setText(entry.source() == ExampleCatalog.Source.MODIFIED_BUILT_IN ? "Restore built-in" : "Delete");
		delete.setDisable(entry.source() == ExampleCatalog.Source.BUILT_IN);
		clearWeights.setDisable(model.weights() == null);
		final boolean twoInputs = model.network().inputs() == 2 && images == null;
		surface.setVisible(twoInputs);
		functionChart.setVisible(!twoInputs && images == null);
		imageView.setVisible(images != null);
		final var layout = images == null ? null : new NetworkCanvas.ImageLayout(images.image().width(),
				images.image().height(), images.image().channels(), images.classes());
		graph.setImageLayout(layout);
		if (images != null) {
			final var splits = new ArrayList<ImageClassificationView.Split>();
			splits.add(new ImageClassificationView.Split("Training", "train", data.getInput(), labels(trainImages)));
			if (testData != null) {
				splits.add(new ImageClassificationView.Split("Test", "test", testData.getInput(), labels(testImages)));
			}
			imageView.setData(layout, splits);
		}
		functionChart.getXAxis().setLabel(table != null ? "training rows sorted by target"
				: model.network().inputs() == 1 ? model.dataset().inputNames().get(0) : "sample index");
		functionChart.getData().remove(decision);
		if (model.isClassification()) {
			functionChart.getData().add(decision);
		}
		final String samples = table != null ? data.getInput().length + " training"
				+ (testData == null ? "" : " + " + testData.getInput().length + " test") + " rows from "
				+ table.repository() + (scaling.isIdentity() ? "" : " (scaled)")
				: images == null ? data.getInput().length + " samples"
				: data.getInput().length + " training" + (testData == null ? "" : " + " + testData.getInput().length
						+ " test") + " images from " + images.repository();
		summary.setText(String.format(Locale.ROOT, "%s%s  |  %s  |  %d parameters  |  %s  |  %s%s",
				model.description().isEmpty() ? "" : model.description() + "  |  ", entry.source(),
				model.network().describe(), model.network().parameterCount(), samples,
				model.task(), model.weights() != null ? "  |  stored weights" : ""));
		resetNetwork();
	}

	private static int[] labels(List<DatasetCache.ImageSample> samples) {
		return samples.stream().mapToInt(DatasetCache.ImageSample::label).toArray();
	}

	private ModelDefinition template() {
		return ExampleCatalog.builtIn("square").withName("New model").withWeights(null);
	}

	private void editModel(ExampleCatalog.Entry existing, ModelDefinition initial, String title) {
		final var dialog = new ModelEditorDialog(stage, title, initial);
		dialog.showAndWait().ifPresent(edited -> {
			final String id = existing != null ? existing.id() : catalog.newIdFor(edited.name());
			try {
				final var saved = catalog.save(id, edited);
				refreshModels(saved.id());
				status.setText("Saved " + saved.model().name() + " (" + saved.source() + ") to "
						+ catalog.fileFor(id));
			} catch (IOException | IllegalArgumentException exception) {
				showError("Could not save model", exception);
			}
		});
	}

	private void saveCurrent(ModelDefinition updated, String message) {
		final var entry = models.getValue();
		try {
			catalog.save(entry.id(), updated);
			final var weights = network.copyWeights();
			refreshModels(entry.id());
			network.setWeights(weights);
			afterWeightsChanged();
			status.setText(message);
		} catch (IOException | IllegalArgumentException exception) {
			showError("Could not save model", exception);
		}
	}

	private void deleteModel() {
		final var entry = models.getValue();
		final boolean restore = entry.source() == ExampleCatalog.Source.MODIFIED_BUILT_IN;
		final var confirm = new Alert(Alert.AlertType.CONFIRMATION,
				restore ? "Discard your changes to '" + entry.model().name() + "' and restore the built-in example?"
						: "Delete '" + entry.model().name() + "' from " + catalog.directory() + "?",
				ButtonType.OK, ButtonType.CANCEL);
		confirm.initOwner(stage);
		if (confirm.showAndWait().orElse(ButtonType.CANCEL) != ButtonType.OK) {
			return;
		}
		try {
			catalog.delete(entry.id());
			refreshModels(restore ? entry.id() : ExampleCatalog.BUILT_IN_IDS.get(0));
			status.setText(restore ? "Restored built-in " + entry.id() : "Deleted " + entry.model().name());
		} catch (IOException | IllegalArgumentException exception) {
			showError("Could not delete model", exception);
		}
	}

	private FileChooser jsonChooser(String title) {
		final var chooser = new FileChooser();
		chooser.setTitle(title);
		chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Model JSON", "*.json"));
		return chooser;
	}

	private void importModel() {
		final var file = jsonChooser("Import model").showOpenDialog(stage);
		if (file == null) {
			return;
		}
		try {
			final var imported = ModelJson.read(file.toPath()).requireValid();
			final var saved = catalog.save(catalog.newIdFor(imported.name()), imported);
			refreshModels(saved.id());
			status.setText("Imported " + imported.name() + " into " + catalog.fileFor(saved.id()));
		} catch (IOException | IllegalArgumentException exception) {
			showError("Could not import " + file.getName(), exception);
		}
	}

	private void exportModel(ExportFormat format) {
		if (format == ExportFormat.PMML) {
			final var reason = PmmlExporter.unsupportedReason(model);
			if (reason.isPresent()) {
				showError("PMML export unavailable", new IllegalArgumentException(reason.get()));
				return;
			}
		}
		final var chooser = new FileChooser();
		chooser.setTitle("Export " + format.title.toLowerCase(Locale.ROOT));
		chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter(format.filter, "*." + format.extension));
		chooser.setInitialFileName(models.getValue().id() + format.suffix + "." + format.extension);
		final var file = chooser.showSaveDialog(stage);
		if (file == null) {
			return;
		}
		exportTo(file.toPath(), format);
	}

	void exportTo(Path path, ExportFormat format) {
		try {
			switch (format) {
				case CONFIGURATION -> ModelJson.write(model, path);
				case WITH_WEIGHTS -> ModelJson.write(model.withWeights(network.copyWeights()), path);
				case ONNX -> OnnxExporter.write(model, network.copyWeights(), path);
				case PMML -> PmmlExporter.write(model, network.copyWeights(), path);
			}
			status.setText("Exported " + format.title.toLowerCase(Locale.ROOT) + " to " + path);
		} catch (IOException | IllegalArgumentException exception) {
			showError("Could not export", exception);
		}
	}

	// ---------------------------------------------------------------- weights

	private void resetNetwork() {
		final long fallback = ThreadLocalRandom.current().nextLong();
		network = model.createNetwork(fallback);
		final String origin = model.weights() != null ? "stored weights"
				: model.initialization().scheme() + " initialization, " + (model.initialization().seed() == null
						? "random seed " + fallback : "model seed " + model.initialization().seed());
		clearHistory();
		afterWeightsChanged();
		weightOrigin = origin;
		continuedTraining = false;
		status.setText("Network reset from " + origin + ". Train continues from the current weights.");
	}

	private void randomizeWeights() {
		final var problems = new ArrayList<String>();
		final Long chosen = parseSeed(problems);
		final float min = parseFloat(initMin, "min", problems);
		final float max = parseFloat(initMax, "max", problems);
		final var spec = new InitializationSpec(scheme.getValue(), min, max, chosen);
		final var validation = spec.withSeed(0L);
		if (problems.isEmpty()) {
			final var check = model.withInitialization(validation).withWeights(null).validate();
			problems.addAll(check);
		}
		if (!problems.isEmpty()) {
			showError("Invalid weight settings", new IllegalArgumentException(String.join("\n", problems)));
			return;
		}
		final long used = chosen != null ? chosen : ThreadLocalRandom.current().nextLong();
		spec.initialize(network, used);
		clearHistory();
		afterWeightsChanged();
		weightOrigin = spec.scheme() + (spec.scheme() == InitializationSpec.Scheme.UNIFORM
				? " [" + min + ", " + max + ")" : "") + " initialization, seed " + used;
		continuedTraining = false;
		status.setText("Randomized with " + spec.scheme() + ", seed " + used
				+ " (enter this seed to reproduce). Model definition unchanged.");
	}

	private Long parseSeed(List<String> problems) {
		final String text = seed.getText().trim();
		if (text.isEmpty()) {
			return null;
		}
		try {
			return Long.parseLong(text);
		} catch (NumberFormatException exception) {
			problems.add("seed must be a whole number or blank for random");
			return null;
		}
	}

	private static float parseFloat(TextField field, String label, List<String> problems) {
		try {
			final float value = Float.parseFloat(field.getText().trim());
			if (Float.isFinite(value)) {
				return value;
			}
		} catch (NumberFormatException ignored) {
			// reported below
		}
		problems.add(label + " must be a finite number");
		return 0;
	}

	private void clearHistory() {
		lastUpdate = null;
		latestUpdate.set(null);
		latestEpoch.set(null);
		lastEpoch = 0;
		lossHistory.clear();
		losses.getData().clear();
		progress = "";
	}

	private void afterWeightsChanged() {
		layer.setMax(network.getDepth());
		layer.setValue(network.getDepth());
		setActive(false);
		lastGrid = computeGrid();
		lastTestPredictions = predictTest();
		updatePredictions(predict(), lastGrid);
		infer();
	}

	// ---------------------------------------------------------------- training

	private void startTraining(boolean singleStep) {
		final float learningRate;
		final float target;
		final int maxEpochs;
		try {
			learningRate = Float.parseFloat(rate.getText().trim());
			target = Float.parseFloat(threshold.getText().trim());
			maxEpochs = Integer.parseInt(epochs.getText().trim());
			if (!Float.isFinite(learningRate) || learningRate <= 0 || !Float.isFinite(target) || target < 0
					|| maxEpochs <= 0) {
				throw new IllegalArgumentException(
						"Use a positive finite rate, positive epoch count and non-negative finite MSE.");
			}
		} catch (IllegalArgumentException exception) {
			showError("Invalid training settings", exception);
			return;
		}
		final var trainingData = data;
		final var shuffle = model.training().shuffleRandom();
		final boolean surfaces = computesSurface();
		final boolean throttled = model.isImageClassification() || trainingData.getInput().length > LARGE_DATASET;
		final var bounds = model.dataset().inputBounds();
		epochOffset = lastEpoch;
		stopping.set(false);
		synchronized (control) {
			paused = singleStep;
			permits = singleStep ? 1 : 0;
		}
		pause.setText(singleStep ? "Resume" : "Pause");
		setActive(true);
		status.setText(singleStep ? "Single update requested" : "Training");
		final int stepOffset = epochOffset;
		final MlflowRunLogger logger = tracking.enabled() ? startRun(learningRate, target, maxEpochs) : null;
		runLogger = logger;
		continuedTraining = true;
		final var task = new Task<Float>() {
			@Override
			protected Float call() {
				final var listener = new TrainingListener() {
					private long lastSurface;
					private long lastObserved;

					@Override
					public void beforeUpdate(int epoch, long step, int sampleIndex) {
						awaitPermission();
					}

					@Override
					public boolean observeUpdate(long step) {
						if (!throttled) {
							return true;
						}
						final long now = System.nanoTime();
						final boolean stepping;
						synchronized (control) {
							stepping = paused;
						}
						if (stepping || now - lastObserved >= IMAGE_OBSERVE_INTERVAL_NANOS) {
							lastObserved = now;
							return true;
						}
						return false;
					}

					@Override
					public void onUpdate(TrainingUpdate update) {
						latestUpdate.set(update);
					}

					@Override
					public void onEpoch(int epoch, float loss) {
						if (logger != null) {
							logger.logMetric("loss", loss, stepOffset + epoch);
						}
						OutputSurface.Grid grid = null;
						final long now = System.nanoTime();
						if (surfaces && now - lastSurface >= SURFACE_INTERVAL_NANOS) {
							grid = OutputSurface.compute(network, bounds, selectedOutput);
							lastSurface = now;
						}
						latestEpoch.set(new EpochState(epochOffset + epoch, loss, predict(), predictTest(), grid));
					}
				};
				return new StochasticGradientDescent(learningRate, shuffle).train(network, trainingData, target,
						maxEpochs, new MeanSquareError(), listener, stopping::get);
			}
		};
		task.setOnSucceeded(event -> finish(String.format(Locale.ROOT, "%s: MSE %.6f",
				task.getValue() <= target ? "Target reached" : "Epoch limit reached", task.getValue()), "FINISHED"));
		task.setOnFailed(event -> {
			final var error = task.getException();
			if (error instanceof CancellationException) {
				finish("Stopped; completed weight updates retained", "KILLED");
			} else {
				finish("Training failed", "FAILED");
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

	private void finish(String message, String runStatus) {
		flushUpdates();
		worker = null;
		setActive(false);
		lastGrid = computeGrid();
		lastTestPredictions = predictTest();
		final var finalPredictions = predict();
		updatePredictions(finalPredictions, lastGrid);
		status.setText(message);
		endRun(runStatus, message, finalPredictions);
	}

	// ---------------------------------------------------------------- MLflow tracking

	private void applyTracking(MlflowSettings settings) {
		tracking = settings;
		trackRuns.setSelected(settings.enabled());
		try {
			settings.save();
		} catch (IOException exception) {
			status.setText("Could not save MLflow settings: " + exception.getMessage());
		}
		showTrackingTarget();
	}

	private void showTrackingTarget() {
		trackingStatus.setText(tracking.enabled()
				? "to " + tracking.trackingUri() + ", experiment \"" + tracking.experimentName() + "\""
				: "off");
	}

	/** Called on the FX thread before the worker starts, so reading UI state and the network is safe. */
	private MlflowRunLogger startRun(float learningRate, float target, int maxEpochs) {
		final var errors = tracking.validate();
		if (!errors.isEmpty()) {
			trackingStatus.setText("Not logging: " + String.join("; ", errors));
			return null;
		}
		final var params = new LinkedHashMap<String, String>();
		params.put("model.id", models.getValue().id());
		params.put("model.name", model.name());
		params.put("model.task", model.task().name().toLowerCase(Locale.ROOT));
		params.put("network.shape", model.network().describe());
		params.put("network.activations", String.join(", ",
				model.network().layers().stream().map(layer -> layer.activation().toString()).toList()));
		params.put("network.parameters", Integer.toString(model.network().parameterCount()));
		params.put("training.learning_rate", Float.toString(learningRate));
		params.put("training.max_epochs", Integer.toString(maxEpochs));
		params.put("training.target_mse", Float.toString(target));
		params.put("training.shuffle_seed",
				model.training().shuffleSeed() == null ? "random" : model.training().shuffleSeed().toString());
		params.put("training.start_epoch", Integer.toString(epochOffset));
		params.put("weights.origin", (continuedTraining ? "continued training from " : "") + weightOrigin);
		params.put("dataset.type", model.dataset() instanceof net.talaatharb.nn.config.FunctionDataset
				? "function" : "table");
		params.put("dataset.samples", Integer.toString(data.getInput().length));
		if (model.dataset() instanceof HuggingFaceDataset images) {
			// 784 pixel names would exceed MLflow's parameter size limit; describe the picture instead.
			params.put("dataset.type", "huggingFace");
			params.put("dataset.repository", images.repository());
			params.put("dataset.config", images.config());
			params.put("dataset.split", images.split());
			if (testData != null) {
				params.put("dataset.test_split", images.testSplit());
				params.put("dataset.test_samples", Integer.toString(testData.getInput().length));
			}
			params.put("dataset.inputs", images.image().width() + "x" + images.image().height() + " "
					+ (images.image().grayscale() ? "grayscale" : "RGB") + " image");
		} else if (model.dataset() instanceof HuggingFaceTableDataset table) {
			params.put("dataset.type", "huggingFaceTable");
			params.put("dataset.repository", table.repository());
			params.put("dataset.config", table.config());
			params.put("dataset.split", table.split());
			params.put("dataset.target", table.targetColumn());
			params.put("dataset.seed", Long.toString(table.seed()));
			params.put("dataset.input_scaling", table.inputScaling().name().toLowerCase(Locale.ROOT));
			params.put("dataset.target_scaling", table.effectiveTargetScaling().name().toLowerCase(Locale.ROOT));
			if (testData != null) {
				params.put("dataset.test_split", table.testSplit() == null ? "held out from " + table.split()
						: table.testSplit());
				params.put("dataset.test_samples", Integer.toString(testData.getInput().length));
			}
			params.put("dataset.inputs", String.join(", ", model.dataset().inputNames()));
		} else {
			params.put("dataset.inputs", String.join(", ", model.dataset().inputNames()));
		}
		params.put("dataset.outputs", String.join(", ", model.dataset().outputNames()));
		final var tags = new LinkedHashMap<String, String>();
		tags.put("mlflow.source.name", "nn-approximations");
		tags.put("mlflow.source.type", "LOCAL");
		tags.put("mlflow.user", System.getProperty("user.name", "unknown"));
		tags.put("nn.model_id", models.getValue().id());
		if (!model.description().isBlank()) {
			tags.put("mlflow.note.content", model.description());
		}
		final String runName = model.name() + " " + java.time.LocalDateTime.now()
				.format(java.time.format.DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
		trackingStatus.setText("Logging run \"" + runName + "\" to " + tracking.trackingUri());
		return MlflowRunLogger.start(MlflowClient.connect(tracking),
				new MlflowRunLogger.RunRequest(tracking.experimentName(), runName, params, tags),
				problem -> Platform.runLater(() -> trackingStatus.setText(problem)));
	}

	/** Runs on the FX thread after the worker exited, so the network can be exported safely. */
	private void endRun(String runStatus, String outcome, List<float[]> finalPredictions) {
		final var logger = runLogger;
		runLogger = null;
		if (logger == null) {
			return;
		}
		final var metrics = new LinkedHashMap<String, Double>();
		metrics.put("epochs_trained", (double) (lastEpoch - epochOffset));
		if (!lossHistory.isEmpty() && lastEpoch > epochOffset) {
			metrics.put("final_loss", (double) lossHistory.get(lossHistory.size() - 1)[1]);
		}
		if (model.dataset() instanceof HuggingFaceTableDataset) {
			metrics.putAll(evaluate(data, finalPredictions));
			if (testData != null) {
				evaluate(testData, lastTestPredictions).forEach((key, value) -> metrics.put("test_" + key, value));
			}
		} else if (model.isClassification()) {
			metrics.put("accuracy", accuracy(finalPredictions));
		} else if (model.isImageClassification()) {
			metrics.put("accuracy", classAccuracy(data, finalPredictions));
			if (testData != null) {
				metrics.put("test_accuracy", classAccuracy(testData, lastTestPredictions));
			}
		}
		final var warnings = new ArrayList<String>();
		final var artifacts = RunArtifacts.build(model, network.copyWeights(), warnings);
		trackingStatus.setText("Uploading run results to " + tracking.trackingUri() + "...");
		logger.finish(new MlflowRunLogger.Outcome(runStatus, metrics, Map.of("outcome", outcome), artifacts))
				.whenComplete((result, error) -> Platform.runLater(() -> {
					if (error != null) {
						final var cause = error.getCause() != null ? error.getCause() : error;
						trackingStatus.setText("MLflow run not completed: " + cause.getMessage());
						return;
					}
					warnings.addAll(result.warnings());
					lastRunUrl = result.runUrl();
					lastRun.setVisible(true);
					trackingStatus.setText("Run logged" + (warnings.isEmpty() ? ""
							: " with warnings: " + String.join("; ", warnings)));
				}));
	}

	private double accuracy(List<float[]> predicted) {
		final float[][] expected = data.getOutput();
		int correct = 0;
		int total = 0;
		for (int sample = 0; sample < expected.length; sample++) {
			for (int o = 0; o < expected[sample].length; o++) {
				total++;
				if (predicted.get(sample)[o] >= 0.5f == expected[sample][o] >= 0.5f) {
					correct++;
				}
			}
		}
		return total == 0 ? 0 : (double) correct / total;
	}

	/** Share of samples whose strongest output is the target class. */
	private static double classAccuracy(TrainingData samples, List<float[]> predicted) {
		final float[][] expected = samples.getOutput();
		if (expected.length == 0 || predicted.size() != expected.length) {
			return 0;
		}
		int correct = 0;
		for (int i = 0; i < expected.length; i++) {
			correct += ImageClassificationView.argmax(predicted.get(i)) == ImageClassificationView
					.argmax(expected[i]) ? 1 : 0;
		}
		return (double) correct / expected.length;
	}

	private void setActive(boolean value) {
		active = value;
		lockedWhileTraining.forEach(node -> node.setDisable(value));
		if (!value) {
			final var entry = models.getValue();
			delete.setDisable(entry == null || entry.source() == ExampleCatalog.Source.BUILT_IN);
			clearWeights.setDisable(model == null || model.weights() == null);
			outputSelector.setDisable(model == null || model.dataset().outputSize() < 2);
			initMin.setDisable(scheme.getValue() != InitializationSpec.Scheme.UNIFORM);
			initMax.setDisable(scheme.getValue() != InitializationSpec.Scheme.UNIFORM);
		}
		pause.setDisable(!value);
		stop.setDisable(!value);
		step.setDisable(false);
	}

	private void flushUpdates() {
		final var update = latestUpdate.getAndSet(null);
		if (update != null) {
			lastUpdate = update;
			progress = String.format(Locale.ROOT, "Epoch %d, update %d, sample %d | sample MSE %.6f -> %.6f | target %s",
					epochOffset + update.epoch(), update.step(), update.sampleIndex(), update.lossBefore(),
					update.lossAfter(), describeTarget(update.expected()));
			showLastUpdate();
		}
		final var epoch = latestEpoch.getAndSet(null);
		if (epoch != null) {
			lastEpoch = epoch.epoch();
			addLoss(epoch.epoch(), epoch.loss());
			if (epoch.grid() != null) {
				lastGrid = epoch.grid();
			}
			lastTestPredictions = epoch.testPredictions();
			updatePredictions(epoch.predictions(), lastGrid);
		}
	}

	private void addLoss(int epoch, float loss) {
		lossHistory.add(new float[] { epoch, loss });
		if (lossHistory.size() > MAX_LOSS_POINTS) {
			// Keep the whole history visible by halving the resolution of older points.
			final var kept = new ArrayList<float[]>();
			for (int i = 0; i < lossHistory.size(); i++) {
				if (i % 2 == 0 || i == lossHistory.size() - 1) {
					kept.add(lossHistory.get(i));
				}
			}
			lossHistory.clear();
			lossHistory.addAll(kept);
			redrawLosses();
		} else {
			losses.getData().add(lossPoint(lossHistory.get(lossHistory.size() - 1)));
		}
	}

	private XYChart.Data<Number, Number> lossPoint(float[] point) {
		return new XYChart.Data<>(point[0], logLoss.isSelected() ? Math.log10(Math.max(point[1], 1e-12)) : point[1]);
	}

	private void redrawLosses() {
		lossChart.getYAxis().setLabel(logLoss.isSelected() ? "log10(MSE)" : "Dataset MSE");
		losses.getData().setAll(lossHistory.stream().map(this::lossPoint).toList());
	}

	// ---------------------------------------------------------------- inference & display

	private void showLastUpdate() {
		if (lastUpdate != null) {
			displayed = before.isSelected() ? lastUpdate.before() : lastUpdate.after();
			output.setText(progress + "\nDisplayed " + (before.isSelected() ? "before" : "after")
					+ " update output: " + describeOutput(displayed.output()));
			if (model.isImageClassification() && lastUpdate.sampleIndex() < data.getInput().length) {
				final int index = lastUpdate.sampleIndex();
				imageView.showSample("train:" + index + " (training)", data.getInput()[index],
						ImageClassificationView.argmax(data.getOutput()[index]), toArray(displayed.output()));
			}
			render();
		}
	}

	private String targetClass(List<Float> expected) {
		int best = 0;
		for (int i = 1; i < expected.size(); i++) {
			best = expected.get(i) > expected.get(best) ? i : best;
		}
		return className(best);
	}

	private boolean multiClass() {
		return model.isImageClassification()
				|| model.dataset() instanceof HuggingFaceTableDataset table && table.isCategorical();
	}

	private String describeTarget(List<Float> expected) {
		if (multiClass()) {
			return "class " + targetClass(expected);
		}
		return scaling.scalesOutputs() ? rawText(scaling.unscaleOutput(toArray(expected))) + " (scaled " + expected
				+ ")" : expected.toString();
	}

	private String describeOutput(List<Float> outputs) {
		if (multiClass()) {
			return describeClasses(outputs);
		}
		return scaling.scalesOutputs() ? rawText(scaling.unscaleOutput(toArray(outputs))) + " (network " + outputs
				+ ")" : outputs.toString();
	}

	private static String rawText(float[] values) {
		final var parts = new ArrayList<String>();
		for (float value : values) {
			parts.add(String.format(Locale.ROOT, "%.5g", value).replaceAll("(\\.\\d*?)0+(e|$)", "$1$2")
					.replaceAll("\\.(e|$)", "$1"));
		}
		return String.join(", ", parts);
	}

	private String className(int index) {
		final var names = model.dataset().outputNames();
		return index < names.size() ? names.get(index) : Integer.toString(index);
	}

	private String describeClasses(List<Float> outputs) {
		final float[] values = toArray(outputs);
		final int best = ImageClassificationView.argmax(values);
		return String.format(Locale.ROOT, "predicted %s (score %.3f)", className(best), values[best]);
	}

	private static float[] toArray(List<Float> values) {
		final float[] array = new float[values.size()];
		for (int i = 0; i < array.length; i++) {
			array[i] = values.get(i);
		}
		return array;
	}

	/** Runs {@code train:N}, {@code test:N} or an image file through the network. */
	private void inferImage(HuggingFaceDataset images) {
		final String text = input.getText().trim();
		final var id = SAMPLE_ID.matcher(text);
		final float[] values;
		final int label;
		final String title;
		final String source;
		if (id.matches()) {
			final boolean test = id.group(1).equals("test");
			final var set = test ? testData : data;
			final var samples = test ? testImages : trainImages;
			if (set == null) {
				throw new IllegalArgumentException("This model has no test split; use train:N");
			}
			final int index;
			try {
				index = Integer.parseInt(id.group(2));
			} catch (NumberFormatException exception) {
				throw new IllegalArgumentException("Sample index too large");
			}
			if (index >= set.getInput().length) {
				throw new IllegalArgumentException(id.group(1) + " samples are numbered 0 to "
						+ (set.getInput().length - 1));
			}
			values = set.getInput()[index];
			label = ImageClassificationView.argmax(set.getOutput()[index]);
			title = text;
			source = index < samples.size() ? samples.get(index).image().getFileName().toString() : "";
		} else {
			final String pathText = text.replaceAll("^\"|\"$", "");
			final Path path;
			try {
				path = Path.of(pathText);
			} catch (java.nio.file.InvalidPathException exception) {
				throw new IllegalArgumentException("Enter train:N, test:N or the path of an image file");
			}
			if (pathText.isEmpty() || !java.nio.file.Files.isRegularFile(path)) {
				throw new IllegalArgumentException("Enter train:N, test:N or the path of an image file");
			}
			values = images.pixels(path);
			label = -1;
			title = path.getFileName().toString();
			source = path.toString();
		}
		displayed = network.inspect(values);
		lastUpdate = null;
		before.setSelected(false);
		final float[] outputs = toArray(displayed.output());
		imageView.showSample(title, values, label, outputs);
		output.setText(title + (label >= 0 ? " (label " + className(label) + ")" : "") + " -> "
				+ describeClasses(displayed.output()) + (source.isEmpty() ? "" : " | " + source));
		render();
	}

	/** Runs {@code train:N}, {@code test:N} or raw column values (scaled like the training rows) through the network. */
	private void inferRow() {
		final String text = input.getText().trim();
		final var id = SAMPLE_ID.matcher(text);
		final float[] scaled;
		final float[] expected;
		final String title;
		if (id.matches()) {
			final boolean test = id.group(1).equals("test");
			final var set = test ? testData : data;
			if (set == null) {
				throw new IllegalArgumentException("This model has no test rows; use train:N");
			}
			final int index;
			try {
				index = Integer.parseInt(id.group(2));
			} catch (NumberFormatException exception) {
				throw new IllegalArgumentException("Row index too large");
			}
			if (index >= set.getInput().length) {
				throw new IllegalArgumentException(id.group(1) + " rows are numbered 0 to " + (set.getInput().length - 1));
			}
			scaled = set.getInput()[index];
			expected = set.getOutput()[index];
			title = text;
		} else {
			final var pieces = text.split("[,\\s]+");
			final var names = model.dataset().inputNames();
			if (text.isEmpty() || pieces.length != names.size()) {
				throw new IllegalArgumentException("Enter " + names.size() + " values (" + String.join(", ", names)
						+ ") or train:N / test:N");
			}
			final float[] raw = new float[pieces.length];
			for (int i = 0; i < raw.length; i++) {
				raw[i] = Float.parseFloat(pieces[i]);
				if (!Float.isFinite(raw[i])) {
					throw new IllegalArgumentException("Inputs must be finite");
				}
			}
			scaled = scaling.scaleInput(raw);
			expected = null;
			title = "Input";
		}
		displayed = network.inspect(scaled);
		lastUpdate = null;
		before.setSelected(false);
		output.setText(title + " [" + rawText(scaling.unscaleInput(scaled)) + "]"
				+ (expected == null ? "" : " (target " + describeTarget(toList(expected)) + ")") + " -> "
				+ describeOutput(displayed.output())
				+ (scaling.scalesInputs() ? "\nThe graph shows the scaled values the network sees." : ""));
		render();
	}

	private static List<Float> toList(float[] values) {
		final var list = new ArrayList<Float>(values.length);
		for (float value : values) {
			list.add(value);
		}
		return list;
	}

	private void browseImage() {
		final var chooser = new FileChooser();
		chooser.setTitle("Classify an image");
		chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Images", "*.png", "*.jpg", "*.jpeg",
				"*.gif", "*.bmp"));
		final var file = chooser.showOpenDialog(stage);
		if (file != null) {
			input.setText(file.getAbsolutePath());
			infer();
		}
	}

	private void infer() {
		if (model.dataset() instanceof HuggingFaceDataset images) {
			try {
				inferImage(images);
			} catch (RuntimeException exception) {
				showError("Invalid inference input", exception);
			}
			return;
		}
		if (model.dataset() instanceof HuggingFaceTableDataset) {
			try {
				inferRow();
			} catch (RuntimeException exception) {
				showError("Invalid inference input", exception);
			}
			return;
		}
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
			final var result = new StringBuilder("Input " + Arrays.toString(values) + " -> output "
					+ displayed.output());
			if (model.isClassification()) {
				result.append(" -> class ").append(displayed.output().stream().map(v -> v >= 0.5f ? "1" : "0").toList());
			}
			output.setText(result.toString());
			render();
		} catch (IllegalArgumentException exception) {
			showError("Invalid inference input", exception);
		}
	}

	private List<String> captions() {
		final var captions = new ArrayList<String>();
		captions.add("Input");
		final var layers = model.network().layers();
		for (int i = 0; i < layers.size(); i++) {
			captions.add((i == layers.size() - 1 ? "Output" : "Hidden " + (i + 1)) + " · " + layers.get(i).activation());
		}
		return captions;
	}

	private void render() {
		if (displayed != null) {
			final int index = (int) Math.round(layer.getValue());
			layerLabel.setText("Reveal to layer " + index);
			graph.show(displayed, lastUpdate != null && !before.isSelected() ? lastUpdate.before() : null, index,
					captions(), model.dataset().inputNames());
		}
	}

	/** Only call from the thread that currently owns the network. */
	private List<float[]> predict() {
		final var values = new ArrayList<float[]>();
		for (float[] row : data.getInput()) {
			values.add(network.apply(row).clone());
		}
		return List.copyOf(values);
	}

	/** Predictions for the held-out image samples (empty without a test split); same threading rule. */
	private List<float[]> predictTest() {
		final var held = testData;
		if (held == null) {
			return List.of();
		}
		final var values = new ArrayList<float[]>();
		for (float[] row : held.getInput()) {
			values.add(network.apply(row).clone());
		}
		return List.copyOf(values);
	}

	private boolean computesSurface() {
		return model.network().inputs() == 2 && !model.isImageClassification();
	}

	private OutputSurface.Grid computeGrid() {
		return computesSurface()
				? OutputSurface.compute(network, model.dataset().inputBounds(), selectedOutput)
				: null;
	}

	private void updateImagePredictions(List<float[]> values) {
		final var guesses = new ArrayList<int[]>();
		guesses.add(argmaxes(values));
		if (testData != null) {
			guesses.add(lastTestPredictions.size() == testData.getInput().length ? argmaxes(lastTestPredictions)
					: null);
		}
		imageView.update(guesses);
		final boolean test = testData != null && lastTestPredictions.size() == testData.getInput().length;
		final var set = test ? testData : data;
		final var predicted = test ? lastTestPredictions : values;
		final var samples = test ? testImages : trainImages;
		final var text = new StringBuilder(String.format(Locale.ROOT, "%-11s %-10s %-12s %-7s %s%n", "Sample",
				"Label", "Predicted", "Score", "File"));
		final int rows = Math.min(predicted.size(), set.getInput().length);
		for (int i = 0; i < Math.min(rows, MAX_RESULT_ROWS); i++) {
			final int label = ImageClassificationView.argmax(set.getOutput()[i]);
			final int guess = ImageClassificationView.argmax(predicted.get(i));
			text.append(String.format(Locale.ROOT, "%-11s %-10s %-12s %-7.3f %s%n", (test ? "test:" : "train:") + i,
					className(label), className(guess) + (guess == label ? "" : " ✗"), predicted.get(i)[guess],
					i < samples.size() ? samples.get(i).image().getFileName() : ""));
		}
		if (rows > MAX_RESULT_ROWS) {
			text.append("… ").append(rows - MAX_RESULT_ROWS).append(" more samples (see the gallery)\n");
		}
		results.setText(text.toString());
	}

	private static int[] argmaxes(List<float[]> values) {
		final int[] result = new int[values.size()];
		for (int i = 0; i < result.length; i++) {
			result[i] = ImageClassificationView.argmax(values.get(i));
		}
		return result;
	}

	private void updatePredictions(List<float[]> values, OutputSurface.Grid grid) {
		lastPredictions = values;
		if (model.isImageClassification()) {
			updateImagePredictions(values);
			return;
		}
		final int out = Math.min(selectedOutput, model.dataset().outputSize() - 1);
		final boolean oneInput = model.network().inputs() == 1;
		final boolean classification = model.isClassification();
		final var targetPoints = new ArrayList<XYChart.Data<Number, Number>>();
		final var predictionPoints = new ArrayList<XYChart.Data<Number, Number>>();
		final StringBuilder text;
		if (model.dataset() instanceof HuggingFaceTableDataset table) {
			text = tableRows(values, table, out, targetPoints, predictionPoints);
		} else {
			text = new StringBuilder(String.format(Locale.ROOT, "%-22s %-22s %s%n", "Input", "Target",
					"Predicted" + (classification ? " (class)" : "")));
			final int rows = Math.min(values.size(), data.getInput().length);
			for (int i = 0; i < rows; i++) {
				final float x = oneInput ? data.getInput()[i][0] : i;
				targetPoints.add(new XYChart.Data<>(x, data.getOutput()[i][out]));
				predictionPoints.add(new XYChart.Data<>(x, values.get(i)[out]));
				if (i < 500) {
					text.append(String.format(Locale.ROOT, "%-22s %-22s %s%n", compact(data.getInput()[i]),
							compact(data.getOutput()[i]), compact(values.get(i))
									+ (classification ? " " + classes(values.get(i)) : "")));
				}
			}
			if (rows > 500) {
				text.append("… ").append(rows - 500).append(" more rows\n");
			}
		}
		targets.getData().setAll(targetPoints);
		predictions.getData().setAll(predictionPoints);
		if (classification && !targetPoints.isEmpty()) {
			final double first = targetPoints.get(0).getXValue().doubleValue();
			final double last = targetPoints.get(targetPoints.size() - 1).getXValue().doubleValue();
			decision.getData().setAll(List.of(new XYChart.Data<>(first, 0.5), new XYChart.Data<>(last, 0.5)));
		}
		results.setText(text.toString());
		if (grid != null) {
			final float[][] inputs = data.getInput();
			final float[] expected = new float[inputs.length];
			for (int i = 0; i < inputs.length; i++) {
				expected[i] = data.getOutput()[i][out];
			}
			surface.show(grid, inputs, expected, classification, model.dataset().inputNames(),
					model.dataset().outputNames().get(out));
		}
	}

	/**
	 * Raw-unit results of a Hugging Face table: metrics, then rows. The chart orders the training rows by
	 * the selected target so predictions can be compared with a monotone target curve.
	 */
	private StringBuilder tableRows(List<float[]> values, HuggingFaceTableDataset table, int out,
			List<XYChart.Data<Number, Number>> targetPoints, List<XYChart.Data<Number, Number>> predictionPoints) {
		final int rows = Math.min(values.size(), data.getInput().length);
		final float[][] rawTargets = new float[rows][];
		final float[][] rawPredictions = new float[rows][];
		for (int i = 0; i < rows; i++) {
			rawTargets[i] = scaling.unscaleOutput(data.getOutput()[i]);
			rawPredictions[i] = scaling.unscaleOutput(values.get(i));
		}
		final Integer[] order = new Integer[rows];
		for (int i = 0; i < rows; i++) {
			order[i] = i;
		}
		Arrays.sort(order, (a, b) -> Float.compare(rawTargets[a][out], rawTargets[b][out]));
		final int step = Math.max(1, (rows + MAX_CHART_POINTS - 1) / MAX_CHART_POINTS);
		for (int k = 0; k < rows; k += step) {
			targetPoints.add(new XYChart.Data<>(k, rawTargets[order[k]][out]));
			predictionPoints.add(new XYChart.Data<>(k, rawPredictions[order[k]][out]));
		}
		final var text = new StringBuilder("Training rows: ").append(describeMetrics(evaluate(data, values)));
		if (testData != null && lastTestPredictions.size() == testData.getInput().length) {
			text.append("  |  test rows: ").append(describeMetrics(evaluate(testData, lastTestPredictions)));
		}
		text.append(scaling.isIdentity() ? "\n\n" : "\nValues in raw column units (the network sees them scaled)\n\n");
		text.append(String.format(Locale.ROOT, "%-10s %-40s %-16s %s%n", "Row", "Inputs", "Target", "Predicted"));
		final boolean categorical = table.isCategorical();
		for (int i = 0; i < Math.min(rows, MAX_RESULT_ROWS); i++) {
			final String target;
			final String predicted;
			if (categorical) {
				final int label = ImageClassificationView.argmax(data.getOutput()[i]);
				final int guess = ImageClassificationView.argmax(values.get(i));
				target = className(label);
				predicted = String.format(Locale.ROOT, "%s (%.3f)%s", className(guess), values.get(i)[guess],
						guess == label ? "" : " ✗");
			} else {
				target = rawText(rawTargets[i]);
				predicted = rawText(rawPredictions[i]) + (model.isClassification() ? " " + classes(values.get(i)) : "");
			}
			text.append(String.format(Locale.ROOT, "%-10s %-40s %-16s %s%n", "train:" + i,
					rawText(scaling.unscaleInput(data.getInput()[i])), target, predicted));
		}
		if (rows > MAX_RESULT_ROWS) {
			text.append("… ").append(rows - MAX_RESULT_ROWS).append(" more rows (infer train:N or test:N to inspect one)\n");
		}
		return text;
	}

	/**
	 * Quality of predictions in raw units: {@code accuracy} for classification (strongest output for several
	 * classes, 0.5 threshold otherwise), {@code mse} and {@code mae} for regression.
	 */
	private Map<String, Double> evaluate(TrainingData set, List<float[]> predicted) {
		final float[][] expected = set.getOutput();
		final var metrics = new LinkedHashMap<String, Double>();
		if (expected.length == 0 || predicted.size() != expected.length) {
			return metrics;
		}
		if (multiClass()) {
			metrics.put("accuracy", classAccuracy(set, predicted));
		} else if (model.isClassification()) {
			int correct = 0;
			int total = 0;
			for (int i = 0; i < expected.length; i++) {
				for (int o = 0; o < expected[i].length; o++) {
					total++;
					correct += predicted.get(i)[o] >= 0.5f == expected[i][o] >= 0.5f ? 1 : 0;
				}
			}
			metrics.put("accuracy", total == 0 ? 0 : (double) correct / total);
		} else {
			double squares = 0;
			double absolute = 0;
			int count = 0;
			for (int i = 0; i < expected.length; i++) {
				final float[] target = scaling.unscaleOutput(expected[i]);
				final float[] guess = scaling.unscaleOutput(predicted.get(i));
				for (int o = 0; o < target.length; o++) {
					final double error = guess[o] - target[o];
					squares += error * error;
					absolute += Math.abs(error);
					count++;
				}
			}
			metrics.put("mse", squares / count);
			metrics.put("mae", absolute / count);
		}
		return metrics;
	}

	private static String describeMetrics(Map<String, Double> metrics) {
		if (metrics.containsKey("accuracy")) {
			return String.format(Locale.ROOT, "accuracy %.1f%%", 100 * metrics.get("accuracy"));
		}
		return metrics.isEmpty() ? "-" : String.format(Locale.ROOT, "MSE %.4g, MAE %.4g", metrics.get("mse"),
				metrics.get("mae"));
	}

	private static String compact(float[] values) {
		final var parts = new ArrayList<String>();
		for (float value : values) {
			parts.add(String.format(Locale.ROOT, "%.4g", value));
		}
		return "[" + String.join(", ", parts) + "]";
	}

	private static String classes(float[] values) {
		final var builder = new StringBuilder("(");
		for (float value : values) {
			builder.append(value >= 0.5f ? '1' : '0');
		}
		return builder.append(')').toString();
	}

	private void showError(String title, Throwable error) {
		status.setText(title + ": " + error.getMessage());
		final var alert = new Alert(Alert.AlertType.ERROR);
		alert.initOwner(stage);
		alert.setHeaderText(title);
		alert.setContentText(error.getMessage() == null ? error.toString() : error.getMessage());
		alert.show();
	}

	/** For tests; only valid while no training is running. */
	float[][][] currentWeights() {
		return network.copyWeights();
	}

	@Override
	public void stop() {
		if (timer != null) {
			timer.stop();
		}
		requestStop();
		final var logger = runLogger;
		runLogger = null;
		if (logger != null) {
			try {
				logger.finish(new MlflowRunLogger.Outcome("KILLED", Map.of(),
						Map.of("outcome", "Application closed during training"), List.of()))
						.get(5, java.util.concurrent.TimeUnit.SECONDS);
			} catch (Exception ignored) {
				// Best effort: the application is exiting.
			}
		}
	}
}
