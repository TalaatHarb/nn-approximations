package net.talaatharb.nn.ui;

import static org.junit.jupiter.api.Assertions.*;

import java.util.concurrent.Callable;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;

import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import javafx.application.Platform;
import javafx.scene.control.Button;
import javafx.scene.control.ComboBox;
import javafx.scene.control.Label;
import javafx.scene.control.Slider;
import javafx.stage.Stage;
import net.talaatharb.nn.examples.ExampleProblem;

/** Opt-in because a desktop display is required. */
@EnabledIfSystemProperty(named = "javafx.smoke", matches = "true")
class VisualizationSmokeTest {

	@BeforeAll
	static void startToolkit() throws Exception {
		final var ready = new java.util.concurrent.CountDownLatch(1);
		Platform.startup(ready::countDown);
		assertTrue(ready.await(15, TimeUnit.SECONDS));
		Platform.setImplicitExit(false);
	}

	@AfterAll
	static void stopToolkit() {
		Platform.exit();
	}

	@Test
	void canStepStopInferSwitchExamplesAndTrainToCompletion() throws Exception {
		final var application = new NetworkVisualizationApplication();
		final var stage = fx(() -> {
			final var window = new Stage();
			application.start(window);
			return window;
		});
		try {
			fx(() -> {
				stage.getScene().getRoot().applyCss();
				stage.getScene().getRoot().layout();
				stage.getScene().getRoot().lookupAll(".label").stream()
						.filter(Label.class::isInstance).map(Label.class::cast)
						.filter(label -> label.getParent() instanceof NetworkView)
						.forEach(label -> {
							assertTrue(label.getWidth() > 0, "Graph label must have visible width");
							assertTrue(label.getHeight() > 0, "Graph label must have visible height");
						});
				stage.getScene().getRoot().lookupAll(".slider").stream()
						.filter(Slider.class::isInstance).map(Slider.class::cast)
						.filter(slider -> slider.getMax() == 200).forEach(slider -> slider.setValue(0));
				button(stage, "Single update").fire();
				return null;
			});
			await(stage, () -> hasLabel(stage, "update 1,"));
			assertTrue(fx(() -> button(stage, "Train").isDisabled()));
			Thread.sleep(150);
			assertFalse(fx(() -> hasLabel(stage, "update 2,")));
			fx(() -> {
				button(stage, "Single update").fire();
				return null;
			});
			await(stage, () -> hasLabel(stage, "update 2,"));
			fx(() -> {
				button(stage, "Stop").fire();
				return null;
			});
			await(stage, () -> !button(stage, "Train").isDisabled());
			fx(() -> {
				button(stage, "Infer").fire();
				assertTrue(hasLabel(stage, "Input [0.0, 1.0]"));
				button(stage, "Train").fire();
				return null;
			});
			await(stage, () -> hasLabel(stage, "Target reached:"));
			fx(() -> {
				final var selector = stage.getScene().getRoot().lookupAll(".combo-box").stream()
						.filter(ComboBox.class::isInstance).map(ComboBox.class::cast).findFirst().orElseThrow();
				selector.getSelectionModel().select(ExampleProblem.SQUARE.ordinal());
				assertTrue(hasLabel(stage, "Input [0.5]"));
				button(stage, "Train").fire();
				return null;
			});
			await(stage, () -> hasLabel(stage, "Target reached:"));
			assertTrue(fx(() -> !button(stage, "Infer").isDisabled()));
		} finally {
			fx(() -> {
				application.stop();
				stage.close();
				return null;
			});
		}
	}

	private static Button button(Stage stage, String text) {
		return stage.getScene().getRoot().lookupAll(".button").stream()
				.filter(Button.class::isInstance).map(Button.class::cast)
				.filter(button -> button.getText().equals(text)).findFirst().orElseThrow();
	}

	private static boolean hasLabel(Stage stage, String text) {
		return stage.getScene().getRoot().lookupAll(".label").stream()
				.filter(Label.class::isInstance).map(Label.class::cast)
				.anyMatch(label -> label.getText().contains(text));
	}

	private static void await(Stage stage, BooleanSupplier condition) throws Exception {
		long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(20);
		while (!fx(condition::getAsBoolean)) {
			if (System.nanoTime() > deadline) {
				fail("Timed out waiting for the desktop state: " + fx(() ->
						stage.getScene().getRoot().lookupAll(".label").stream()
								.filter(Label.class::isInstance).map(Label.class::cast)
								.map(Label::getText).toList()));
			}
			Thread.sleep(30);
		}
	}

	private static <T> T fx(Callable<T> action) throws Exception {
		final var task = new FutureTask<>(action);
		Platform.runLater(task);
		return task.get(15, TimeUnit.SECONDS);
	}
}
