package net.talaatharb.nn.tracking;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

import net.talaatharb.nn.tracking.MlflowClient.Metric;
import net.talaatharb.nn.tracking.MlflowClient.RunInfo;

/**
 * Logs one training session as an MLflow run without ever blocking the caller: requests run on a private thread,
 * metrics are buffered and sent in batches, and failures are reported through a callback instead of interrupting
 * training.
 */
public final class MlflowRunLogger {

	public record RunRequest(String experimentName, String runName, Map<String, String> params,
			Map<String, String> tags) {
	}

	public record Artifact(String path, byte[] content) {
	}

	/** Status is FINISHED, FAILED or KILLED. */
	public record Outcome(String status, Map<String, Double> metrics, Map<String, String> tags,
			List<Artifact> artifacts) {
	}

	public record Result(String runUrl, List<String> warnings) {
	}

	private static final long FLUSH_INTERVAL_MILLIS = 1000;

	private final MlflowClient client;
	private final Consumer<String> onProblem;
	private final ScheduledExecutorService executor;
	private final ConcurrentLinkedQueue<Metric> pending = new ConcurrentLinkedQueue<>();
	private final CompletableFuture<RunInfo> started = new CompletableFuture<>();
	private volatile boolean metricsBroken;
	private volatile boolean finishing;

	private MlflowRunLogger(MlflowClient client, Consumer<String> onProblem) {
		this.client = client;
		this.onProblem = onProblem;
		this.executor = Executors.newSingleThreadScheduledExecutor(runnable -> {
			final var thread = new Thread(runnable, "mlflow-logger");
			thread.setDaemon(true);
			return thread;
		});
	}

	/** Creates the run in the background; {@code onProblem} receives connection or API errors. */
	public static MlflowRunLogger start(MlflowClient client, RunRequest request, Consumer<String> onProblem) {
		final var logger = new MlflowRunLogger(client, onProblem);
		logger.executor.execute(() -> {
			try {
				final String experimentId = client.experimentId(request.experimentName());
				final RunInfo run = client.createRun(experimentId, request.runName(), System.currentTimeMillis(),
						request.tags());
				try {
					client.logBatch(run.runId(), List.of(), request.params(), Map.of());
				} catch (IOException | RuntimeException exception) {
					onProblem.accept("MLflow parameters not logged: " + exception.getMessage());
				}
				logger.started.complete(run);
			} catch (IOException | RuntimeException exception) {
				logger.started.completeExceptionally(exception);
				onProblem.accept("MLflow run not created: " + exception.getMessage());
			}
		});
		logger.executor.scheduleWithFixedDelay(logger::flush, FLUSH_INTERVAL_MILLIS, FLUSH_INTERVAL_MILLIS,
				TimeUnit.MILLISECONDS);
		return logger;
	}

	/** Thread-safe and cheap; called from the training thread. */
	public void logMetric(String key, double value, long step) {
		if (!metricsBroken && !finishing) {
			pending.add(new Metric(key, value, System.currentTimeMillis(), step));
		}
	}

	/** Ends the run; completes with its URL once metrics, artifacts and status are sent. */
	public CompletableFuture<Result> finish(Outcome outcome) {
		finishing = true;
		final var result = new CompletableFuture<Result>();
		executor.execute(() -> {
			try {
				final RunInfo run = started.getNow(null);
				if (run == null) {
					throw new IOException("the MLflow run was never created");
				}
				flush();
				final long now = System.currentTimeMillis();
				final var finalMetrics = outcome.metrics().entrySet().stream()
						.map(entry -> new Metric(entry.getKey(), entry.getValue(), now, 0)).toList();
				final var warnings = new ArrayList<String>();
				try {
					client.logBatch(run.runId(), finalMetrics, Map.of(), outcome.tags());
				} catch (IOException exception) {
					warnings.add("summary not logged: " + exception.getMessage());
				}
				for (var artifact : outcome.artifacts()) {
					try {
						client.uploadArtifact(run, artifact.path(), artifact.content());
					} catch (IOException exception) {
						warnings.add(artifact.path() + " not uploaded: " + exception.getMessage());
						if (!run.artifactUri().startsWith("mlflow-artifacts:")) {
							break;
						}
					}
				}
				client.updateRun(run.runId(), outcome.status(), System.currentTimeMillis());
				if (metricsBroken) {
					warnings.add("some loss values were not logged");
				}
				result.complete(new Result(client.runUrl(run), warnings));
			} catch (IOException | RuntimeException exception) {
				result.completeExceptionally(exception);
			} finally {
				executor.shutdown();
			}
		});
		return result;
	}

	private void flush() {
		final RunInfo run = started.getNow(null);
		if (run == null || metricsBroken) {
			if (started.isCompletedExceptionally()) {
				pending.clear();
			}
			return;
		}
		final var batch = new ArrayList<Metric>();
		for (Metric metric; (metric = pending.poll()) != null;) {
			batch.add(metric);
		}
		if (batch.isEmpty()) {
			return;
		}
		try {
			client.logBatch(run.runId(), batch, Map.of(), Map.of());
		} catch (IOException | RuntimeException exception) {
			metricsBroken = true;
			pending.clear();
			onProblem.accept("MLflow metrics stopped: " + exception.getMessage());
		}
	}
}
