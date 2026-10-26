# nn-approximations
Building a project for building neural network approximations for mathematical functions and visualizing the process of training and inference

## Run the desktop application

Install JDK 21 or newer and Maven 3.9+, then run from the repository root:

```powershell
Set-Location nn-approximations
mvn clean verify
mvn javafx:run
```

JavaFX dependencies and platform-native libraries are resolved by Maven; no separate JavaFX SDK is needed. A graphical desktop is required to run the viewer, but not for the normal test suite. The packaged JAR is not a standalone distribution: launch with `javafx:run`, which supplies the JavaFX runtime.

## Models and examples

Every example is a JSON **model definition**. Eleven built-ins ship as resources (`src/main/resources/examples`: XOR, AND, OR, NAND, NOR, linear, square, sine, MNIST handwritten digits, iris and California housing). User models are stored as `<id>.json` in `~/.nn-approximations/models` (override with the `nn.models.dir` Java system property). A user file with a built-in id (e.g. `xor.json`) overrides that built-in; **Restore built-in** deletes the override. Invalid files are skipped and reported in the status line.

- **New / Edit / Duplicate** open the model editor. Editing a built-in saves a modified copy; New and Duplicate save under a fresh id derived from the name.
- **Import** copies a JSON file into the model directory after validation. **Export** writes the configuration (optionally with the network's current weights) or the current network in a standard format (see below). **View JSON** opens a read-only, syntax-highlighted view of the model with the network's current weights (toggle weights, wrap lines, find, copy).
- The editor has three tabs:
  - **Model**: name, description, task (regression/classification/image classification), any number of hidden layers (neurons, activation and activation parameters, reorder/remove), output activation, weight initialization and training defaults. Input and output layer sizes follow the dataset.
  - **Data**: either a *function* (input variables with min/max/sample count, sampled as a grid, plus one expression per output, e.g. `sin(pi*x)`, `x1*x2`, `xor(a, b)`), a *table* of rows such as `0, 1 -> 1`, *Hugging Face images* (see [Image classification](#image-classification-with-hugging-face-datasets)) or a *Hugging Face table* (see [Tabular datasets](#tabular-hugging-face-datasets)). Picking the image classification task selects Hugging Face images and vice versa.
  - **JSON**: edit the raw definition (syntax highlighting and line numbers, via RichTextFX) and apply it back to the form.
  Stored weights are kept only when the topology is unchanged.

```json
{
  "modelType" : "feedForward",
  "formatVersion" : 1,
  "name" : "XOR",
  "task" : "classification",
  "network" : { "inputs" : 2, "layers" : [ { "neurons" : 4, "activation" : "tanh" }, { "neurons" : 1, "activation" : "sigmoid" } ] },
  "initialization" : { "scheme" : "uniform", "min" : -1.0, "max" : 1.0, "seed" : 42 },
  "training" : { "learningRate" : 0.3, "targetLoss" : 0.005, "maxEpochs" : 5000, "shuffleSeed" : 42 },
  "dataset" : { "type" : "table", "inputNames" : [ "a", "b" ], "outputNames" : [ "xor" ],
                "inputs" : [ [ 0, 0 ], [ 0, 1 ], [ 1, 0 ], [ 1, 1 ] ], "outputs" : [ [ 0 ], [ 1 ], [ 1 ], [ 0 ] ] }
}
```

- `network.layers` lists hidden layers followed by the output layer. Activations: `linear`, `sigmoid`, `tanh`, `relu`, `leakyRelu`, `elu`, and parameterized `{"type":"parametricRelu","alpha":0.1}`, `{"type":"scaledElu","alpha":1}`, `{"type":"piecewiseLinear","xMin":-1,"xMax":1}`, `{"type":"step","threshold":0}` (step has zero gradient, so it is not trainable).
- `initialization.scheme`: `uniform` (biases and weights in `[min, max)`), `xavierUniform` or `heUniform` (zero biases). Omit `seed` for a new random seed each reset; omit `training.shuffleSeed` for a different sample order on each run.
- A function dataset looks like `{"type":"function","inputs":[{"name":"x","min":-1,"max":1,"samples":41}],"outputs":["x^2"]}`. Expressions support `+ - * / % ^`, `pi`, `e`, common math functions (`sin`, `exp`, `log`, `sqrt`, `abs`, `min`, `max`, `pow`, `atan2`, ...) and logic helpers (`not`, `and`, `or`, `xor`, `nand`, `nor`, `step`) that treat values `>= 0.5` as true.
- `weights` is optional: `[layer][neuron][bias, w1..wn]`. When present, **Reset to model** restores it instead of initializing.
- Networks are limited to 250,000 weights, because every weight is drawn and copied into each visualization snapshot.
- Unknown properties and unsupported `modelType`s are rejected. `modelType` reserves room for other model kinds (such as RBMs) later.

## Viewer controls

- **Train** continues from the current weights; **Pause / Resume**, **Single update** and **Stop** control training. Single update also starts a paused session. The delay slider sets milliseconds per update; use zero to converge quickly. Rate, epochs and target MSE default to the model's training settings.
- **Weights** row: pick a scheme, range and seed, then **Randomize**. A blank seed draws a new one each time and the status line reports it so the result can be reproduced. **Reset to model** rebuilds from the model's stored weights or initialization. **Store weights in model** saves the current weights into the model file; **Clear stored weights** removes them.
- The network canvas scales to any layer size. Edge width and opacity are relative to the largest value shown. Blue is positive and red is negative; the node fill shows activation and the ring shows bias. Switch **Edges** to *Last update change* to see where SGD moved weights. Hover a neuron to highlight its connections and list z, bias, deltas and its strongest incoming/outgoing weights.
- Use the layer slider or **Next layer** to reveal hidden activations of one inference progressively. When idle, enter comma-separated inputs and click **Infer**.
- The loss chart shows dataset MSE per epoch, optionally on a log10 scale. Long runs are downsampled, so the full history stays visible.
- Single-input models plot target vs. prediction. Two-input models show a live heatmap of the selected output over the input plane with training points overlaid and, for classification, the 0.5 decision boundary. Other input counts plot by sample index. The table lists all inputs, targets and outputs; classification tasks also show the class at 0.5. The charts and results panel scrolls vertically when the window is too short to show it all.

Training runs on a background thread. Model management, weight changes and inference are disabled until that worker exits. The UI consumes detached snapshots at up to 30 frames/second and coalesces intermediate frames; listeners still receive every SGD update. The heatmap is recomputed on the training thread at most every 120 ms.

## Image classification with Hugging Face datasets

The `imageClassification` task trains on images from a [Hugging Face](https://huggingface.co/datasets) dataset. The built-in **MNIST handwritten digits** model uses [`ylecun/mnist`](https://huggingface.co/datasets/ylecun/mnist): 2,000 training and 500 test digits, each 28x28 grayscale, fed to a `784 → 32 sigmoid → 10 sigmoid` network. It reaches about 88–95% test accuracy within a few epochs.

```json
"task" : "imageClassification",
"dataset" : {
  "type" : "huggingFace",
  "repository" : "ylecun/mnist",
  "config" : "mnist",
  "split" : "train",
  "imageColumn" : "image",
  "labelColumn" : "label",
  "classes" : [ "0", "1", "2", "3", "4", "5", "6", "7", "8", "9" ],
  "samples" : 2000,
  "testSplit" : "test",
  "testSamples" : 500,
  "image" : { "width" : 28, "height" : 28, "grayscale" : true }
}
```

How the data is handled:

- **Inputs and outputs.** Each sample is an image file; the network input is its pixels, row-major, scaled to 0..1 (3 values per pixel when `grayscale` is false). The target is one-hot over `classes`, in the dataset's label order, and the predicted class is the output with the highest score. Images are resized to `image.width` x `image.height`, at most 64 px a side.
- **Fetching.** The first time a model is opened, the app asks the datasets server for the dataset info. It then downloads the Hub's Parquet export of each split (`/api/datasets/<repo>/parquet/<config>/<split>`) and reads only as many rows as needed, using a built-in Parquet reader (Snappy, gzip or uncompressed). The images are extracted to `~/.nn-approximations/datasets/<owner>__<name>/<config>/<split>/` with an `index.json` of labels; override the location with `-Dnn.datasets.dir=<dir>`. Later loads work offline, and the UI stays responsive while downloading.
- **Access and mirrors.** Set `HF_TOKEN` for gated or private datasets, and `HF_ENDPOINT` to use a mirror.
- **Editor.** In the model editor's *Data* tab, choose **Hugging Face images**, enter a repository and click **Load info**. This fills the configs, splits, image and label columns, class names and image size from the dataset's features. The layer sizes follow automatically: inputs = width x height x channels, outputs = number of classes.

Visualizing image models:

- **Network canvas.** The input layer is drawn as a picture of the current sample instead of 784 nodes, with one line fanning out to each first hidden unit. Hover the picture to read pixel values. Hover a first hidden unit to see its **receptive field**, which shows its 784 input weights (or their last update change) as a blue/red image. Output nodes are labelled with their class, and the strongest one is shown in bold.
- **Right panel.**
  - The current sample, with its label and predicted class, plus a bar chart of the class scores.
  - A **confusion matrix** for the test split, or for the training split if there is none, with training/test accuracy.
  - A paged **gallery** of training or test thumbnails, with green/red borders for correct/wrong predictions and a **Mistakes only** filter. Click a thumbnail to run it through the network.
  - The results table, which lists the first 200 samples.
- **Inference.** Enter `train:N`, `test:N` or the path of any image file (or use **Browse…**) and click **Infer**.
- **Training speed.** The delay slider starts at 0 for image models. To keep training fast, snapshots of the network are taken only every 30 ms, or on every update when paused or single-stepping.

MLflow runs for image models log the dataset repository, splits and sample counts, `accuracy` (argmax), and `test_accuracy` from the held-out split.

## Tabular Hugging Face datasets

Any Hugging Face dataset with numeric feature columns can be used for classification or regression. Two built-ins show both cases:

- **Iris** ([`scikit-learn/iris`](https://huggingface.co/datasets/scikit-learn/iris)): four flower measurements predict one of three species with a `4 → 8 tanh → 3 sigmoid` network. 30 of the 150 rows are held out for testing, and the test accuracy is typically 97–100%. The `Id` column is not an input, because the rows are sorted by species and the id would give the answer away.
- **California housing** ([`gvlassis/california_housing`](https://huggingface.co/datasets/gvlassis/california_housing)): eight census-block features predict the median house value (in 100,000 USD) with a `8 → 16 tanh → 1 linear` network. It trains on 5,000 rows and is measured on 500 rows of the `test` split, reaching a test MSE of about 0.37 (the target's variance is about 1.3).

```json
"task" : "classification",
"dataset" : {
  "type" : "huggingFaceTable",
  "repository" : "scikit-learn/iris",
  "config" : "default",
  "split" : "train",
  "inputColumns" : [ "SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm" ],
  "targetColumn" : "Species",
  "classes" : [ "Iris-setosa", "Iris-versicolor", "Iris-virginica" ],
  "samples" : 120,
  "testSamples" : 30,
  "seed" : 42,
  "inputScaling" : "standardize",
  "targetScaling" : "none"
}
```

How the data is handled:

- **Columns.** `inputColumns` must be numeric (integers, floats or booleans). With `classes`, the target is categorical: each class gets one output neuron (one-hot), and the target may be a string or an integer label (the class name, or an index into `classes`). Categorical targets need the `classification` task. Without `classes`, the target is a single numeric output, used for `regression` (or for `classification` of a 0/1 column, with `targetScaling` set to `none`).
- **Rows.** Rows are downloaded from the Hub's Parquet export (up to 100,000 per split) and cached as `table.json` under `~/.nn-approximations/datasets/`. Rows with a missing or non-numeric input or target, or a class not in `classes`, are skipped, and the rows are shuffled with `seed`.
- **Test rows.** With a `testSplit`, the first `samples` shuffled rows of `split` are used for training and up to `testSamples` rows of `testSplit` for testing. Without one, the first `testSamples` shuffled rows are held out of `split` for testing and training takes the next `samples` rows.
- **Scaling.** `inputScaling` and `targetScaling` are `standardize` (mean 0, standard deviation 1), `minMax` (0..1) or `none`. The statistics are computed from the training rows when the model is loaded, and are not stored in the JSON. Stored weights therefore stay valid as long as the dataset, `seed` and `samples` do not change. The network trains on scaled values, but everything shown in the app is in raw column units.
- **Editor.** In the model editor's *Data* tab, choose **Hugging Face table**, enter a repository and click **Load info**. The app reads the dataset's features and column statistics. It selects numeric columns as inputs, leaving out id-like columns such as `Id` or `index`. It picks the target from a class-label column, a column named like `label`, `target`, `class` or `species`, or else the last column. For a categorical target (a ClassLabel or a string column), it fills in the class names, and the task follows: classes mean classification, a numeric target means regression.

Visualizing table models:

- **Inference.** Enter raw feature values, such as `5.1, 3.5, 1.4, 0.2`, or `train:N` / `test:N`, and click **Infer**. The output shows the raw inputs, the target and the predicted value or class. The network canvas shows the scaled values the network sees.
- **Results.** The chart plots target and prediction for the training rows sorted by target, using at most 400 points. Above the table, the training and test metrics are shown: accuracy, or MSE and MAE in target units.
- **Training speed.** Datasets with more than 500 rows start with the delay slider at 0 and take snapshots less often.

ONNX and PMML exports include the scaling: they take raw feature values and return the target in its own units. ONNX adds `Sub`/`Div` nodes before the network and `Mul`/`Add` after it, and PMML uses `NormContinuous` fields. In ONNX, the metadata properties `nn.input_scaling`, `nn.output_scaling` and `nn.dataset` record the scaling and the source. MLflow runs log the repository, splits, target, scaling and seed, plus `accuracy` or `mse`/`mae`, with `test_` prefixes for the test rows.

## Standard export formats

The **Export** menu writes the network with its current weights. Files are written atomically.

- **ONNX** (`.onnx`, IR 8 / opset 13): one `Gemm` per layer followed by the activation operator. It works with onnxruntime, Netron, PyTorch/TensorFlow converters, ML.NET and MLflow. The input `input` is `float[batch, inputs]` and the output `output` is `float[batch, outputs]`. Metadata properties hold `nn.name`, `nn.task`, `nn.input_names`, `nn.output_names` and `nn.model_json`, which is the definition without weights. Every activation is supported. `piecewiseLinear` maps to `HardSigmoid`, and `step` maps to `Greater` followed by `Cast`.
- **PMML 4.4** (`.pmml`, `NeuralNetwork`): for tools such as PyPMML, JPMML and KNIME. Only the `linear`, `sigmoid`, `tanh`, `relu` and `step` activations exist in PMML. For other activations the menu item is disabled and explains why; use ONNX instead. Input and output names must be distinct.

Both exporters are checked against the Java network in the tests: ONNX through onnxruntime, and PMML through an independent evaluator.

### Running exported models from Python

Requirements:

- Python 3.9 or later.
- ONNX: `pip install onnxruntime numpy`.
- PMML: `pip install pypmml`. PyPMML wraps the Java PMML4S engine, so it also needs a Java 8+ runtime on the `PATH`. Any other PMML 4.4 consumer (e.g. JPMML, KNIME) works too.

Save the script as `run_model.py`:

```python
"""Run a model exported by nn-approximations (ONNX or PMML).

    python run_model.py xor.onnx 0,1 1,1
    python run_model.py xor.pmml 0,1 1,1
"""
import sys


def run_onnx(path, rows):
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    meta = session.get_modelmeta().custom_metadata_map
    inputs = np.asarray(rows, dtype=np.float32)            # shape [batch, inputs]
    (outputs,) = session.run(["output"], {"input": inputs})  # shape [batch, outputs]
    names = meta["nn.output_names"].split(",")
    return [dict(zip(names, map(float, row))) for row in outputs]


def run_pmml(path, rows):
    from pypmml import Model

    model = Model.fromFile(path)
    names = model.inputNames
    return [model.predict(dict(zip(names, row))) for row in rows]


if __name__ == "__main__":
    path, rows = sys.argv[1], [[float(v) for v in arg.split(",")] for arg in sys.argv[2:]]
    run = run_onnx if path.endswith(".onnx") else run_pmml
    for row, result in zip(rows, run(path, rows)):
        print(row, "->", result)
```

Example output for a trained XOR network (exact values depend on the weights):

```text
> python run_model.py xor.onnx 0,0 0,1
[0.0, 0.0] -> {'xor': 0.0312...}
[0.0, 1.0] -> {'xor': 0.9714...}
> python run_model.py xor.pmml 0,0 0,1
[0.0, 0.0] -> {'predicted_xor': 0.0312...}
[0.0, 1.0] -> {'predicted_xor': 0.9714...}
```

Inputs are given in the model's input order (`nn.input_names` in the ONNX metadata, or the PMML `MiningSchema`). PyPMML prefixes each output with `predicted_`. A model logged to MLflow can also be run with `mlflow.pyfunc.load_model("runs:/<run id>/model")`, as shown below.

Image classification models take the pixels as inputs. The ONNX metadata `nn.input_image` describes the expected layout (e.g. `28x28 grayscale, row-major, pixel values scaled to 0..1`), and `nn.dataset` names the source dataset. To classify a digit with the exported MNIST model (`pip install onnxruntime numpy pillow`):

```python
import numpy as np, onnxruntime as ort
from PIL import Image

session = ort.InferenceSession("mnist.onnx", providers=["CPUExecutionProvider"])
pixels = np.asarray(Image.open("digit.png").convert("L").resize((28, 28)), dtype=np.float32) / 255.0
(scores,) = session.run(["output"], {"input": pixels.reshape(1, 784)})
print("digit:", int(scores[0].argmax()))
```

MNIST digits are white on black; invert other images (`255 - pixels`) to match.

## MLflow tracking

Tick **Log runs to MLflow** to record each Train or Single update session as an MLflow run. Tracking is off by default. **MLflow settings…** sets the tracking URI and the experiment, which is created if missing, and has a **Test connection** button. Settings are stored in `~/.nn-approximations/mlflow.json`; override the path with `-Dnn.mlflow.settings=<file>`. The default URI is `https://mlflow.talaatharb.net`. The defaults can also come from `-Dnn.mlflow.uri`/`MLFLOW_TRACKING_URI` and `-Dnn.mlflow.experiment`/`MLFLOW_EXPERIMENT_NAME`; enable tracking with `-Dnn.mlflow.enabled=true`. For servers with authentication (e.g. `mlflow server --app-name basic-auth`), enter a **Username** and **Password**; they are sent with HTTP basic authentication, and **Test connection** checks that they are accepted. The password is kept for the session only unless **Save password** is ticked, in which case it is stored as plain text in the settings file (restricted to the current user where the file system allows). Without a username, `MLFLOW_TRACKING_TOKEN` or `MLFLOW_TRACKING_USERNAME`/`MLFLOW_TRACKING_PASSWORD` are used.

Each run records:

- **Params:** the model id, name and task; the network shape, activations and parameter count; the learning rate, epochs, target MSE, shuffle seed and starting epoch; the weight origin (scheme/seed or stored weights); and the dataset size.
- **Metrics:** `loss` per epoch, `final_loss`, `epochs_trained`, and `accuracy` for classification models. Hugging Face table models log `accuracy` or `mse`/`mae` in raw units, and `test_*` versions for their test rows.
- **Tags:** source, user, model id, description and outcome. The status is `FINISHED`, `KILLED` (stopped or app closed) or `FAILED`.
- **Artifacts:**
  - `nn-model.json`, the definition with the trained weights.
  - `model.pmml`, when PMML supports the model's activations.
  - `model/`, an MLflow model with the ONNX flavor and its signature.

You can load the logged model in Python:

```python
import mlflow
model = mlflow.pyfunc.load_model("runs:/<run id>/model")   # needs mlflow, onnx and onnxruntime
```

Logging happens on a background thread, so a slow or unreachable server never blocks training. Problems are shown in the tracking row, and **Open last run** opens the run in the browser. Artifact upload requires a server that proxies artifacts (`mlflow-artifacts:` URIs, the default for `mlflow server`).

## Observation API

The numerical core has no JavaFX dependency. Existing `apply`, layer getters and the original `train` overload remain available.

```java
var model = ExampleCatalog.builtIn("xor");          // or ModelJson.read(path).requireValid()
var network = model.createNetwork(42);              // stored weights, or initialization (42 if no seed)
var data = model.dataset().toTrainingData();
var snapshot = network.inspect(new float[] {0, 1});
// snapshot.layers(): input, hidden and output activations, z values and biases
// snapshot.connections(): topology and signed weights; all lists are immutable

var listener = new TrainingListener() {
    @Override
    public void onUpdate(TrainingUpdate update) {
        // epoch, step, sample index, targets, sample losses and before/after snapshots
    }

    @Override
    public void onEpoch(int epoch, float datasetLoss) {
        // Persist metrics or marshal them to a UI.
    }
};
var training = model.training();
new StochasticGradientDescent(training.learningRate(), training.shuffleRandom()).train(
    network, data, training.targetLoss(), training.maxEpochs(),
    new MeanSquareError(), listener, () -> false);
ModelJson.write(model.withWeights(network.copyWeights()), Path.of("xor-trained.json"));
```
Classes above live under `net.talaatharb.nn.config`, `.algorithms`, and `.metrics`. `beforeUpdate` can gate execution for interactive stepping, and `observeUpdate(step)` lets a listener skip building the (costly) before/after snapshots for updates it will not display. Callbacks run synchronously on the training thread; callback exceptions propagate. Cancellation (the supplier or thread interruption) throws `CancellationException` and retains completed updates. Do not interrogate or mutate a live network from another thread; pass snapshots instead.

`InspectableNetwork` and `NetworkSnapshot` in `.visualization` form the extension boundary. The graph format does not require forward-only edges, so a future RBM adapter can expose visible/hidden units and bidirectional connections without rewriting the renderer. RBM sampling, reconstruction phases and contrastive-divergence training are **not implemented**; SGD remains specifically feed-forward.

## Future plans

The next models are planned in this order. Each step builds on the previous one. None of them is implemented yet.

### 1. Autoencoder

An autoencoder is a feed-forward network trained to reproduce its input through a narrow middle layer (the *bottleneck*). It reuses the existing layers, SGD and visualization, so it is the cheapest next step.

- **Model.** A new task `reconstruction`, where the target is the input itself. For example, `784 → 64 → 16 → 64 → 784` on MNIST, or `4 → 2 → 4` on iris. The JSON format stays `feedForward`, and the dataset's targets are not used.
- **Training.** The existing SGD with MSE between the input and the reconstruction. An optional *denoising* mode corrupts each input (Gaussian noise or masked pixels) and trains the network to restore the clean version.
- **Visualization.**
  - Each input is shown next to its reconstruction, with an error heatmap for images.
  - A 2-D scatter of the bottleneck codes (bottleneck size 2, or the first two principal components), colored by class when labels exist.
  - A *latent explorer*: drag a point in the code space and see the decoder's output.
  - Interpolation between two samples through their codes.
- **Export.** ONNX/PMML of the whole network as today, plus separate encoder and decoder ONNX files.
- **Changes needed.** A way to mark the bottleneck layer, a `reconstruction` task in validation, MLflow metrics, and a reconstruction view (reusing the image view for image datasets).

### 2. Restricted Boltzmann machine (RBM)

An RBM is a two-layer *generative* model: visible units and hidden units, connected symmetrically, with no connections inside a layer. It is trained without labels by contrastive divergence. The project already prepares for it: `modelType` reserves room for other model kinds, and `NetworkSnapshot` connections are not required to point forward.

- **Model.** `"modelType" : "rbm"` with `visible`, `hidden`, the unit types (Bernoulli, or Gaussian visible units for real-valued tables), the weight matrix and the visible and hidden biases. It can train on binarized MNIST or on logic truth tables.
- **Training.** CD-k (k = 1 by default) or persistent CD, with learning rate, momentum, weight decay, mini-batch size and sparsity targets. Training is tracked by reconstruction error and pseudo-log-likelihood; MSE is not the training objective.
- **Visualization.**
  - Bidirectional edges in the network canvas.
  - The CD phases step by step: the data-driven *positive* phase (v → h), the Gibbs chain (h → v' → h'), and the weight update as the difference between the two.
  - Each hidden unit's weights drawn as an image filter.
  - The probabilities next to the sampled 0/1 states.
  - A free-running Gibbs sampler ("dreaming") that starts from noise or from a sample.
  - Reconstructions next to the inputs.
- **Use with the other models.** The hidden activations can feed a classifier (RBM features → logistic layer). Several RBMs stacked and pretrained can initialize a feed-forward network or the autoencoder, as in a deep belief network.
- **Export.** ONNX for the deterministic visible → hidden probabilities (`MatMul`, `Add`, `Sigmoid`), and for the reconstruction. Sampling and PMML are not supported.
- **Changes needed.** A new `ModelDefinition` subtype and editor form, an `InspectableNetwork` adapter for RBMs, a CD trainer with its own listener events (phase, Gibbs step), and MLflow parameters and metrics for those concepts.

### 3. Convolutional neural network (CNN)

A CNN uses small shared kernels instead of full connections. It is the most practical of the three for images, and is expected to bring MNIST from about 90% to about 99% test accuracy. It is also the largest change, because layers would work on height x width x channels instead of flat vectors.

- **Model.** New layer kinds in `network.layers`: `conv2d` (filters, kernel size, stride, padding), `maxPool`/`avgPool`, `flatten` and `dropout`, followed by the existing dense layers. For example, `28x28x1 → conv 8@3x3 relu → maxPool 2 → conv 16@3x3 relu → maxPool 2 → dense 10`. A softmax output with cross-entropy loss would come with it, which also helps the existing classifiers.
- **Training.** Backpropagation through convolution and pooling, mini-batches, and possibly momentum or Adam. Per-sample SGD in plain Java will be slow, so this needs efficient loops (im2col or vectorized arrays) and a background worker that trains more often than it takes snapshots.
- **Visualization.**
  - Feature maps per layer for the current image, as a grid of small heatmaps.
  - The kernels themselves.
  - The input pixels each output pixel depends on (its receptive field), traced back on hover.
  - The pooling choices (which input won each max).
  - Saliency or occlusion maps of the input pixels that drove a prediction.
  - The confusion matrix and gallery as for MNIST today. The network canvas would show each layer as a stack of maps rather than individual neurons.
- **Export.** ONNX `Conv`, `MaxPool`/`AveragePool` and `Flatten` (opset 13), checked against onnxruntime like the current exporter. PMML has no convolutions, so it is disabled for CNNs.
- **Changes needed.** Shaped tensors in `NetworkSpec` and the snapshots, the new layers in the trainer and in validation (shape checks, a parameter limit), the editor and JSON for layer kinds, and a tiled feature-map view.

### Shared groundwork

- Generalize `TrainingUpdate` and the listener API to describe any model's update (not just per-layer deltas), so that the UI, MLflow logging and exports stay model-agnostic.
- Mini-batch training and a softmax/cross-entropy option are useful for all three models, and for the existing classifiers.
- Each new model kind gets fake-server tests and smoke-test coverage, like the current examples, and at least one built-in example (MNIST for all three; iris for the autoencoder).

## Build and checks

The Maven setup targets Java 21, explicitly configures Lombok annotation processing (including newer JDK support), and uses the real desktop launcher rather than the template's nonexistent main class. Unused MapStruct/Mockito dependencies and the overwriting fat-JAR assembly were removed. JaCoCo reports are generated in `target/site/jacoco`; the previous template's blanket per-package coverage gate and unused mutation/static-analysis configuration are not part of the default build.

Tests cover finite-difference backpropagation across multiple hidden layers and outputs, immutable snapshots, cancellation, validation, truth tables, the expression language, JSON round trips and validation, initializer reproducibility, the model catalog, and convergence/interpolation of every built-in example. They also cover ONNX/PMML export, comparing against onnxruntime (a test-only dependency) and an independent PMML evaluator, the MLflow client, basic authentication and run logger, against an in-process fake MLflow server, and JSON syntax highlighting. The Parquet reader is tested against fixtures written by pyarrow (all column types, Snappy/gzip/plain, nested image structs across row groups; regenerate with `src/test/resources/data/make_fixtures.py`), and the Hugging Face client (including column statistics), the image and table dataset caches, scaling, held-out and split test rows, scaled ONNX/PMML exports and end-to-end image and table training runs against an in-process fake Hugging Face server, so no test needs network access. An optional desktop smoke test exercises stepping, stopping, inference, model switching, seeded randomization, JSON/ONNX/PMML export, the JSON viewer, MLflow run logging (against the fake server), the editor, image classification (download, gallery, inference, training), tabular datasets (download, raw-value inference, training, editor **Load info**) and convergence:

```powershell
mvn "-Dtest=VisualizationSmokeTest" "-Djavafx.smoke=true" test
```

The dependency audit is opt-in because it requires external vulnerability feeds:

```powershell
mvn -Psecurity-audit verify
```

If Maven reports an unreachable corporate mirror, correct your Maven `settings.xml` or use `mvn -s <settings-file> ...` with settings appropriate to your network; the project itself uses Maven Central.
