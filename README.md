## License

Copyright (C) 2021-2026 Ivelin Kirilov Ivanov

ANN.Net is licensed under the [GNU General Public License v3.0](LICENSE).
You may use, modify and redistribute it, provided that any distributed
modifications or derivative works are also released under GPLv3 with
source code made available.

 # ANN.Net Artificial Neural Network
C# implementation of standart deep neural network, with different types of neuro activation functions.


<p align = "center">
   <img src = "https://img.shields.io/github/languages/count/xtremista89/ANN.Net?style=for-the-badge">
   <img src = "https://img.shields.io/github/license/xtremista89/ANN.Net?style=for-the-badge">
   <img src = "https://img.shields.io/github/repo-size/xtremista89/ANN.Net?style=for-the-badge">
   <img src = "https://img.shields.io/github/languages/top/xtremista89/ANN.Net?style=for-the-badge">
   <img src = "https://img.shields.io/github/last-commit/xtremista89/ANN.Net?style=for-the-badge">
  </p>
<br>

## Compact networks for inference

A trained network keeps an object for every neuron, synapse, layer and optimizer. `NetworkCompactor.Compact`
copies it into flat weight arrays for inference:

```csharp
ICompactNetwork compact = NetworkCompactor.Compact(network);   // layered, graph or convolutional
Quad[] outputs = compact.Propagate(inputs);                    // the same values as network.Propagate
network = null;                                                // the original can now be freed
```

| Network | Memory | One propagation |
|---|---|---|
| Layered 784-256-256-10 | 80.3 MB → 1.1 MB | 60.7 ms → 0.036 ms |
| ResNet-18, 224 × 224 | 555 MB → 46.7 MB | 7.9 s → 0.08 s |

### 16-bit and 8-bit weights

`CompactOptions.Precision` stores the weight matrices as 16-bit floats or as 8-bit integers with one scale per neuron
or filter (symmetric quantization, the largest weight maps to ±127). Biases, input normalization and batch
normalization stay 32-bit.

```csharp
ICompactNetwork small = NetworkCompactor.Compact(network, new CompactOptions { Precision = WeightPrecision.Int8 });
byte[] onnx = OnnxExporter.Export(network, new OnnxExportOptions { WeightPrecision = WeightPrecision.Int8 });
```

| | Single | Half | Int8 |
|---|---|---|---|
| ResNet-18 weights | 46.7 MB | 23.4 MB | 11.7 MB |
| ResNet-18, one image | 81 ms | 86 ms | 84 ms |
| 784-256-256-10, one propagation | 36 µs | 137 µs | 42 µs |
| Trained 10-48-48-4 classifier, test accuracy | 98.4 % | 98.4 % | 98.4 % |
| Largest output change of that classifier | — | 0.001 | 0.014 |

16-bit weights cost conversion time when every weight is used once per propagation, as in dense layers; batches
(below) and convolutions convert every weight row once for many inputs. `OnnxExportOptions.WeightPrecision` writes the same
weights to ONNX (float16 initializers cast to float, int8 initializers with `DequantizeLinear`), so the exported
graph computes what the compact network of that precision computes. ONNX Runtime may then also quantize the inputs
of the matrix products it fuses with `DequantizeLinear`, which changes the outputs by a little more rounding.

### Speed

```csharp
var outputs = new float[compact.OutputCount];
compact.Propagate(inputs, outputs);                  // spans: no allocation in graph and layered networks
float[][] results = compact.PropagateBatch(samples); // many samples on all processors
```

`PropagateBatch` gives exactly the outputs of one sample at a time. Graph and layered networks compute blocks of 32
samples together, converting every 16-bit and 8-bit weight row once per block, and spread the blocks over the
processors; convolutional networks spread the samples. A single convolution large enough spreads its output rows
over the processors. Matrix products use SIMD and four weight rows per pass, and rows whose input is 0 (after ReLU,
or the padding of a convolution) are skipped. On 4 cores:

| | One sample | Batch, per sample |
|---|---|---|
| 784-256-256-10, single / half / int8 | 36 / 137 / 42 µs | 12 / 15 / 14 µs |
| ResNet-18, single / half / int8 | 81 / 86 / 84 ms | 50 / 66 / 58 ms |

### Pruning

`CompactOptions.PruneThreshold` removes the weights below a magnitude, `PruneFraction` the given share of the
smallest weights across the whole network. In neuron graphs a neuron left without inputs computes a constant that is
added to the biases of the neurons it feeds, and neurons that feed nothing any more are removed, so whole branches
disappear. Matrices that end up mostly zero are stored as their non-zero weights with column indexes (6 bytes per
float weight, 3 per int8 weight), which pays off once less than about a third (int8: a sixth) of them remain.
Biases are never pruned.

```csharp
var options = new CompactOptions { PruneFraction = 0.8, Precision = WeightPrecision.Int8 };
ICompactNetwork pruned = NetworkCompactor.Compact(network, options);
```

| Trained 10-48-48-4 classifier | Accuracy | Size |
|---|---|---|
| Not pruned | 98.4 % | 12,744 bytes |
| 70 % of the weights pruned | 98.8 % | 8,212 bytes (int8: 2,863) |
| 80 % pruned | 97.9 % | 6,552 bytes (int8: 2,358) |
| 90 % pruned | 26.8 % | 2,378 bytes |

Pruning does not retrain the network, so how much it tolerates depends on the network; check the accuracy.

### Saving compact networks

```csharp
CompactNetworkSerializer.Save(compact, "model.annc");
ICompactNetwork loaded = CompactNetworkSerializer.Load("model.annc");
```

The binary file holds the weights as they are stored (32-bit, 16-bit or 8-bit, dense or sparse), so it is about
`SizeInBytes` long — 11.7 MB for an int8 ResNet-18 — and the loaded network computes bit for bit what the saved one
computes. Loading checks that every operation fits the one before it and throws `InvalidDataException` for a damaged
file. A compact network cannot be trained; save the network itself with `NetworkSerializer` for that, or export it
to ONNX for other tools.

Neuron graphs become a few weight matrices evaluated level by level (the same plan the ONNX export uses), and
batch normalizations that follow a convolution are folded into it. Convolutions run as one matrix product per
output position over all filters, with SIMD. Convolutional networks compute as with
`Training = false`. A compact network is a snapshot: training the original later does not change it. It keeps no
state between calls, so one copy can serve several threads at once. `NetworkCompactor.CompactGraph(settings)`
compiles `GraphNetworkSettings` straight into a compact network without building the graph network first, which is
how NEAT evaluates its genomes.

### Transformers

`NetworkCompactor.CompactSequence(transformer, options)` copies a transformer into one weight matrix per projection,
applied to all tokens of a sequence at once. With 32-bit weights it computes bit for bit what the transformer
computes; precision, pruning, `PropagateBatch` and `CompactNetworkSerializer.Save`/`LoadSequence` work as for other
compact networks.

| d = 256, 8 heads, 6 layers, 4.8 M parameters, 64 tokens | Memory | One sequence | Batch of 8, per sequence |
|---|---|---|---|
| Transformer, trained with Adam | 57.4 MB | 570 ms | — |
| Compact, single | 19.3 MB | 40 ms | 15 ms |
| Compact, half | 9.7 MB | 41 ms | 13 ms |
| Compact, int8 | 5.0 MB | 39 ms | 15 ms |

Recurrent networks cannot be compacted yet: [LSTM and GRU networks](#recurrent-networks-lstm-and-gru) keep their
weights in arrays already, but the compactor does not take them; `BuildRNN` and `BuildCellRNN` return null, and
layers with `NeuronType.RNN` in `BuildFFN` connect the neurons of a layer to each other within one propagation instead
of to the previous time step, so they keep no state between inputs.

## Training in batches on all processors

Matrix, convolutional and transformer networks train mini-batches on all processors with `TrainBatch`:

```csharp
Quad loss = network.TrainBatch(inputs, targets);                                    // mean loss of the batch
loss = network.TrainBatch(inputs, targets, new BatchOptions { ChunkSize = 4, MaxDegreeOfParallelism = 2 });
```

The batch is split into chunks of `ChunkSize` samples (8 by default). Every chunk runs on one thread on a replica of
the network that shares its weights but keeps its own per-sample state, and whose optimizers only add up gradients.
The weights do not change during the batch; afterwards every weight is updated once with the mean gradient, the
chunks added in a fixed order. The result therefore depends on the chunk size but not on the number of processors:
the same training gives the same network on every machine. A batch of one sample trains exactly like `Propagate`
and `Backpropagate`. Batch normalization normalizes every image by its own statistics, as when training image by
image, and its running averages advance image by image in batch order. Every chunk keeps a copy of the gradients, so a
batch needs about (batch size / chunk size) x 4 bytes per parameter more memory; the replicas are kept for the next
batch. Transformers with neuron feed-forward networks cannot train in batches.

| 4 cores, Adam | One sample at a time | `TrainBatch` | |
|---|---|---|---|
| Matrix 784-256-256-10, batch 64 | 2,485 µs per sample | 720 µs | 3.5× |
| Small ResNet, 3 × 32 × 32, batch 32 | 119 ms per image | 34 ms | 3.5× |
| Transformer d = 64, 32 tokens, batch 32 | 15.6 ms per sequence | 4.2 ms | 3.7× |

## How many parameters fit in memory

Measured bytes per weight (for a 256 M parameter matrix network the measurement matches to two decimals) and what
that means for 32 GB (32 GiB, before the room the operating system and runtime need):

| Network | Bytes per parameter | Parameters in 32 GB |
|---|---|---|
| Neuron network (`BuildFFN`), plain gradient descent | 70 | 0.49 billion |
| Neuron network, Adam | 125 | 0.27 billion |
| Graph network (`BuildGraph`, NEAT), per connection | ≈ 200 | 0.17 billion |
| Matrix, convolutional or transformer network, plain gradient descent | 4 | 8.6 billion |
| … Momentum, Nesterov or RMSprop (one state value per weight) | 8 | 4.3 billion |
| … Adam, AdaMax or Nadam (two) | 12 | 2.9 billion |
| … AmsGrad (three) | 16 | 2.1 billion |
| Compact, 32-bit | 4 | 8.6 billion |
| Compact, 16-bit | 2 | 17 billion |
| Compact, 8-bit | 1 | 34 billion |
| Compact, 8-bit, 80 % pruned (3 bytes per kept weight) | 0.6 per original weight | 11 billion kept, of a 57 billion network |

One weight matrix (a layer, a convolution, a projection) holds at most about 2.1 billion weights, the length limit
of a .NET array. Training also keeps the activations of the last propagation (small for layered networks, tens of
MB per image for ResNet-18), compacting needs the original and the copy at once, and the JSON files of
`NetworkSerializer` hold an object per weight, so they do not suit networks of billions of weights; compact networks
save in their own binary format.

## Optimizers

Every trainable value is updated by an optimizer of the network's `OptimizerType`: convolutional networks,
transformers and embeddings keep one optimizer per weight array with its state in arrays (4 bytes per value for
`Normal`, 12 for Adam), layered and graph networks one per synapse, shared by a layer for `Normal`. Learning rates start at 0, so set them
before training (`network.Optimizers`):

| `OptimizerTypes` | Update | Typical learning rate |
|---|---|---|
| `Normal` | plain gradient descent | 0.01 – 0.1 |
| `Momentum` | classical momentum (β = 0.9) | 0.01 |
| `Nesterov` | Nesterov momentum, PyTorch form (β = 0.9) | 0.01 |
| `Adam` | Adam (β1 = 0.9, β2 = 0.999, ε = 1e-8) | 0.001 – 0.01 |
| `AdaMax` | Adam with the infinity norm | about twice the Adam rate |
| `Nadam` | Adam with Nesterov momentum | 0.001 – 0.01 |
| `AmsGrad` | Adam with the largest squared average so far | 0.001 – 0.01 |
| `RMSprop` | moving average of squared gradients (ρ = 0.9) | 0.001 – 0.005 |

The tests check every optimizer against its published update rule and train a layered network, a graph network, a
1-D convolutional layered network, a convolutional network and a transformer with each of them.

## Convolutional networks

`NetworkFactory.BuildConvolutional` builds a convolutional network from convolution, pooling and dense layers,
followed by a dense output layer. An image goes in as a flat array of channels × height × width values in channel,
row, column order (as in NCHW).

```csharp
IConvolutionalNetwork cnn = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
{
    InputChannels = 1,
    InputHeight = 28,
    InputWidth = 28,
    Layers =
    {
        new ConvolutionLayerSettings { Filters = 8, KernelSize = 3, Padding = 1, ActivationType = ActivationTypes.RELU },
        new PoolingLayerSettings { Type = PoolingType.Max, Size = 2 },
        new ConvolutionLayerSettings { Filters = 16, KernelSize = 3, Stride = 1 },
        new PoolingLayerSettings { Size = 2 },
        new DenseLayerSettings { NeuronsCount = 32, ActivationType = ActivationTypes.RELU }
    },
    OutputDimension = 10,
    OutputApplySoftMax = true,           // trained with cross entropy; otherwise the squared error is used
    OptimizerType = OptimizerTypes.Adam
});

foreach (LearningRateOptimizer optimizer in cnn.Optimizers)
{
    optimizer.LearningRate = 0.001f;
}

Quad[] probabilities = cnn.Propagate(image);       // 784 values
Quad loss = cnn.Backpropagate(oneHotTarget);       // trains on that image
```

- **Convolution layers** take a number of filters, a kernel size (`KernelSize`, or `KernelHeight` and `KernelWidth`),
  a stride, zero padding (smaller than the kernel) and an activation.
- **Pooling layers** take the max or the average over `Size` × `Size` windows, with a stride that defaults to `Size`.
- **Dense layers** flatten their input and may only be followed by other dense layers.
- **Batch normalization layers** normalize every channel and apply an activation. While `cnn.Training` is true
  (the default) they use the statistics of the current image and update running averages; set it to false for
  inference, which then uses the running averages, as an ONNX export does.
- **Residual blocks** are the basic blocks of ResNet-18/34: two 3×3 convolutions with batch normalization, added to
  the input (through a 1×1 convolution when the stride or the channels change).
- **Global average pooling** averages every channel; convolutions can be built without bias (`UseBias = false`)
  and pooling can be padded (`Padding`).

ResNet-18, as in torchvision, with exactly its 11,689,512 parameters:

```csharp
var resNet18 = new ConvolutionalNetworkSettings
{
    InputChannels = 3, InputHeight = 224, InputWidth = 224, OutputDimension = 1000, OutputApplySoftMax = true,
    Layers =
    {
        new ConvolutionLayerSettings { Filters = 64, KernelSize = 7, Stride = 2, Padding = 3, UseBias = false, ActivationType = ActivationTypes.Identity },
        new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.RELU },
        new PoolingLayerSettings { Type = PoolingType.Max, Size = 3, Stride = 2, Padding = 1 },
        new ResidualBlockSettings { Filters = 64 },  new ResidualBlockSettings { Filters = 64 },
        new ResidualBlockSettings { Filters = 128, Stride = 2 }, new ResidualBlockSettings { Filters = 128 },
        new ResidualBlockSettings { Filters = 256, Stride = 2 }, new ResidualBlockSettings { Filters = 256 },
        new ResidualBlockSettings { Filters = 512, Stride = 2 }, new ResidualBlockSettings { Filters = 512 },
        new GlobalAveragePoolingLayerSettings()
    }
};
```

Networks are trained one image at a time, so batch normalization sees a batch of one: it normalizes over the
positions of each channel. A full-size ResNet-18 forward pass takes several seconds on a CPU; the implementation
favours clarity over speed.

`Seed` (on `ConvolutionalNetworkSettings`, `TransformerSettings` and `NetworkSettings`) makes the initial weights
reproducible.

Every weight and bias array has an optimizer of `OptimizerType`, as in the transformer. Convolutional networks can be
saved and loaded like the other networks (`Save`, `ANNetwork.Load`, `NetworkSerializer`) and exported to ONNX
(`OnnxExporter`), where they become `Conv`, `MaxPool` / `AveragePool`, `Flatten`, `MatMul` and `Softmax` operators.

### 1-D convolution in layered networks

Hidden layers of `NetworkFactory.BuildFFN` can also be convolutional or deconvolutional (transposed convolution).
They treat the neurons of the previous layer, without its bias neuron, as a 1-D signal, and every filter shares its
kernel weights across all positions:

```csharp
var hidden = new RecurrentLayerSettings
{
    LayersCount = 1,
    CellType = NeuronType.Normal,
    PrevLayerConnectionType = LayerConnectionType.Convolutional,   // or Deconvolutional
    KernelSize = 3,
    Stride = 1,
    Padding = 1,               // smaller than the kernel
    FiltersCount = 4,          // the neurons count follows from these settings
    ActivationType = ActivationTypes.RELU
};
```

A bias neuron in the previous layer feeds every filter through one shared bias weight. Shared weights add up the
gradients of all their positions, so they usually need a smaller learning rate than plain weights.

## Free-form graphs (NEAT)

`NetworkFactory.BuildGraph` builds a feed-forward network with any acyclic topology, such as the
phenotype of a NEAT genome. Node ids become neuron ids, connection ids let you read the weights back after
training. Disabled connections and neurons that are not on a path from an input to an output are left out.

```csharp
IGraphNetwork network = NetworkFactory.BuildGraph(new GraphNetworkSettings
{
    Nodes = { new GraphNode(0, NodeType.INPUT), new GraphNode(1, NodeType.BIAS),
              new GraphNode(2, NodeType.HIDDEN), new GraphNode(3, NodeType.OUTPUT) },
    Connections = { new GraphConnection(0, 0, 2, 1.5f), new GraphConnection(1, 2, 3, -2f),
                    new GraphConnection(2, 0, 3, 0.5f), new GraphConnection(3, 1, 3, 0.1f) },
    HiddenActivationType = ActivationTypes.Sigmoid,
    OutputActivationType = ActivationTypes.Sigmoid
});

Quad[] output = network.Propagate(0.7f);
IReadOnlyDictionary<uint, Quad> weights = network.ConnectionWeights;
```

## Saving and loading

Networks from `NetworkFactory.BuildFFN` and `NetworkFactory.BuildGraph` are saved as JSON (`.ann` files)
and loaded back with the same topology, weights, activations, optimizer types, learning rates, loss,
normalization and soft-max settings. Optimizer internal state (such as Adam moments) is not saved.

```csharp
network.Save(@"C:\models", "my-network");                    // writes C:\models\my-network.ann
INetwork loaded = ANNetwork.Load(@"C:\models\my-network.ann");

// or with streams / a known network type
NetworkSerializer.Save(network, stream);
IGraphNetwork graph = NetworkSerializer.Load<IGraphNetwork>(stream);
```

## ONNX export

`OnnxExporter` writes networks from `BuildFFN` and `BuildGraph` (without recurrent connections) as
ONNX models (opset 13) that compute the same values as `Propagate`, including input normalization and
the output soft-max. Run them with ONNX Runtime or any other ONNX tool, from any language. ONNX keeps
only the inference graph; use `NetworkSerializer` to save a network you want to keep training.

```csharp
OnnxExporter.Export(network, @"C:\models\my-network.onnx");    // input "input" [batch, inputs], output "output" [batch, outputs]
OnnxExporter.Export(network, path, new OnnxExportOptions { InputName = "features", OutputName = "scores" });
```

```python
import onnxruntime as ort, numpy as np
session = ort.InferenceSession("my-network.onnx")
outputs = session.run(None, {"input": np.array([[0.5, -1.0]], dtype=np.float32)})[0]
```

## Transformer

`NetworkFactory.BuildTransformer` builds an encoder-only transformer that maps a
sequence of vectors to a sequence of vectors of the same length. It follows the
same conventions as the other networks: every weight array has an optimizer
from `OptimizerTypes`, and `Backpropagate` updates the weights immediately.

```csharp
ISequenceNetwork transformer = NetworkFactory.BuildTransformer(new TransformerSettings
{
    InputDimension = 4,          // size of every input vector
    ModelDimension = 8,          // d_model, must be divisible by HeadsCount
    HeadsCount = 2,
    FeedForwardDimension = 16,
    LayersCount = 2,
    OutputDimension = 4,
    MaxSequenceLength = 16,
    UseCausalMask = true,        // every position sees only itself and earlier positions
    OutputApplySoftMax = true,   // soft-max per output vector, trained with cross entropy
    OptimizerType = OptimizerTypes.Adam
});

foreach (LearningRateOptimizer optimizer in transformer.Optimizers)
{
    optimizer.LearningRate = 0.01f;
}

Quad[][] outputs = transformer.Propagate(sequence);   // one Quad[] per input vector
Quad totalError = transformer.Backpropagate(targets); // same shape as outputs
```

The feed forward sub-layer of every block comes in two interchangeable
implementations, chosen with `FeedForwardType`:

- `FeedForwardType.Matrix` (default): two dense projections over weight
  matrices, updated once per sequence.
- `FeedForwardType.NeuronGraph`: the same expand, activate, contract network
  built from neuron objects connected by synapses with `NetworkBuilder`, the
  way `NetworkFactory.BuildFFN` builds a classic network. It is applied to
  one position at a time, so its weights update after every position, and it
  is much slower than the matrix version; use it to study or debug the
  neuron graph, not for speed.

```csharp
settings.FeedForwardType = FeedForwardType.NeuronGraph;
```

## Recurrent networks (LSTM and GRU)

`NetworkFactory.BuildRecurrent` builds a stack of recurrent layers with a linear output, or a soft-max, at every
step. It is an `ISequenceNetwork` like the transformer, so it trains with `Propagate` and `Backpropagate` or
`TrainBatch` on all processors and uses every optimizer. The weights of a layer are one matrix with a row per value
of [x, h] and a column per gate unit, so a step is one pass over it.

| `CellType` | Step (z = [x(t), h(t-1)]) |
|---|---|
| `LSTM` | i, f, o = σ(W z + b), g = tanh(W_g z + b_g), c(t) = f ⊙ c(t-1) + i ⊙ g, h(t) = o ⊙ tanh(c(t)) |
| `LSTMPeepHole` | the same, and the gates also see the cell: p_i ⊙ c(t-1), p_f ⊙ c(t-1), p_o ⊙ c(t) (Gers and Schmidhuber) |
| `GRU` | r, u = σ(W z + b), n = tanh(W_n [x, r ⊙ h(t-1)] + b_n), h(t) = (1 - u) ⊙ n + u ⊙ h(t-1) (Cho et al.) |
| `RNN` | h(t) = tanh(W z + b) |

```csharp
IRecurrentNetwork lstm = NetworkFactory.BuildRecurrent(new RecurrentNetworkSettings
{
    InputDimension = 2,
    HiddenDimension = 8,         // size of h (and c) in every layer
    LayersCount = 1,             // every layer reads the hidden vectors of the one below
    OutputDimension = 2,
    CellType = NeuronType.LSTM,
    OutputApplySoftMax = true,   // cross entropy; otherwise linear outputs and the squared error
    ForgetGateBias = 3,          // forget gates (update gates of a GRU) start open; 1 by default
    GradientClipping = 1,        // largest L2 norm of the gradient of one sequence; 0 for none
    OptimizerType = OptimizerTypes.Adam,
    Seed = 1
});

foreach (LearningRateOptimizer optimizer in lstm.Optimizers) optimizer.LearningRate = 0.01f;

Quad[][] targets = new Quad[sequence.Length][];   // null: no target at that step
targets[^1] = label;                              // classify the whole sequence by its last output
lstm.Propagate(sequence);
lstm.Backpropagate(targets);                      // backpropagation through time
lstm.TrainBatch(sequences, targetsOfEach);        // or mini-batches on every processor

lstm.ResetState();                                // run inputs as they arrive, keeping the state
Quad[] output = lstm.Step(input);
```

Every `Propagate` starts from a zero state and trains through the whole sequence. Sequences can have any length.
The gradient of all weights is collected for the sequence, clipped if `GradientClipping` is set, and applied once.
`Step` gives exactly what `Propagate` gives at the same position.

The tests compare the gradient of every weight with finite differences, for every cell type, with two layers,
soft-max and linear outputs, and targets at every step or only the last one. They also train LSTM, peephole LSTM
and GRU networks to remember a bit across 40 noisy steps. In that task the bit comes at the first step on the same
input as noise that follows at every other step, a second input marks the step that counts, and the network gives
the bit after the last step. Measured with 8 hidden units, Adam at 0.01, clipping at 1, and up to 400 batches of 16
sequences. Each cell shows the seeds that reached 99 % (median batches):

| Cell | Gate bias | 30 steps | 60 steps | 100 steps |
|---|---|---|---|---|
| `RNN` | — | 4/5 (70) | 2/5 (100) | 2/5 (400) |
| `GRU` | 1 | 5/5 (100) | 0/5 | 0/5 |
| `GRU` | 3 | 5/5 (70) | 5/5 (70) | 5/5 (100) |
| `LSTM` | 1 | 5/5 (40) | 2/5 (160) | 0/5 |
| `LSTM` | 3 | 5/5 (40) | 5/5 (50) | 3/5 (50) |
| `LSTMPeepHole` | 3 | 5/5 (30) | 5/5 (40) | 5/5 (40) |

A gate bias of 1 is the usual default and lets the network forget quickly. For memories over more than about 30
steps, start the gates more open: at 3 the gated cells reach 100 steps, while the plain RNN stays unreliable.

On 4 cores, with 64 inputs, 128 hidden units, 10 soft-max outputs and sequences of 100 steps:

| Cell | Parameters | Forward | Forward and training | Batch of 32 (per sequence) |
|---|---|---|---|---|
| `RNN` | 25,994 | 0.57 ms | 1.27 ms | 23.3 ms (0.73 ms) |
| `GRU` | 75,402 | 1.22 ms | 3.12 ms | 49.5 ms (1.55 ms) |
| `LSTM` | 100,106 | 1.83 ms | 3.94 ms | 67.0 ms (2.09 ms) |

`LSTM`, `LSTMPeepHole` and `GRU` layers cannot be put into `BuildFFN`; it throws and points to `BuildRecurrent`.
Recurrent networks cannot be saved, compacted, exported to ONNX or put into nested networks yet, and a network must
not be used by several threads at once (`TrainBatch` uses its own copies).

## Neuron network

`NetworkFactory.BuildFFN` builds a classic fully connected network out of
neuron objects connected by synapses. Inputs pass through the input layer's
activation (`ActivationTypes.Identity` leaves them unchanged), every hidden
and output neuron sums its weighted inputs and activates, bias neurons feed a
constant 1 into the next layer, and `Backpropagate(targets)` feeds
`value - target` back through the graph, updating every synapse with its own
optimizer. `InputGradients` exposes the gradient at the inputs afterwards, so a
network can sit inside a larger model.

```csharp
INetwork network = NetworkFactory.BuildFFN(new NetworkSettings
{
    IdentityGenerator = new IncrementalGenerator(),
    LossType = LossFunctionTypes.Squared,
    InputSettings = new InputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true },
    OutputSettings = new OutputLayerSettings { NeuronsCount = 1, ActivationType = ActivationTypes.Sigmoid }
}, new RecurrentLayerSettings
{
    NeuronsCount = 8,
    LayersCount = 1,
    CellType = NeuronType.Normal,
    PrevLayerConnectionType = LayerConnectionType.FullyConnected,
    ActivationType = ActivationTypes.Sigmoid,
    HasBiasNeuron = true
});

Quad[] output = network.Propagate(0, 1);
Quad loss = network.Backpropagate(1);
```

### Matrix networks

`NetworkFactory.BuildMatrixFFN` takes the same settings and builds the same network with one weight matrix per
layer instead of neuron and synapse objects. With the same `Seed` it starts from the same weights and trains
bit-identically, so it is a drop-in replacement whenever the hidden layers are fully connected layers of normal
neurons. `NetworkFactory.ToMatrix` converts an existing network; the state of optimizers such as Adam starts afresh.
Matrix networks save, load, export to ONNX and compact like the neuron network.

```csharp
IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(settings, hiddenLayerSettings);
foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.01f;
```

| 784-256-256-10 | Neuron network | Matrix network |
|---|---|---|
| Memory, plain gradient descent | 18.8 MB (70 B per weight) | 1.1 MB (4 B per weight) |
| Memory, Adam | 33.7 MB (125 B per weight) | 3.2 MB (12 B per weight) |
| One training step, Adam | 57 ms | 3.9 ms |
| One propagation | 22 ms | 0.4 ms |

## Networks of networks

`NetworkFactory.BuildNested` builds a network whose nodes are whole networks: every node takes a vector, gives a
vector, and passes it to the nodes that read it. A node joins the vectors it receives one after the other
(`NodeCombination.Concatenate`) or adds them up (`NodeCombination.Sum`). Node 0 (`NestedNetworkSettings.InputNode`)
is the input of the whole network, and the output is the outputs of the listed nodes one after the other.

```csharp
IMatrixNetwork encoder = NetworkFactory.BuildMatrixFFN(encoderSettings, hidden);          // 16 -> 24
IVectorNetwork attention = NetworkFactory.AsVectorNetwork(transformer, sequenceLength: 4);  // 4 tokens of 6 -> 4 tokens of 6
INestedNetwork inner = NetworkFactory.BuildNested(innerSettings);                           // a network of networks itself

INestedNetwork network = NetworkFactory.BuildNested(new NestedNetworkSettings
{
    InputSize = 16,
    Nodes =
    {
        new NestedNode(1, encoder, NestedNetworkSettings.InputNode),
        new NestedNode(2, attention, 1),
        new NestedNode(3, inner, 2, NestedNetworkSettings.InputNode),                      // 24 + 16 values
    },
    Outputs = { 3 }
});

foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.01f;
network.Propagate(input);
network.Backpropagate(target);   // trains the networks of all nodes together
```

A node can hold any `IVectorNetwork`: matrix, neuron, graph, convolutional and nested networks are vector networks,
and `AsVectorNetwork` turns a transformer into one for sequences of a fixed length, so nested networks nest to any
depth. Backpropagation visits the nodes in reverse order, adds up the gradients from every node that reads a node,
and hands them to `BackpropagateGradient` of the network inside, which trains it and returns the gradient of its
inputs (through the normalization and soft-max of the inner network). Any class implementing `IVectorNetwork` can be
a node too. Rules: the nodes must not form a cycle, every node needs its own network, and nodes that feed no output
are left out. Nested networks save and load with `NetworkSerializer` when every node can be saved (transformers
cannot yet); they cannot be compacted or exported to ONNX yet.

## Spiking networks

A spiking neuron sends no numbers, only spikes: it adds the weights of the synapses whose source fired to its membrane
potential and fires when the potential reaches a threshold. Information is carried by how often neurons fire, and a
silent neuron costs no work. ANN.Net simulates spiking networks in discrete time steps with their own classes: the
synapses of the neuron networks wait every propagation until all inputs have delivered a value, which is the opposite
of sparse spikes, and their neurons forget everything between propagations, while a membrane potential has to last
from step to step. `SpikingSynapse` (source, target, weight) is stored in flat arrays grouped by source, 8 bytes per
synapse; a spike reads its source's weights one after another and adds them with SIMD when they go to consecutive
neurons, as in a fully connected layer. A synapse into a later layer delivers in the same step, a synapse within a
layer or back to an earlier one in the next step.

Every layer has a spike function, an `ISpikingActivation`, which can be replaced with `SetActivation`:

| Spike function | Fires for a constant input current c | Counterpart |
|---|---|---|
| `IntegrateAndFireActivation` | max(0, c) / threshold spikes per step; with `leak` below 1 a leaky integrate-and-fire neuron | ReLU |
| `SignedIntegrateAndFireActivation` | c / threshold spikes for c > 0, slope · c / threshold negative spikes for c < 0 | leaky ReLU (slope 1: identity) |
| `SigmoidRateActivation` | sigmoid(gain · c) spikes per step, the average input since the start | sigmoid (gain 2: tanh) |
| `PoissonActivation` | a random spike with probability \|c\| | random input encoding |

Integrate-and-fire neurons also take `resetToZero` (instead of subtracting the threshold) and `thresholdAdaptation`
(every spike raises the neuron's threshold, which decays by `adaptationDecay` per step). Any class implementing
`ISpikingActivation` can be a spike function.

### Train with ReLU or sigmoid, then convert

```csharp
IMatrixNetwork network = NetworkFactory.BuildMatrixFFN(settings, hidden);          // ReLU, leaky ReLU, identity, sigmoid or tanh
// ... train as usual ...

ISpikingNetwork spiking = SpikingNetworkConverter.Convert(network, new SpikingConversionOptions
{
    CalibrationInputs = trainingInputs.Take(500).ToList(),                          // float[] each
    TimeSteps = 16
});
float[] outputs = spiking.Propagate(input);                                          // about what network.Propagate computes
SpikingRun run = spiking.Run(input, timeSteps: 32);                                  // also counts spikes and synapse deliveries
```

`Convert` takes networks from `BuildFFN` (without recurrent connections), `BuildMatrixFFN` and `BuildGraph`;
`ConvertGraph` takes `GraphNetworkSettings` such as NEAT phenotypes. Every neuron becomes the spike function of its
activation. The calibration inputs normalize the inputs: each input is mapped onto [0, 1] (onto [-1, 1] with
`SignedInputs`) and fed every step as the current of an integrate-and-fire neuron, so an input of 0.25 fires every
fourth step. They also scale the weights so that the 99.9th percentile (`Percentile`) of every layer's activations
fires once per step. The outputs are read from the average input current of the output neurons and passed through the
original output activation and soft-max. Convolutional, nested and recurrent networks cannot be converted yet.

Measured with a 784-256-256-10 ReLU network (268,000 synapses) trained on 10 noisy prototypes, on one core:

| Time steps | Same class as the network | Output difference | Synapse deliveries | Time per input |
|---|---|---|---|---|
| 4 | 93 % | 0.22 | 0.9 × synapses | 71 µs |
| 8 | 100 % | 0.009 | 2.1 × | 187 µs |
| 16 | 100 % | 0.0008 | 4.3 × | 335 µs |
| 32 | 100 % | 0.0002 | 8.9 × | 679 µs |
| 64 | 100 % | 0.0001 | 18 × | 1,372 µs |
| compact network | | | 1 × (multiply and add) | 22.5 µs |

On a CPU a spiking network is therefore slower than a compact one, not faster: about a quarter of the neurons fire
in every step, every step repeats the work, and an addition costs a processor as much as a multiplication and
addition. Spiking networks save energy on neuromorphic chips, which spend power only on spikes. ReLU networks fire
least (a neuron with a negative input never fires), while sigmoid and tanh neurons fire about twice as often because
their zero is half the maximum rate. Time-to-first-spike coding needs far fewer spikes.

### Time-to-first-spike coding

```csharp
IFirstSpikeNetwork spiking = SpikingNetworkConverter.ConvertToFirstSpike(network, new SpikingConversionOptions
{
    CalibrationInputs = trainingInputs.Take(500).ToList(),
    TimeSteps = 256                                                                  // window of every layer
});
float[] outputs = spiking.Propagate(input);
IReadOnlyList<FirstSpike> spikes = spiking.GetLastSpikes();                         // neuron, step and sign of every spike
```

Here every neuron fires at most once: layer d fires in window d of W steps, and a neuron fires in step
round(W · (1 − |value| / scale)) of it, the larger its value the earlier; a value that rounds to 0 does not fire at
all. On a neuromorphic chip the spike switches on a constant current into every target that flows until the end of the
window, so a target receives weight · (W − step) / W; when its own window starts it has its whole input, and a ReLU
neuron fires when that input plus a rising ramp reaches its threshold. The simulation computes these spike steps
directly, one multiply-add per synapse of every neuron that fires, so the work does not grow with the window: a longer
window only gives finer values (1/W of every layer's scale) and a longer delay (`SpikingRun.TimeSteps`, one window per
layer). Neurons of any activation convert, also GELU and the like, with negative spikes for negative values.
`ConvertGraphToFirstSpike` converts graph settings such as NEAT phenotypes, connections that skip layers included.

The same 784-256-256-10 network, one core, best of 7 runs; once with 65 % of the inputs non-zero, once with 21 %
(as in handwritten digits):

| | Synapse deliveries | Time, 65 % non-zero | Time, 21 % non-zero |
|---|---|---|---|
| Rate coding, 16 steps | 4.3 × synapses | 335 µs | |
| First spike, window 16 (100 % same class) | 0.56 × / 0.25 × | 33 µs | 22 µs |
| First spike, window 256 | 0.59 × / 0.25 × | 38 µs | 23 µs |
| Compact network | the non-zero inputs | 22 µs | 11 µs |

Even a window of 4 steps gave the same classes, the outputs differing by 0.0002 on average. First-spike coding is 10
to 15 times faster than rate coding, but still about 1.5 to 2 times slower than the compact network: its saving is that
neurons with the value 0 cost nothing, and the compact network already skips zero inputs in its matrix products. What
is left is the bookkeeping of encoding every neuron's value as a spike step. Spikes therefore save processor time over
a dense network that multiplies zeros, not over one that skips them; their real advantage is on neuromorphic chips,
where first-spike coding needs the fewest spikes of all codings.

### Spiking networks of any topology

```csharp
ISpikingNetwork network = NetworkFactory.BuildSpiking(new SpikingGraphSettings
{
    Layers = { new SpikingGraphLayer(2), new SpikingGraphLayer(3, new IntegrateAndFireActivation(leak: 0.9f), learn: true), new SpikingGraphLayer(1) },
    Synapses = { new SpikingSynapse(0, 2, 0.8f), new SpikingSynapse(1, 3, 0.5f), new SpikingSynapse(0, 5, 1.2f), new SpikingSynapse(4, 4, 0.3f) },
    Bias = new[] { 0f, 0f, 0.1f, 0f, 0f, 0.05f },                                    // current per neuron and step
    Outputs = { 5, 4 },                                                              // neurons whose rates are the outputs
    TimeSteps = 40
});
```

Neurons are numbered across the layers, layer 0 holding the inputs, and a synapse may connect any two neurons except
into an input. A synapse into a later layer delivers in the same time step; one within a layer or back to an earlier
layer delivers in the next step, so cycles are allowed and give the network memory (a neuron exciting itself keeps
firing after its input made it fire once). NEAT.net builds the spiking networks of genomes this way.

### Learning with spike timing (STDP)

```csharp
// two neurons learn two patterns of 8 inputs (the setup of CompetingNeuronsLearnDifferentPatterns)
ISpikingNetwork network = NetworkFactory.BuildSpiking(new SpikingNetworkSettings
{
    InputCount = 8,                                                                 // inputs already in [0, 1]; else set InputScaling
    InputActivation = new PoissonActivation(seed: 0),
    TimeSteps = 40,
    Plasticity = new StdpOptions
    {
        PotentiationRate = 0.05f, DepressionRate = 0.03f,
        PotentiationTimeConstant = 4, DepressionTimeConstant = 4, IncomingWeightSum = 3
    },
    Layers =
    {
        new SpikingLayerSettings
        {
            NeuronsCount = 2, MinimumWeight = 0.2f, MaximumWeight = 0.5f, LateralInhibition = 3,
            Activation = new IntegrateAndFireActivation(leak: 0.9f, thresholdAdaptation: 0.05f, adaptationDecay: 0.999f)
        }
    }
});

for (int round = 0; round < 30; round++)
{
    network.Learn(first);                                                           // no labels
    network.Learn(second);
}
```

Spike-timing-dependent plasticity changes the synapses into layers with `Learn` set. When the target fires, a synapse
grows by `PotentiationRate` times its source's trace; when the source fires, it shrinks by `DepressionRate` times its
target's trace. A trace jumps by 1 at every spike of its neuron and decays with `PotentiationTimeConstant` or
`DepressionTimeConstant` steps, so sources that fire just before their target are strengthened and sources that fire
just after it are weakened. `SoftBounds` slows changes near `MinimumWeight` and `MaximumWeight`, and
`IncomingWeightSum` rescales each neuron's incoming weights after every input. STDP alone finds patterns but not
classes. Neurons that compete through `LateralInhibition` (the first to fire silence the others) and
`thresholdAdaptation` (frequent winners get harder to fire) learn different patterns: with the settings above, the two
neurons answered different patterns in 51 of 100 seeds before learning and in all 100 after it, each neuron's weights
from its own pattern ending up more than 12 times those from the other. The initial weights must differ (neurons with
equal weights receive the same input and stay identical), and random input spikes (`PoissonActivation`) keep two
neurons from reaching their thresholds in the same step and inhibiting each other equally. Larger inputs such as images need other
settings (thresholds, rates and weight sums grow with the number of inputs); label the learned neurons afterwards by
the class they answer most.

A spiking network keeps the state of its neurons, so one instance must not be used by several threads at once.
Spiking networks cannot be saved yet, and first-spike networks cannot learn with STDP.

## Splitting a neuron network across processes

`ANN.Net.Remote` runs the first layers of a neuron network in one process and
the remaining layers in another, connected through
[RPC.Library](https://github.com/xtremista89/RPC.Library) (checked out next to
this repository). Both processes describe the whole network the same way and
name the hidden layer whose outgoing synapses cross the wire. The head keeps
that layer's neurons; each of them gets a `RemoteSynapse`, a weightless axon
that sends the activation and receives the gradient. The tail is an ordinary
network whose identity input layer stands in for those neurons, so the weights
between the two layers live in the tail, and the whole boundary travels as one
message per propagation and one per backpropagation.

```csharp
NetworkSettings settings = /* input and output layers, as for BuildFFN */;
RecurrentLayerSettings[] hidden = { layer8, layer4 };   // two hidden layers

// process B: serve everything after hidden layer 0
INetwork tail = RemoteNetworkFactory.BuildTail(settings, hidden, boundaryLayer: 0);
using RemoteNetworkHost host = await RemoteNetworkHost.StartAsync(tail, new TcpTransport(), port: 8990);

// process A: the input layer and hidden layer 0, plus a remote synapse per neuron of that layer
SynapseConnection connection = await SynapseConnection.ConnectAsync(new TcpTransport(), "serverB", 8990);
using SplitNetwork network = RemoteNetworkFactory.BuildHead(settings, hidden, boundaryLayer: 0, connection);

network.SetLearningRate(0.1f);              // both sides
Quad[] output = network.Propagate(0, 1);    // outputs of the tail
Quad loss = network.Backpropagate(1);       // trains the tail, then the head
```

`SplitNetwork` is an `INetwork`, so training code does not change. The tail can
itself be a `SplitNetwork`, which chains three or more processes. The same
pieces work in one process with `new SynapseConnection(RemoteNetworkPart.Create(tail))`,
which is how the tests check the gradients without sockets.

### Breadth split: a run of layers shared by several machines

The depth cut leaves one side idle while the other computes. A breadth cut
spreads a run of hidden layers over several parts that each hold a slice of
every layer, so they are equally loaded and work on the same sample at the
same time; the price is one exchange of slices per layer. Each part keeps a
ghost identity neuron for every neuron the other parts own in the layer
before, so the weights stay where they are in one network, each in exactly one
part. Part 0 also coordinates: towards the stage in front it is an ordinary
remote part, behind it sits the tail (or the next section).

```csharp
// everybody: the same description; hidden layers 1 and 2 are shared by two parts
NetworkPartition partition = RemoteNetworkFactory.Partition(settings, hidden, firstLayer: 1, lastLayer: 2, parts: 2);

// machine 4: the tail behind the section
using RemoteNetworkHost tailHost = await RemoteNetworkHost.StartAsync(RemoteNetworkFactory.BuildTail(settings, hidden, 2), tcp, port: 8992);

// machine 2: part 0, coordinator; accepts part 1, connects to the tail, serves the section to the head
PartitionNode part0 = new PartitionNode(partition, 0);
await part0.StartAsync(tcp, new IPEndPoint(IPAddress.Any, 8991));
var toTail = await RemoteNetworkPeer.ConnectAsync(tcp, "machine4", 8992);
using PartitionCoordinator coordinator = new PartitionCoordinator(part0, toTail.Remote, toTail);
using RemoteNetworkHost sectionHost = await RemoteNetworkHost.StartAsync(coordinator, tcp, port: 8990);

// machine 3: part 1, connects to part 0 (parts with a higher index connect to the ones below)
PartitionNode part1 = new PartitionNode(partition, 1);
await part1.ConnectAsync(0, tcp, "machine2", 8991);

// machine 1: the head in front of the section
using SplitNetwork network = RemoteNetworkFactory.BuildHead(settings, hidden, 0, await SynapseConnection.ConnectAsync(tcp, "machine2", 8990));
```

`FourMachineSimulationTests` runs exactly this layout on four loopback
endpoints with random ports; `PartitionChainTests` runs it in one process with
`PartitionNode.ConnectInProcess`, which hops through the thread pool like the
messages would.

## Tokenizer

`NetworkFactory.BuildBpeTokenizer` builds a byte-level byte pair encoding
tokenizer behind the `ITokenizer` interface, so another tokenizer can be dropped
in later. The 256 byte values are the base vocabulary, which means any UTF-8 text
encodes without an unknown token and `Decode(Encode(text))` always returns the
original text. Merges never cross a word boundary (a word carries the whitespace
in front of it), and special tokens get the ids right after the bytes.

```csharp
ITokenizer tokenizer = NetworkFactory.BuildBpeTokenizer(new BpeTokenizerSettings
{
    VocabularySize = 1024,                       // 256 bytes + special tokens + learned merges
    MinimumPairFrequency = 2,                    // stop early when no pair repeats this often
    SpecialTokens = new List<string> { "<pad>", "<eos>" }
});

tokenizer.Train(lines);                          // IEnumerable<string>
int[] tokens = tokenizer.Encode("Здравей, свят!");
string text = tokenizer.Decode(tokens);
int endOfSequence = tokenizer.SpecialTokens["<eos>"];

// one-hot the ids to feed the transformer (InputDimension = tokenizer.VocabularySize)
Quad[][] sequence = new Quad[tokens.Length][];
for (int t = 0; t < tokens.Length; t++)
{
    sequence[t] = new Quad[tokenizer.VocabularySize];
    sequence[t][tokens[t]] = 1;
}
```

## Word2Vec embeddings

`NetworkFactory.BuildWord2Vec` builds a word2vec network behind the
`IEmbeddingNetwork` interface. It learns one dense vector per token id with
negative sampling, either as skip-gram (the center token predicts its context)
or as continuous bag of words (the context predicts the center token). Like the
other networks every embedding row has an optimizer and `Backpropagate` updates the
weights immediately, so the learning rate is set the same way. The input and
output matrices hold `2 * VocabularySize * EmbeddingDimension` weights, whose
optimizer state is kept in arrays, so size the vocabulary accordingly.

```csharp
IEmbeddingNetwork word2vec = NetworkFactory.BuildWord2Vec(new Word2VecSettings
{
    VocabularySize = tokenizer.VocabularySize,
    EmbeddingDimension = 64,
    Architecture = Word2VecArchitecture.SkipGram,
    WindowSize = 5,                // context tokens on each side of the center
    UseDynamicWindow = true,       // shrink the window at random, nearer tokens weigh more
    NegativeSamplesCount = 5,      // noise tokens scored against every real context token
    NoiseDistributionPower = 0.75f,
    SubsamplingThreshold = 1e-3f,  // drop very frequent tokens; 0 disables
    Seed = 42,                     // null for a different run every time
    OptimizerType = OptimizerTypes.Normal
});

foreach (LearningRateOptimizer optimizer in word2vec.Optimizers)
{
    optimizer.LearningRate = 0.025f;
}

// one call is one epoch; it returns the mean loss per sample
List<int[]> corpus = lines.Select(tokenizer.Encode).ToList();
for (int epoch = 0; epoch < 5; epoch++)
{
    Quad loss = word2vec.Train(corpus);
}

Quad[] vector = word2vec.GetEmbedding(tokenId);              // copy of the input embedding
Quad similarity = word2vec.CosineSimilarity(first, second);
var neighbours = word2vec.MostSimilar(tokenId, 10);          // (Token, Similarity), best first

// learned embeddings as transformer input (InputDimension = EmbeddingDimension)
Quad[][] sequence = word2vec.Embed(tokenizer.Encode(text));
```

`Propagate(inputTokens, outputTokens)` and `Backpropagate(targets)` are public
as well, for a custom training loop: the scores are the sigmoid of the dot
product between the mean input embedding of the input tokens and the output
embedding of every output token, and the targets are 1 for a real context token
and 0 for a noise token.

## Language model

`NetworkFactory.BuildLanguageModel` ties a trained tokenizer, optional word2vec
embeddings and a soft-max transformer into a next-token language model behind
`ILanguageModel`. It cuts token sequences into training windows, builds the
one-hot targets, trains one epoch per call, reports perplexity and generates
text. Build the pieces in this order, because the vocabulary size of the
trained tokenizer sizes the transformer.

```csharp
ITokenizer tokenizer = NetworkFactory.BuildBpeTokenizer(new BpeTokenizerSettings
{
    VocabularySize = 1024,
    SpecialTokens = new List<string> { "<bos>", "<eos>" }
});
tokenizer.Train(lines);
List<int[]> corpus = lines.Select(tokenizer.Encode).ToList();

ISequenceNetwork transformer = NetworkFactory.BuildTransformer(new TransformerSettings
{
    InputDimension = (ushort)tokenizer.VocabularySize,   // or the word2vec EmbeddingDimension
    ModelDimension = 64,
    HeadsCount = 4,
    FeedForwardDimension = 128,
    LayersCount = 2,
    OutputDimension = (ushort)tokenizer.VocabularySize,
    MaxSequenceLength = 64,
    UseCausalMask = true,          // required for next-token prediction
    OutputApplySoftMax = true,     // required, the outputs must be distributions
    OptimizerType = OptimizerTypes.Adam
});
foreach (LearningRateOptimizer optimizer in transformer.Optimizers)
{
    optimizer.LearningRate = 0.001f;
}

ILanguageModel model = NetworkFactory.BuildLanguageModel(new LanguageModelSettings
{
    ContextLength = 64,            // tokens the model sees at once, <= MaxSequenceLength
    WindowStride = 0,              // 0 = ContextLength, windows overlap by one token
    BeginToken = "<bos>",          // put in front of every sequence and every prompt
    EndToken = "<eos>",            // appended to every sequence, stops generation
    ShuffleWindows = true,
    Seed = 42
}, tokenizer, transformer);        // pass the word2vec network as 4th argument to embed with it

for (int epoch = 0; epoch < 10; epoch++)
{
    Quad trainLoss = model.Train(corpus);               // mean cross entropy per predicted token
    Quad perplexity = model.Perplexity(validation);     // exp of the validation loss, no weight update
}

string continuation = model.Generate("Once upon a", new GenerationSettings
{
    MaxNewTokens = 50,
    Temperature = 0.8f,            // 0 = greedy
    TopK = 40,                     // 0 = sample from the whole vocabulary
    StopAtEndToken = true
});
Quad[] next = model.PredictNext(tokenizer.Encode("Once upon a"));  // distribution over the next token
```

With word2vec embeddings the vectors are fixed input features and the
transformer's input projection learns on top of them; without them the
input is one-hot and the input projection is the embedding table. Generation
re-propagates the last `ContextLength` tokens for every new token.
