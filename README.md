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

Neuron graphs become a few weight matrices evaluated level by level (the same plan the ONNX export uses), and
batch normalizations that follow a convolution are folded into it. Convolutions run as one matrix product per
output position over all filters, with SIMD. Convolutional networks compute as with
`Training = false`. A compact network is a snapshot: training the original later does not change it. It keeps no
state between calls, so one copy can serve several threads at once. `NetworkCompactor.CompactGraph(settings)`
compiles `GraphNetworkSettings` straight into a compact network without building the graph network first, which is
how NEAT evaluates its genomes. Recurrent networks and transformers cannot be
compacted yet.

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
