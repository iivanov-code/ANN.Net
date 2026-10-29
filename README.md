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

## Transformer

`NetworkFactory.BuildTransformer` builds an encoder-only transformer that maps a
sequence of vectors to a sequence of vectors of the same length. It follows the
same conventions as the other networks: every trainable value owns an optimizer
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
other networks every weight owns an optimizer and `Backpropagate` updates the
weights immediately, so the learning rate is set the same way. The input and
output matrices hold `2 * VocabularySize * EmbeddingDimension` weights, each
with its own optimizer, so size the vocabulary accordingly.

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
