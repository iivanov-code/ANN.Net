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
