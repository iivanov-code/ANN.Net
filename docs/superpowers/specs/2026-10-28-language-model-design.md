# Language model helper design

Date: 2026-10-28

## Goal

Tie the tokenizer, the optional word2vec embeddings and the transformer into a
language model that can be trained on token sequences, evaluated with
perplexity and used to generate text, so the caller no longer writes the
window, one-hot and sampling code by hand.

## Decisions

- **Composition, not a new network.** `LanguageModel` owns an `ITokenizer`, an
  `ISequenceNetwork` and optionally an `IEmbeddingNetwork`, all built by the
  caller through the factory. Nothing in it knows about transformer internals.
- **Next-token windows.** Every sequence gets the configured begin and end
  special tokens, then is cut into windows of `ContextLength + 1` tokens that
  start every `WindowStride` tokens (default `ContextLength`, so consecutive
  windows overlap by one token and every token is predicted once). The input
  of a window is all but its last token, the target all but its first.
- **Input vectors** are the word2vec embeddings when an embedding network is
  given (fixed features; the transformer's input projection learns on top of
  them) and one-hot vectors over the vocabulary otherwise.
- **Loss and perplexity.** `Train` returns the mean cross entropy per predicted
  token of one epoch; `Evaluate` computes the same number from the propagated
  probabilities without touching the weights; `Perplexity` is its exponential.
- **Generation** re-propagates the last `ContextLength` tokens for every new
  token and picks the next one from the last position's distribution: greedy
  when `Temperature` is 0, otherwise sampling after dividing the log
  probabilities by the temperature and optionally keeping only the `TopK` most
  likely tokens. Generation stops at the end token or after `MaxNewTokens` and
  returns only the new tokens.
- **Validation at build time.** `ISequenceNetwork` gains `InputDimension`,
  `OutputDimension`, `MaxSequenceLength` and `OutputApplySoftMax`, so the
  factory can reject a network whose output is not a distribution over the
  tokenizer's vocabulary, whose input does not match the embedding dimension,
  or whose maximum sequence is shorter than the context length. The tokenizer
  must therefore be trained before the model is built.

## Public surface

```csharp
public interface ISequenceNetwork
{
    // existing members, plus
    int InputDimension { get; }
    int OutputDimension { get; }
    int MaxSequenceLength { get; }
    bool OutputApplySoftMax { get; }
}

public class LanguageModelSettings
{
    public ushort ContextLength { get; set; }           // <= network.MaxSequenceLength
    public ushort WindowStride { get; set; }            // 0 = ContextLength
    public string BeginToken { get; set; } = "<bos>";   // null or empty = none
    public string EndToken { get; set; } = "<eos>";     // null or empty = none
    public bool ShuffleWindows { get; set; } = true;
    public int? Seed { get; set; }
}

public class GenerationSettings
{
    public ushort MaxNewTokens { get; set; } = 64;
    public float Temperature { get; set; } = 1f;        // 0 = greedy
    public ushort TopK { get; set; }                    // 0 = all tokens
    public bool StopAtEndToken { get; set; } = true;
}

public interface ILanguageModel
{
    ITokenizer Tokenizer { get; }
    ISequenceNetwork Network { get; }
    IEmbeddingNetwork Embeddings { get; }               // null when one-hot
    int ContextLength { get; }
    Quad Train(IEnumerable<int[]> sequences);           // one epoch, mean loss per token
    Quad Evaluate(IEnumerable<int[]> sequences);        // mean loss, no update
    Quad Perplexity(IEnumerable<int[]> sequences);
    Quad[] PredictNext(int[] prompt);                   // distribution over the next token
    int[] Generate(int[] prompt, GenerationSettings settings);
    string Generate(string prompt, GenerationSettings settings);
}

ILanguageModel NetworkFactory.BuildLanguageModel(LanguageModelSettings settings,
                                                 ITokenizer tokenizer,
                                                 ISequenceNetwork network,
                                                 IEmbeddingNetwork embeddings = null)
```

## Implementation

- `TokenWindows.Build(sequences, contextLength, stride, beginToken, endToken)`
  returns the windows as `int[]` of length 2 to `ContextLength + 1`; shorter
  leftovers are dropped.
- `TokenSampler.Sample(probabilities, temperature, topK, random)` implements
  greedy, temperature and top-k selection.
- `LanguageModel` encodes windows to `Quad[][]`, builds one-hot targets, runs
  `Propagate` / `Backpropagate`, shuffles windows with Fisher-Yates when
  configured, and keeps a lazily created `Random` seeded from the settings.

## Error handling

Build rejects null tokenizer or network, a context length of 0 or above the
network maximum, a network without soft-max output, dimension mismatches and a
begin or end token name the tokenizer does not know. Token ids outside the
vocabulary throw `ArgumentOutOfRangeException`. Generating from an empty prompt
without a begin token throws `ArgumentException`. A negative temperature throws.

## Testing

xunit tests in `ANN.Net.Tests/LanguageModels` with a tiny word tokenizer and a
scripted sequence network as fakes:

- windows get the special tokens, overlap by one, honour the stride and drop
  leftovers shorter than two tokens,
- the sampler is greedy at temperature 0 and with top-k 1, never leaves the
  top-k set and roughly follows the distribution at temperature 1,
- build validation throws for every mismatch,
- a small transformer trained on a periodic token sequence lowers its loss and
  continues the sequence greedily,
- evaluation leaves the weights unchanged and perplexity is the exponential of
  the loss,
- generation stops at the end token, honours `MaxNewTokens`, returns only new
  tokens, feeds at most `ContextLength` tokens to the network, and the string
  overload decodes the continuation.

## Out of scope

Saving and loading, padding masks and mini-batches, dropout, learning rate
schedules, a key-value cache for generation, beam search and nucleus sampling.
