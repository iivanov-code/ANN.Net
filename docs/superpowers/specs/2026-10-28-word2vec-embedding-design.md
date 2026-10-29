# Word2Vec embedding network design

Date: 2026-10-28

## Goal

Add a word2vec network to ANN.Net that learns a dense vector per token id, so
the ids produced by an `ITokenizer` can be fed to the transformer as learned
embeddings instead of one-hot vectors. The network ships with its training
procedure (windows, negative sampling, subsampling) but is trained by the caller.

## Decisions

- **Both word2vec architectures.** `SkipGram` predicts every context token from
  the center token; `ContinuousBagOfWords` predicts the center token from the
  mean of its context. Selected through the settings.
- **Negative sampling** instead of hierarchical soft-max: every sample scores
  one real context token and `NegativeSamplesCount` noise tokens with a sigmoid
  and is trained with binary cross entropy. Noise tokens are drawn from the
  unigram distribution raised to `NoiseDistributionPower` (0.75 as in the paper).
- **Two matrices.** Input embeddings start uniform in
  `[-0.5 / dimension, 0.5 / dimension]`, output embeddings start at zero, as in
  the reference implementation. `GetEmbedding` returns the input vector.
- **Library conventions.** `Propagate` / `Backpropagate` are the primitives and
  `Backpropagate` updates the weights immediately. Every weight owns an optimizer
  of the configured `OptimizerTypes`, registered in `Optimizers` so the caller
  sets the learning rate the same way as for the transformer. Interface and
  settings live in `ANN.Net.Abstractions`, the implementation is `internal` in
  `ANN.Net/Embeddings` and is exposed through `NetworkFactory.BuildWord2Vec`.
- **One epoch per call.** `Train(sequences)` runs a single pass over the token
  sequences: counts tokens, builds the noise distribution, optionally drops
  frequent tokens (`SubsamplingThreshold`), slides a window of `WindowSize`
  (shrunk to a random size per center token when `UseDynamicWindow` is on) and
  calls `Propagate` / `Backpropagate` per sample. It returns the mean loss per
  sample so the caller can loop epochs and watch convergence.
- **Optional seed.** `Seed` fixes initialization, window shrinking, subsampling
  and negative sampling, which makes training reproducible and the tests stable.

## Public surface

```csharp
public enum Word2VecArchitecture { SkipGram, ContinuousBagOfWords }

public class Word2VecSettings
{
    public int VocabularySize { get; set; }                 // ids are in [0, VocabularySize)
    public ushort EmbeddingDimension { get; set; } = 100;
    public Word2VecArchitecture Architecture { get; set; }  // SkipGram
    public ushort WindowSize { get; set; } = 5;             // tokens on each side of the center
    public bool UseDynamicWindow { get; set; } = true;
    public ushort NegativeSamplesCount { get; set; } = 5;
    public float NoiseDistributionPower { get; set; } = 0.75f;
    public float SubsamplingThreshold { get; set; } = 1e-3f; // 0 disables
    public int? Seed { get; set; }
    public OptimizerTypes OptimizerType { get; set; }
}

public interface IEmbeddingNetwork
{
    int VocabularySize { get; }
    int EmbeddingDimension { get; }
    Quad[][] Embeddings { get; }                            // live input matrix, one row per id
    Quad[] Values { get; }                                  // scores of the last propagation
    Quad[] Errors { get; }                                  // score - target of the last backpropagation
    ICollection<LearningRateOptimizer> Optimizers { get; }

    Quad[] Propagate(int[] inputTokens, int[] outputTokens); // sigmoid(mean(in[inputs]) . out[o]) per output
    Quad Backpropagate(Quad[] targets);                      // 1 = context, 0 = noise; returns total loss
    Quad Train(IEnumerable<int[]> sequences);                // one epoch; returns mean loss per sample

    Quad[] GetEmbedding(int token);                          // copy
    Quad[][] Embed(int[] tokens);                            // one copy per token
    Quad CosineSimilarity(int first, int second);
    IReadOnlyList<(int Token, Quad Similarity)> MostSimilar(int token, int count);
}

IEmbeddingNetwork NetworkFactory.BuildWord2Vec(Word2VecSettings settings)
```

## Implementation

`Word2VecNetwork`

- `inputEmbeddings`, `outputEmbeddings : Quad[V][D]` with matching
  `IOptimizer[V][D]` tables created eagerly, like `LinearProjection`.
- `Propagate`: `hidden = mean of inputEmbeddings[inputTokens]`,
  `score[k] = sigmoid(hidden . outputEmbeddings[outputTokens[k]])`. Remembers
  the tokens and the hidden vector.
- `Backpropagate`: `g[k] = score[k] - target[k]`; the hidden gradient is
  `sum(g[k] * out[o_k])` computed before any update, then every output row gets
  `g[k] * hidden` and every input row gets `hiddenGradient / inputCount`. The
  loss is `-sum(t log s + (1 - t) log(1 - s))` with the scores clamped away
  from 0 and 1.
- `Train`: materializes the sequences, counts tokens, builds a
  `NoiseDistribution` (cumulative `count^power` table, sampled by binary
  search), subsamples with the keep probability
  `(sqrt(f / t) + 1) * t / f`, then per center token and window produces the
  samples described above. Negatives equal to the positive token are redrawn
  a few times.
- Similarity helpers work on the input matrix; a zero vector has similarity 0.

## Error handling

Settings are validated in the constructor (`VocabularySize >= 2`,
`EmbeddingDimension >= 1`, `WindowSize >= 1`, `NegativeSamplesCount >= 1`,
non-negative power and threshold). Token ids outside the vocabulary throw
`ArgumentOutOfRangeException`; `Backpropagate` before `Propagate` throws
`InvalidOperationException`; a targets length that differs from the last
output tokens throws `ArgumentException`.

## Testing

xunit tests in `ANN.Net.Tests/Embeddings`:

- propagation returns one score in (0, 1) per output token and exposes `Values`,
- repeated backpropagation of one sample drives its loss down,
- skip-gram and CBOW trained on a two-cluster corpus put tokens of the same
  cluster closer than tokens of different clusters, and `MostSimilar` returns
  cluster mates,
- the epoch loss decreases over epochs and is reproducible with a seed,
- `GetEmbedding` / `Embed` return copies of the right shape,
- the noise distribution never draws zero-count tokens and favours frequent ones,
- invalid settings, out-of-range ids and wrong call order throw.

## Out of scope

Hierarchical soft-max, learning rate decay schedules, phrase detection,
loading pre-trained vector files, and wiring the embeddings into the
transformer as a trainable layer.
