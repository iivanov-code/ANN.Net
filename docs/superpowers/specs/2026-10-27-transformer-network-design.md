# Transformer network design

Date: 2026-10-27

## Goal

Add an encoder-style transformer network to ANN.Net that follows the conventions
already used by the library, so it feels like a sibling of `Network`, `LSTMCell`
and `GRUCell` rather than a foreign module.

## Observed conventions

- Two layers of projects: `ANN.Net.Abstractions` holds interfaces, enums, settings
  and the `Quad` numeric type; `ANN.Net` (folder `NeuralNetwork`) holds the
  implementations, which are `internal` and exposed through `NetworkFactory`.
- Configuration is a plain settings class (`NetworkSettings`, `RecurrentLayerSettings`)
  with enum-typed choices (`OptimizerTypes`, `WeightInitType`, `ActivationTypes`).
- Every trainable scalar owns an `IOptimizer` created by `NetworkUtils.GetOptimizer`,
  and every `LearningRateOptimizer` is registered in an `Optimizers` collection so a
  training strategy can set the learning rate on all of them at once.
- Optimizers are applied immediately during `Backpropagate` (online updates), with
  the gradient convention "weight = weight - learningRate * gradient".
- The newer cells (`LSTMCell`, `GRUCell`) work on `Quad[]` vectors through helper
  math instead of the per-neuron event graph. The transformer follows this
  vectorised style, since attention is a dense all-to-all operation.
- Method names are `Propagate` / `Backpropagate`; `Network` exposes `Values`,
  `Errors` and `Optimizers`.
- Code style: Allman braces, block-scoped namespaces, `using` directives outside
  the namespace with `System` first, explicit types, UTF-8 BOM, CRLF.

## Architecture

```
TransformerNetwork (ISequenceNetwork)
  inputProjection      : LinearProjection   InputDimension -> ModelDimension
  positionalEncodings  : sinusoidal table    MaxSequenceLength x ModelDimension
  blocks[LayersCount]  : TransformerEncoderBlock
        attentionNormalization   : LayerNormalization
        attention                : MultiHeadAttention (Q, K, V, O LinearProjections)
        feedForwardNormalization : LayerNormalization
        feedForward              : FeedForwardNetwork (two LinearProjections + activation)
  outputNormalization  : LayerNormalization
  outputProjection     : LinearProjection   ModelDimension -> OutputDimension
  optional soft-max on every output vector
```

Blocks use the pre-normalisation layout (`x + Attention(LN(x))`,
`h + FFN(LN(h))`) with a final normalisation, because it trains reliably with
plain gradient descent, which is the library's default optimizer.

Sequences are `Quad[][]` (one `Quad[]` per position), weights are `Quad[,]`.

## Public surface

- `ANN.Net.Abstractions.Settings.TransformerSettings`: `InputDimension`,
  `ModelDimension`, `HeadsCount`, `FeedForwardDimension`, `LayersCount`,
  `OutputDimension`, `MaxSequenceLength`, `UseCausalMask`, `OutputApplySoftMax`,
  `FeedForwardActivationType` (default RELU), `OptimizerType`, `InitializationType`.
- `ANN.Net.Abstractions.Interfaces.ISequenceNetwork`: `Propagate(Quad[][])`,
  `Backpropagate(Quad[][] targets)`, `Values`, `Errors`, `Optimizers`.
- `NetworkFactory.BuildTransformer(TransformerSettings)` returning `ISequenceNetwork`.

## Training semantics

`Backpropagate(targets)` mirrors `Network.Backpropagate`: the error signal per
output is `predicted - target`. That is the exact gradient both for mean squared
error on a linear output and for cross-entropy on a soft-max output, so one code
path serves both. The returned total is the summed loss (half squared error, or
cross-entropy when soft-max is on). Weight updates happen inside the call.

Activation derivatives in the feed-forward block receive the pre-activation value,
which is what `RELUActivation.Derivative` expects (the library default).

## Error handling

Constructor validates the settings (positive sizes, `ModelDimension` divisible
by `HeadsCount`). `Propagate` validates sequence length and vector sizes.
`Backpropagate` requires a previous `Propagate` and matching target shape.

## Testing

A new `ANN.Net.Tests` xunit project with:

- output shape and soft-max normalisation,
- causal mask: later tokens cannot influence earlier outputs,
- finite-difference gradient checks for `LinearProjection`, `LayerNormalization`,
  `MultiHeadAttention`, `FeedForwardNetwork` and a whole encoder block,
- a tiny next-token task whose loss must fall while training.

The test project targets net8.0 because the xunit runner available locally has
no net6.0 build; it references the net6.0 library normally.

## Pre-existing defects fixed on the way

Only unambiguous, one-line fixes that the transformer (or anything else) needs to
run at all:

- `Quad`'s implicit conversion to `float` returned the struct itself and recursed
  until stack overflow; it now returns the wrapped value.
- `Rate`'s constructor validated against `Rate.MaxValue` while that static field
  was still being initialised, so every `Rate` threw; it now checks the literal
  bounds 0 and 1.
- `ISequenceSynapse` referenced the deleted `CellPropagateEventArgs`; renamed to
  its replacement `SequencePropagateEventArgs`.
- `NetworkUtils` called `Math.Round`, which the new `ANN.Net.Utils.Math` partial
  class shadows; qualified as `System.Math.Round`.

## Out of scope

Decoder cross-attention, dropout, learned positional embeddings, batching, and
the remaining in-progress compile errors in the LSTM/GRU cells, the loss-function
refactor and `NetworkBuilder`. Until those are resolved the `ANN.Net` project and
therefore `ANN.Net.Tests` do not build; the transformer was verified with a scratch
project compiling only the sources it depends on.
