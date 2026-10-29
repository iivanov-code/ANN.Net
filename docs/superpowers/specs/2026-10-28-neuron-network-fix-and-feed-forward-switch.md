# Neuron network fix and switchable transformer feed forward

Date: 2026-10-28

## Goal

Make the transformer's position-wise feed forward sub-layer switchable
between the matrix implementation (`FeedForwardNetwork`) and the classic
neuron graph (`Network`, neuron objects connected by synapses), without
removing either. The neuron graph could not do the job as it stood, so its
propagation and backpropagation were repaired first.

## Defects found in the neuron graph (verified with a probe before the fix)

1. **Synapse weights compounded.** One `NeuronPropagateEventArgs` instance was
   handed to every outgoing synapse and `ActiveSynapse.Propagate` multiplied it
   in place, so the k-th synapse carried the product of the first k weights.
2. **Sums were never reset.** `Synapses.Value` was only cleared inside
   `FeedbackError`, so propagating twice doubled the output and inference
   without backpropagation was impossible.
3. **Backpropagation crashed.** Input neurons have no `Inputs` collection and
   `Neuron.FeedbackError` dereferenced it.
4. **The gradient math was wrong.** The activation derivative was evaluated on
   the accumulated error, and weights were updated with the running error sum
   instead of the input neuron's activation. Output neurons also had an input
   sum of zero, so every delta was zero.
5. **Bias neurons never fired**, and a bias neuron in a hidden layer stalled
   the next layer because its synapse never delivered.
6. `TanHActivation` inherited `BaseActivation<HardSigmoidActivation>`, so the
   hyperbolic tangent activation was in fact a sigmoid. Sigmoid and tanh took
   the activated value in `Derivative` while every other activation took the
   pre-activation.
7. `NetworkUtils.XavierWeight` produced `1 / U(1, n)`, positive weights in
   `(1/n, 1]`, not a Xavier distribution.

## Decisions

- **Message per synapse.** `ActiveSynapse.Propagate` sends a new weighted
  message; the neuron's own message is never mutated.
- **Neurons remember their own state.** `Neuron` keeps `PreActivation`,
  `Value` (after activation) and `Delta` (gradient at the pre-activation).
  `Synapses.AccountSignal` and `AccumulateError` hand the completed sum to the
  neuron and reset to zero immediately, so the next propagation starts clean.
- **Textbook gradients.** `delta = upstream * f'(pre-activation)`; each synapse
  passes `delta * weight` to its input neuron and updates its own weight with
  `delta * input.Value`. Input neurons store `delta` as `Gradient`, exposed
  through `IInputNeuron.Gradient` and `INetwork.InputGradients`.
- **Output neurons accumulate on `Inputs`** like hidden neurons and honour the
  `shouldActivate` flag; `INeuron.Value` replaces the `IOutputNeuron.Value`
  special case.
- **Bias neurons output a constant 1**, have no inputs, and are fired by
  `Network.Propagate` for the input layer and every hidden layer.
- **Derivative convention.** `IActivationFunction.Derivative(x)` takes the
  value before activation. Sigmoid and tanh were changed to match; the
  transformer's `FeedForwardNetwork` already used that convention.
- **`ActivationTypes.Identity`** was added so inputs and linear outputs pass
  through unchanged.
- **Xavier initialization** in `NetworkBuilder.ConnectAxon` now uses the Glorot
  uniform formula (`NetworkUtils.GlorotWeight`), symmetric around zero.
- **Switch.** `TransformerSettings.FeedForwardType` selects `Matrix` (default)
  or `NeuronGraph`. Both implement the internal `IFeedForwardNetwork`
  (`Propagate(Quad[][])`, `Backpropagate(Quad[][])` returning input gradients).
  `NeuronFeedForwardNetwork` builds `identity input (+bias) -> activated hidden
  (+bias) -> identity output` with `NetworkFactory.BuildFFN`, applies it per
  position, and during backpropagation re-propagates each position, then calls
  `Network.Backpropagate` with `target = value - gradient` so the fed-back
  error equals the upstream gradient. Its optimizers join the transformer's
  `Optimizers` collection. Weights update after every position.

## Concurrency fixes found on the way

- `BaseActivation<T>.Instance` and `BaseLossFunction<T>.Instance` used an
  unsynchronized check-then-add on a static dictionary. Two networks built at
  the same time on different threads threw `ArgumentException: An item with
  the same key has already been added`. Both are now `Lazy<T>` singletons.
- `NetworkUtils.Rand` was a hand-made `System.Random` shared by every thread;
  concurrent use can corrupt it so that it only returns zeros. It is now
  `Random.Shared`.

## Not changed

`LSTMCell`, `GRUCell`, the sequence synapses, `MultiDimentionalNeuron`
semantics (only adapted to the new neuron state), recurrent connections,
`Network.Values` soft-max normalization, `Cell.BuildGate` (still passes no
identity generator), and `ANNetwork.Save` (`BinaryFormatter`).

## Testing

- `ANN.Net.Tests/Networks/NetworkTests.cs`: repeated propagation is stable,
  outputs match a manual computation with and without bias neurons, input
  gradients and every synapse's weight update match central finite
  differences, zero learning rate leaves weights alone, XOR is learned,
  tanh/sigmoid/identity behave as named.
- `GradientCheckTests`: `NeuronFeedForwardNetwork` and the encoder block with
  either feed forward type match finite differences.
- `TransformerNetworkTests`: next-token learning passes with both feed forward
  types; the neuron graph variant gives the same output on repeated
  propagation.
