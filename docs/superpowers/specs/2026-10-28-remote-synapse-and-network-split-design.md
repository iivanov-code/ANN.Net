# Remote synapse and network split over RPC.Library

Date: 2026-10-28

## Goal

Implement `IRemoteSynapse` with the owner's RPC library (sibling repository
`RPC.Library`) and give the neuron graph an easy way to run its first layers
in one process and the remaining layers in another.

## Where the wire goes

A synapse joins an upstream neuron to a downstream neuron and holds the
weight. When the two neurons live in different processes the wire has to cross
somewhere, and the choice decides how much traffic there is:

- one message per synapse per propagation means `|k| x |k+1|` round trips for a
  boundary between layers k and k+1;
- one message per boundary per propagation means one round trip forward and one
  backward, whatever the layer sizes.

The design takes the second option and keeps the object model intact by
placing the cut *inside* the synapse: the sending end is a weightless axon on
the local side, the receiving end is an ordinary input neuron on the remote
side, and the weights sit in the remote part's first layer of `ActiveSynapse`s,
exactly where they sit in a local network.

## Pieces

`ANN.Net.Remote.Abstactions` (no RPC dependency):

- `IRemoteSynapse : ISynapse` with `Id` and `Connection`.
- `ISynapseConnection`: `Attach(synapse)` and `Propagate(RemotePropagationDTO)`.
  One propagation is complete once every attached synapse has delivered; the
  gradients come back through `ISynapse.Backpropagate` on each synapse, called
  by the connection the way a downstream neuron would call it.
- `IRemoteNetworkPart`: the RPC contract served by the process that holds the
  tail. `GetInfo()`, `Propagate(RemotePropagationDTO[]) -> RemotePropagationResult`,
  `Backpropagate(double[]) -> RemoteBackpropagationResult` (loss, errors and one
  `RemoteBackpropagationDTO` per boundary synapse keyed by id), and
  `SetLearningRateAsync(float)`. Values travel as `double`, `Quad` stays local.

`ANN.Net.Remote` (references `ANN.Net`, the abstractions and `RPC.Library`):

- `RemoteSynapse`: sending end. `Propagate` hands the neuron's value to the
  connection with its id; `Backpropagate` passes the remote gradient on to its
  neuron unchanged; `UpdateWeight` does nothing because there is no weight on
  this side.
- `SynapseConnection : ISynapseConnection`: client side of one boundary. It
  collects values by synapse index, sends the full set in one `Propagate` call
  when the last synapse fires (that happens inside the head's own propagation),
  stores the remote outputs, and on `Backpropagate(targets)` sends the targets
  and feeds each returned gradient into the synapse with that id. It works with
  any `IRemoteNetworkPart`: an RPC proxy from `ConnectAsync` or an in-process
  `RemoteNetworkPart`, which is how the tests check the math without sockets.
- `RemoteNetworkPart : IRemoteNetworkPart`: server side, wraps an `INetwork`
  whose input layer stands in for the boundary. `Propagate` feeds the values to
  the network, remembers the synapse ids, answers with the outputs;
  `Backpropagate` trains the network and answers with `InputGradients` mapped
  back to the ids. Calls are serialised with a lock; one part serves one head.
  `Create(INetwork)` reads the shape from a `Network` or a `SplitNetwork`, so a
  tail can itself be split again.
- `RemoteNetworkHost`: `RpcHost<IRemoteNetworkPart, NoService>` around a part.
- `SplitNetwork : INetwork`: the head `Network` plus one `RemoteSynapse` per
  output neuron. `Propagate` runs the head and returns the remote outputs
  (and throws if the boundary did not reach the remote part); `Backpropagate`
  goes through the connection; `InputGradients` come from the head;
  `SetLearningRate` sets both sides; `Save` is not supported.
- `RemoteNetworkFactory.Split(settings, hiddenLayers, boundaryLayer)` cuts one
  description in two (`NetworkSplit`): the head keeps the input layer, the
  hidden layers before the boundary and the boundary layer built as its output
  layer (same activation, no soft-max, so `Values` are the activations); the
  tail gets an identity input layer of the boundary's size carrying the
  boundary layer's bias neuron, the remaining hidden layers and the real
  output layer with the real loss. `BuildHead` and `BuildTail` build each half.
  Both processes describe the whole network the same way and name the same
  boundary; the synapse ids match because `Propagate` sends them.

`ANN.Net` gained an internal `BuildFFN(settings, additionalInputs, RecurrentLayerSettings[])`
overload and `InternalsVisibleTo ANN.Net.Remote`. `RPC.Library` is referenced as
a project (`..\..\RPC.Library\RPC.Library\RPC.Library.csproj`, overridable with
the `RpcLibraryProject` property) and was added to `ANN.NET.sln`.

## Why the math is unchanged

In one network: `a_i = f(s_i)`, `s_j = sum_i w_ij a_i`, backward
`dL/da_i = sum_j w_ij delta_j`, `delta_i = dL/da_i * f'(s_i)`.
In the split: the head's output neuron i activates `a_i` and its remote synapse
sends it; the tail's identity input neuron i takes the value, its `ActiveSynapse`s
hold `w_ij`; `Network.Backpropagate` on the tail yields `InputGradients[i] = sum_j w_ij delta_j`
(identity derivative is 1); back on the head the remote synapse calls
`OutputNeuron.Backpropagate` with that value as `ErrorSignal`, and `Neuron.ToDelta`
multiplies by `f'(s_i)`. The finite-difference tests confirm it for every
synapse on both sides.

## Not covered

- Target normalisation (`ShouldNormalize` in `Network.Backpropagate`) is not
  applied in the tail; the head still normalises inputs.
- The `ShouldActivate` / `ClearValue` flags are carried on the wire but the
  tail propagates through `INetwork.Propagate`, which has no use for them.
- No reconnection: a dropped connection surfaces as `RpcConnectionException`
  from `Propagate` or `Backpropagate`.
- `LSTMCell` / `GRUCell` layers cannot be the boundary.

## Testing

`ANN.Net.Tests/Remote`:

- `RemoteNetworkFactoryTests`: layer expansion and cut, first/last boundary,
  bad input, tail shape, one remote synapse per boundary neuron.
- `SplitNetworkTests` (in-process part): outputs equal the tail applied to the
  head's values, repeated propagation is stable, loss and errors are the tail's,
  input gradients and every weight update on both sides match central finite
  differences, zero learning rate changes nothing, XOR is learned across the
  boundary.
- `RemoteNetworkPartTests`: shape, propagation, gradient ids, argument checks.
- `RemoteNetworkTcpTests` (TCP loopback through `RemoteNetworkHost` and
  `SynapseConnection.ConnectAsync`): outputs, finite-difference gradients across
  the wire, training moves weights on both sides and cuts the loss, a remote
  argument error arrives as `RpcRemoteException` and the connection survives,
  the host sees the peer come and go.
