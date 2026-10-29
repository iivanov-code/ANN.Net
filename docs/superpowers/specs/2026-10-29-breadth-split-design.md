# Breadth split: a run of layers shared by several processes

Date: 2026-10-29

## Goal

Complement the depth split (`2026-10-28-remote-synapse-and-network-split-design.md`)
with a breadth split, so that a run of hidden layers can be spread over several
machines that each hold a slice of every layer and are therefore equally
loaded, and so that both cuts compose: whole layers on one machine, half
layers on two machines, whole layers on a fourth.

## Why breadth as well as depth

The depth split keeps one machine idle while the other computes the same
sample; overlapping samples would make weights stale, which this library's
"update right after each sample" rule forbids. In the breadth split every part
computes every layer of the same sample, and the arithmetic stays exactly plain
SGD. The price is one exchange per layer instead of one per boundary. With
this object graph a synapse costs on the order of 100 ns and a loopback or LAN
message a few hundred microseconds, so layers of a few hundred neurons gain
close to the full factor and layers of a few dozen lose.

## Shape of a part

For a section of layers `m..n` cut into `P` parts, part `p` holds:

- an identity input layer of the full width of layer `m-1` (plus that layer's
  bias neuron, repeated in every part);
- for every section layer `k < n`: its own contiguous slice of the layer, a
  **ghost** identity neuron for every neuron the other parts own in that
  layer, and its own bias neuron;
- its slice of the last layer `n` as activated output neurons.

Every neuron of a layer is connected to every neuron of the layer before it,
local or ghost, with ordinary `ActiveSynapse`s, so the weights sit exactly
where they sit in one network, each in one part. A local neuron of layer
`k < n` carries one weightless `RemoteSynapse` per other part; the slice of a
layer is complete when its last neuron fires and the part hands the slice to
the parts that need it. A ghost's gradient is the sum over its local
consumers; when every ghost of one layer owned by one part is complete the
partial gradient goes back to the owner, where it enters the neuron's
`Outputs.AccumulateError` through its `RemoteSynapse`, next to the local
synapses. Slices follow `start = width * p / P`.

## Pieces

- `NetworkPartition`: the description (`RemoteNetworkFactory.Partition(settings, hidden, firstLayer, lastLayer, parts)`),
  with slice arithmetic. The head in front is `Split(..., firstLayer - 1)`,
  the tail behind is `Split(..., lastLayer)`; the bias of layer `lastLayer`
  belongs to the tail's input layer, as in the depth split.
- `PartNetwork`: the graph of one part, free of threads. Deliveries
  (`DeliverInputs`, `DeliverActivations`, `DeliverOutputGradients`,
  `DeliverGradients`) run the graph on the calling thread; completions are
  callbacks (`ActivationsReady`, `OutputsReady`, `GradientsReady`,
  `InputGradientsReady`) fired by whichever thread delivered the last piece.
  `Synapses` lists every weighted synapse with global `(layer, from, to)`.
- `IPartitionPeer`: the symmetric one-way contract between parts, plus
  `HelloAsync` (introduction) and `SetLearningRateAsync`.
- `PartitionNode`: threads and RPC around a part. Every delivery runs the
  graph and then flushes an outbox, so a message leaves only after the
  delivery that produced it has fully unwound. Each part hosts
  `RpcHost<IPartitionPeer, IPartitionPeer>`; parts with a higher index connect
  to the ones below them and introduce themselves; `ConnectInProcess` wires two
  nodes of one process through the thread pool with the same semantics.
  Failures are reported to part 0.
- `PartitionCoordinator : IRemoteNetworkPart`: runs on part 0 and makes the
  section look like one remote part to the stage before it. `Propagate` hands
  the inputs to every part, waits for every part's slice of the last layer,
  forwards the joined vector to the next stage and returns its answer.
  `Backpropagate` asks the next stage first, hands every part its slice of the
  gradient, waits for every part's input gradient and answers with the sum.
  Waits have a timeout (25 s by default) and fail on any reported failure.
- `RemoteNetworkHost` now serves any `IRemoteNetworkPart`, so a coordinator is
  hosted like a tail; `RemoteNetworkPeer.ConnectAsync` gives the coordinator its
  proxy to the next stage.

## Why messages are one-way and why the outbox

Two parts send each other their slices of the same layer at the same time; a
request that waits for an answer would hold a neuron lock while blocking, so
every delivery is a `void` call and completion follows from the neurons' own
input counting, in any arrival order.

`ActiveSynapse.Backpropagate` now updates its weight before passing the
gradient on (it still passes `delta * old weight`, so the numbers are
unchanged). Together with the outbox this means that when a part learns a
layer or the section is complete, no weight update of the sending part is
still pending on some stack, so the next sample cannot run on half-updated
weights.

## Composition

`SplitNetwork` head (depth) -> `SynapseConnection` -> `PartitionCoordinator`
(breadth, P parts) -> `RemoteNetworkPart` tail (depth). A tail can itself be a
`SplitNetwork` or another coordinator. The four-machine simulation in the tests
is exactly this chain over TCP on the loopback interface with random ports.

## Reproducibility

Slices arrive in any order, so a neuron may add its incoming values in a
different order from one propagation to the next. The outputs of a partitioned
section are therefore equal only up to float rounding (about the seventh
significant digit), not bit for bit. Summing per neuron in a fixed order would
need `Synapses` to keep one slot per synapse; it was left as is.

## Not covered

- No reconnection and no retransmission; a lost part surfaces as a timeout or
  `RemotePartitionException` on the coordinator.
- Target normalisation is not applied behind the head; the flags
  `ShouldActivate` / `ClearValue` are not carried between parts.
- LSTM/GRU, convolutional or straight connected layers cannot be cut in breadth.
- The thread pool must have a few free threads: the coordinator blocks one
  while it waits for the parts.

## Testing

- `PartNetworkTests`: 1, 2 and 3 parts wired directly against one plain network
  of the same section with identical weights: the same synapses exist once,
  outputs match, input gradients and every weight update match, delivery order
  does not matter.
- `PartitionChainTests`: the four-stage chain in one process with thread-pool
  messaging: shape, stable outputs, finite-difference input gradients, every
  weight update on all four stages (2 and 3 parts) against finite differences,
  training lowers the loss and moves weights on every stage.
- `FourMachineSimulationTests`: the same chain over TCP on four loopback
  endpoints: stable outputs, finite-difference gradients across the wire,
  training moves the weights on all four machines, a remote error two hops
  away arrives as `RpcRemoteException` and the chain survives it.
