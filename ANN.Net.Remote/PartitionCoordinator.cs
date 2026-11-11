using System;
using System.Collections.Generic;
using System.Runtime.ExceptionServices;
using System.Threading.Tasks;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Makes a breadth-split section look like one <see cref="IRemoteNetworkPart"/> to the stage before it.
    /// It runs on the node of part 0: a propagation hands the inputs to every part, waits until every part
    /// has delivered its slice of the last layer, and forwards the joined vector to the next stage; a
    /// backpropagation asks the next stage first, hands every part its slice of the gradient, waits for every
    /// part's input gradient and answers with their sum
    /// </summary>
    public sealed class PartitionCoordinator : IRemoteNetworkPart, IDisposable
    {
        public static readonly TimeSpan DEFAULT_TIMEOUT = TimeSpan.FromSeconds(25);

        private readonly PartitionNode node;
        private readonly IRemoteNetworkPart next;
        private readonly IDisposable owned;
        private readonly TimeSpan timeout;
        private readonly object padlock;
        private readonly Quad[][] outputSlices;
        private readonly Quad[][] gradientSlices;
        private TaskCompletionSource<bool> outputsReady;
        private TaskCompletionSource<bool> gradientsReady;
        private int outputsPending;
        private int gradientsPending;
        private string[] lastIds;
        private RemoteNetworkInfo nextInfo;

        public PartitionCoordinator(PartitionNode node, IRemoteNetworkPart next, TimeSpan? timeout = null)
            : this(node, next, null, timeout)
        { }

        /// <summary>
        /// <paramref name="owned"/> is closed with the coordinator, for example the peer that connects to the next stage
        /// </summary>
        public PartitionCoordinator(PartitionNode node, IRemoteNetworkPart next, IDisposable owned, TimeSpan? timeout = null)
        {
            this.node = node ?? throw new ArgumentNullException(nameof(node));
            this.next = next ?? throw new ArgumentNullException(nameof(next));
            if (!node.IsCoordinator) throw new ArgumentException("The coordinator runs on part 0", nameof(node));

            this.owned = owned;
            this.timeout = timeout ?? DEFAULT_TIMEOUT;
            this.padlock = new object();
            this.outputSlices = new Quad[node.Parts][];
            this.gradientSlices = new Quad[node.Parts][];

            node.OutputsArrived += this.OnOutputsArrived;
            node.InputGradientsArrived += this.OnInputGradientsArrived;
            node.Failed += this.OnFailed;
        }

        public PartitionNode Node => this.node;

        public IRemoteNetworkPart Next => this.next;

        public NetworkPartition Partition => this.node.Network.Partition;

        public RemoteNetworkInfo GetInfo()
        {
            return new RemoteNetworkInfo
            {
                InputCount = this.Partition.InputWidth,
                OutputCount = this.NextInfo.OutputCount
            };
        }

        public RemotePropagationResult Propagate(RemotePropagationDTO[] signals)
        {
            if (signals == null) throw new ArgumentNullException(nameof(signals));
            if (signals.Length != this.Partition.InputWidth)
                throw new ArgumentException($"Expected {this.Partition.InputWidth} signals but received {signals.Length}", nameof(signals));
            if (!this.node.IsConnected) throw new InvalidOperationException("Not every part is connected");

            Quad[] values = new Quad[signals.Length];
            double[] doubles = new double[signals.Length];
            string[] ids = new string[signals.Length];
            for (int i = 0; i < signals.Length; i++)
            {
                RemotePropagationDTO signal = signals[i];
                if (signal == null || signal.Values == null || signal.Values.Length != 1)
                    throw new ArgumentException($"Signal {i} must carry exactly one value", nameof(signals));

                doubles[i] = signal.Values[0];
                values[i] = (Quad)doubles[i];
                ids[i] = signal.Id;
            }

            lock (this.padlock)
            {
                TaskCompletionSource<bool> ready = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                this.outputsReady = ready;
                this.outputsPending = this.node.Parts;
                Array.Clear(this.outputSlices, 0, this.outputSlices.Length);

                for (int part = 1; part < this.node.Parts; part++)
                {
                    this.node.Peer(part).DeliverInputs(doubles);
                }

                this.node.DeliverInputsLocal(values);
                this.Wait(ready.Task, "propagation of the partitioned section");

                Quad[] boundary = Join(this.outputSlices, this.Partition.OutputWidth);
                RemotePropagationDTO[] forward = new RemotePropagationDTO[boundary.Length];
                for (int i = 0; i < boundary.Length; i++)
                {
                    forward[i] = new RemotePropagationDTO { Id = BoundaryId(i), ShouldActivate = true, Values = new[] { boundary[i].ToDouble(null) } };
                }

                RemotePropagationResult result = this.next.Propagate(forward);
                this.lastIds = ids;
                return result;
            }
        }

        public RemoteBackpropagationResult Backpropagate(double[] targets)
        {
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            lock (this.padlock)
            {
                if (this.lastIds == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

                RemoteBackpropagationResult downstream = this.next.Backpropagate(targets);
                if (downstream == null || downstream.Gradients == null) throw new InvalidOperationException("The next stage answered without gradients");

                int width = this.Partition.OutputWidth;
                if (downstream.Gradients.Length != width)
                    throw new InvalidOperationException($"The next stage returned {downstream.Gradients.Length} gradients for {width} boundary neurons");

                Quad[] boundaryGradient = new Quad[width];
                foreach (RemoteBackpropagationDTO gradient in downstream.Gradients)
                {
                    boundaryGradient[BoundaryIndex(gradient.Id, width)] = (Quad)gradient.ErrorSignal;
                }

                TaskCompletionSource<bool> ready = new TaskCompletionSource<bool>(TaskCreationOptions.RunContinuationsAsynchronously);
                this.gradientsReady = ready;
                this.gradientsPending = this.node.Parts;
                Array.Clear(this.gradientSlices, 0, this.gradientSlices.Length);

                int last = this.Partition.LayerCount - 1;
                for (int part = 1; part < this.node.Parts; part++)
                {
                    this.node.Peer(part).DeliverOutputGradients(ToDoubles(Slice(boundaryGradient, this.Partition.SliceStart(last, part), this.Partition.SliceCount(last, part))));
                }

                this.node.DeliverOutputGradientsLocal(Slice(boundaryGradient, this.Partition.SliceStart(last, 0), this.Partition.SliceCount(last, 0)));
                this.Wait(ready.Task, "backpropagation of the partitioned section");

                Quad[] inputGradient = new Quad[this.Partition.InputWidth];
                foreach (Quad[] partial in this.gradientSlices)
                {
                    for (int i = 0; i < inputGradient.Length; i++)
                    {
                        inputGradient[i] += partial[i];
                    }
                }

                RemoteBackpropagationDTO[] gradients = new RemoteBackpropagationDTO[inputGradient.Length];
                for (int i = 0; i < gradients.Length; i++)
                {
                    double value = inputGradient[i].ToDouble(null);
                    gradients[i] = new RemoteBackpropagationDTO { Id = this.lastIds[i], ErrorSignal = value, ErrorWeightedSignal = (float)value, ShouldActivate = true };
                }

                return new RemoteBackpropagationResult { Loss = downstream.Loss, Errors = downstream.Errors, Gradients = gradients };
            }
        }

        public async Task SetLearningRateAsync(float learningRate)
        {
            this.node.SetLearningRate(learningRate);

            for (int part = 1; part < this.node.Parts; part++)
            {
                await this.node.Peer(part).SetLearningRateAsync(learningRate).ConfigureAwait(false);
            }

            await this.next.SetLearningRateAsync(learningRate).ConfigureAwait(false);
        }

        public void Dispose()
        {
            this.node.OutputsArrived -= this.OnOutputsArrived;
            this.node.InputGradientsArrived -= this.OnInputGradientsArrived;
            this.node.Failed -= this.OnFailed;
            this.owned?.Dispose();
        }

        private RemoteNetworkInfo NextInfo => this.nextInfo ?? (this.nextInfo = this.next.GetInfo());

        private void OnOutputsArrived(int part, Quad[] slice)
        {
            TaskCompletionSource<bool> ready = this.outputsReady;
            if (ready == null) return;

            if (this.outputSlices[part] != null)
            {
                ready.TrySetException(new RemotePartitionException(part, "delivered its outputs twice in one propagation"));
                return;
            }

            this.outputSlices[part] = slice;
            if (System.Threading.Interlocked.Decrement(ref this.outputsPending) == 0)
            {
                ready.TrySetResult(true);
            }
        }

        private void OnInputGradientsArrived(int part, Quad[] values)
        {
            TaskCompletionSource<bool> ready = this.gradientsReady;
            if (ready == null) return;

            if (this.gradientSlices[part] != null)
            {
                ready.TrySetException(new RemotePartitionException(part, "delivered its input gradients twice in one backpropagation"));
                return;
            }

            this.gradientSlices[part] = values;
            if (System.Threading.Interlocked.Decrement(ref this.gradientsPending) == 0)
            {
                ready.TrySetResult(true);
            }
        }

        private void OnFailed(Exception error)
        {
            this.outputsReady?.TrySetException(error);
            this.gradientsReady?.TrySetException(error);
        }

        private void Wait(Task task, string what)
        {
            bool completed;
            try
            {
                completed = task.Wait(this.timeout);
            }
            catch (AggregateException ex) when (ex.InnerException != null)
            {
                ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
                return;
            }

            if (!completed) throw new TimeoutException($"The {what} did not complete within {this.timeout}");
        }

        private static string BoundaryId(int index)
        {
            return "b" + index.ToString(System.Globalization.CultureInfo.InvariantCulture);
        }

        private static int BoundaryIndex(string id, int width)
        {
            if (id == null || id.Length < 2 || id[0] != 'b' || !int.TryParse(id.Substring(1), System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out int index) || index >= width)
                throw new InvalidOperationException($"The next stage returned a gradient for unknown boundary neuron {id}");

            return index;
        }

        private static Quad[] Join(IReadOnlyList<Quad[]> slices, int width)
        {
            Quad[] joined = new Quad[width];
            int offset = 0;
            foreach (Quad[] slice in slices)
            {
                Array.Copy(slice, 0, joined, offset, slice.Length);
                offset += slice.Length;
            }

            if (offset != width) throw new InvalidOperationException($"The parts delivered {offset} outputs for {width} boundary neurons");

            return joined;
        }

        private static Quad[] Slice(Quad[] values, int start, int count)
        {
            Quad[] slice = new Quad[count];
            Array.Copy(values, start, slice, 0, count);
            return slice;
        }

        private static double[] ToDoubles(Quad[] values)
        {
            double[] doubles = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                doubles[i] = values[i].ToDouble(null);
            }

            return doubles;
        }
    }
}
