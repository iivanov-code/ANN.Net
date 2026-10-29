using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Neurons;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Server side of a network boundary: serves an ordinary <see cref="INetwork"/> whose input layer stands in
    /// for the boundary neurons of the other process. Its input synapses are the weights that cross the wire,
    /// so <see cref="INetwork.InputGradients"/> is exactly what the remote synapses need back.
    /// Calls are serialised, one network part belongs to one caller
    /// </summary>
    public sealed class RemoteNetworkPart : IRemoteNetworkPart
    {
        private readonly INetwork network;
        private readonly Action<float> setLearningRate;
        private readonly object padlock;
        private string[] lastIds;

        public RemoteNetworkPart(INetwork network, int inputCount, int outputCount, Action<float> setLearningRate = null)
        {
            if (inputCount < 1) throw new ArgumentOutOfRangeException(nameof(inputCount), "Must be greater than 0");
            if (outputCount < 1) throw new ArgumentOutOfRangeException(nameof(outputCount), "Must be greater than 0");

            this.network = network ?? throw new ArgumentNullException(nameof(network));
            this.InputCount = inputCount;
            this.OutputCount = outputCount;
            this.setLearningRate = setLearningRate;
            this.padlock = new object();
        }

        public INetwork LocalNetwork => this.network;

        public int InputCount { get; }

        public int OutputCount { get; }

        /// <summary>
        /// Wraps a network built by <see cref="NetworkFactory"/> or <see cref="RemoteNetworkFactory"/>, reading its shape and optimizers from it
        /// </summary>
        public static RemoteNetworkPart Create(INetwork network)
        {
            switch (network)
            {
                case null:
                    throw new ArgumentNullException(nameof(network));

                case Network local:
                    int inputs = local.InputNeurons.Count(n => !(n is BiasNeuron));
                    return new RemoteNetworkPart(local, inputs, local.OutputNeurons.Count, rate => SetRate(local.Optimizers, rate));

                case SplitNetwork split:
                    return new RemoteNetworkPart(split, split.InputCount, split.OutputCount, split.SetLearningRate);

                default:
                    throw new NotSupportedException($"Cannot read the shape of a {network.GetType().Name}; use the constructor that takes the input and output counts");
            }
        }

        public RemoteNetworkInfo GetInfo()
        {
            return new RemoteNetworkInfo { InputCount = this.InputCount, OutputCount = this.OutputCount };
        }

        public RemotePropagationResult Propagate(RemotePropagationDTO[] signals)
        {
            if (signals == null) throw new ArgumentNullException(nameof(signals));
            if (signals.Length != this.InputCount)
                throw new ArgumentException($"Expected {this.InputCount} signals but received {signals.Length}", nameof(signals));

            Quad[] values = new Quad[signals.Length];
            string[] ids = new string[signals.Length];
            for (int i = 0; i < signals.Length; i++)
            {
                RemotePropagationDTO signal = signals[i];
                if (signal == null || signal.Values == null || signal.Values.Length != 1)
                    throw new ArgumentException($"Signal {i} must carry exactly one value", nameof(signals));

                values[i] = (Quad)signal.Values[0];
                ids[i] = signal.Id;
            }

            lock (this.padlock)
            {
                Quad[] outputs = this.network.Propagate(values);
                this.lastIds = ids;

                return new RemotePropagationResult { Values = ToDoubles(outputs) };
            }
        }

        public RemoteBackpropagationResult Backpropagate(double[] targets)
        {
            if (targets == null) throw new ArgumentNullException(nameof(targets));
            if (targets.Length != this.OutputCount)
                throw new ArgumentException($"Expected {this.OutputCount} targets but received {targets.Length}", nameof(targets));

            Quad[] localTargets = new Quad[targets.Length];
            for (int i = 0; i < targets.Length; i++)
            {
                localTargets[i] = (Quad)targets[i];
            }

            lock (this.padlock)
            {
                if (this.lastIds == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

                Quad loss = this.network.Backpropagate(localTargets);
                Quad[] gradients = this.network.InputGradients;
                if (gradients.Length != this.lastIds.Length)
                    throw new InvalidOperationException($"The network produced {gradients.Length} input gradients for {this.lastIds.Length} signals");

                RemoteBackpropagationDTO[] remoteGradients = new RemoteBackpropagationDTO[gradients.Length];
                for (int i = 0; i < gradients.Length; i++)
                {
                    double gradient = gradients[i].ToDouble(null);
                    remoteGradients[i] = new RemoteBackpropagationDTO
                    {
                        Id = this.lastIds[i],
                        ErrorSignal = gradient,
                        ErrorWeightedSignal = (float)gradient,
                        ShouldActivate = true
                    };
                }

                return new RemoteBackpropagationResult
                {
                    Loss = loss.ToDouble(null),
                    Errors = ToDoubles(this.network.Errors),
                    Gradients = remoteGradients
                };
            }
        }

        public Task SetLearningRateAsync(float learningRate)
        {
            if (this.setLearningRate == null)
                throw new NotSupportedException("This network part does not expose its optimizers");

            lock (this.padlock)
            {
                this.setLearningRate(learningRate);
            }

            return Task.CompletedTask;
        }

        internal static void SetRate(IEnumerable<LearningRateOptimizer> optimizers, float learningRate)
        {
            foreach (LearningRateOptimizer optimizer in optimizers)
            {
                optimizer.LearningRate = learningRate;
            }
        }

        private static double[] ToDoubles(Quad[] values)
        {
            if (values == null) return Array.Empty<double>();

            double[] doubles = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                doubles[i] = values[i].ToDouble(null);
            }

            return doubles;
        }
    }
}
