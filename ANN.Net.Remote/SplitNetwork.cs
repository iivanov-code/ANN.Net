using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Neurons;

namespace ANN.Net.Remote
{
    /// <summary>
    /// A network whose first layers run here and whose remaining layers run in another process.
    /// The local part is an ordinary neuron graph whose last layer is the boundary; every boundary neuron
    /// owns one <see cref="RemoteSynapse"/>. Propagating fills the boundary, which sends one message and
    /// receives the remote outputs; backpropagating sends the targets and receives the boundary gradients,
    /// which then flow back through the local synapses. To the caller it behaves like one <see cref="INetwork"/>
    /// </summary>
    public sealed class SplitNetwork : INetwork, IDisposable
    {
        private readonly Network head;
        private readonly SynapseConnection connection;
        private readonly List<RemoteSynapse> synapses;

        internal SplitNetwork(Network head, SynapseConnection connection, IUniqueIdentityGenerator identityGenerator)
        {
            this.head = head ?? throw new ArgumentNullException(nameof(head));
            this.connection = connection ?? throw new ArgumentNullException(nameof(connection));
            if (identityGenerator == null) throw new ArgumentNullException(nameof(identityGenerator));

            this.synapses = new List<RemoteSynapse>(head.OutputNeurons.Count);
            foreach (IOutputNeuron neuron in head.OutputNeurons)
            {
                RemoteSynapse synapse = new RemoteSynapse(neuron, connection, identityGenerator);
                neuron.Outputs.Add(synapse);
                this.synapses.Add(synapse);
            }

            this.InputCount = head.InputNeurons.Count(n => !(n is BiasNeuron));
        }

        /// <summary>
        /// The layers that run in this process
        /// </summary>
        public INetwork Head => this.head;

        public SynapseConnection Connection => this.connection;

        public IReadOnlyList<RemoteSynapse> RemoteSynapses => this.synapses;

        /// <summary>
        /// Optimizers of the local synapses; the remote ones are reached through <see cref="SetLearningRate"/>
        /// </summary>
        public ICollection<LearningRateOptimizer> Optimizers => this.head.Optimizers;

        public int InputCount { get; }

        public int BoundaryCount => this.synapses.Count;

        public int OutputCount => this.connection.Info.OutputCount;

        /// <summary>
        /// Outputs of the remote part from the last propagation
        /// </summary>
        public Quad[] Values => this.connection.Values ?? Array.Empty<Quad>();

        public Quad[] Errors => this.connection.Errors ?? Array.Empty<Quad>();

        public Quad[] InputGradients => this.head.InputGradients;

        public IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues => this.head.MinMaxValues;

        public Quad[] Propagate(params Quad[] values)
        {
            long before = this.connection.PropagationCount;

            this.head.Propagate(values);

            if (this.connection.PropagationCount == before)
            {
                throw new InvalidOperationException(this.connection.HasPendingSignals
                    ? "Not every boundary neuron delivered its value, so nothing was sent to the remote part"
                    : "The boundary did not fire; the remote part was not reached");
            }

            return this.connection.Values;
        }

        public Quad Backpropagate(params Quad[] targets)
        {
            return this.connection.Backpropagate(targets);
        }

        /// <summary>
        /// Sets the learning rate of every optimizer, local and remote
        /// </summary>
        public void SetLearningRate(float learningRate)
        {
            RemoteNetworkPart.SetRate(this.head.Optimizers, learningRate);
            this.connection.SetLearningRate(learningRate);
        }

        public void Save(string fullDirPath, string fileName)
        {
            throw new NotSupportedException("A split network lives in two processes; save each part where it runs");
        }

        public void Dispose()
        {
            this.connection.Dispose();
        }
    }
}
