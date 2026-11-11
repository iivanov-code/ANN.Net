using System;
using System.Collections.Generic;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Remote.Abstactions.Models;
using RPC.Library;
using RPC.Library.Abstractions;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Client side of a depth boundary. Collects the value of every attached <see cref="IRemoteSynapse"/>,
    /// sends the complete set to the remote part in one call and keeps the outputs it answers with; on
    /// backpropagation it sends the targets and feeds the returned gradients back into the synapses.
    /// The remote part can be an RPC proxy (see <see cref="ConnectAsync(IRpcTransport, EndPoint, RpcOptions, CancellationToken)"/>),
    /// an in-process <see cref="RemoteNetworkPart"/> or a <see cref="PartitionCoordinator"/>
    /// </summary>
    public sealed class SynapseConnection : ISynapseConnection, IDisposable
    {
        private readonly IRemoteNetworkPart remote;
        private readonly IDisposable owned;
        private readonly SynapseBatch batch;
        private readonly object padlock;
        private RemoteNetworkInfo info;

        public SynapseConnection(IRemoteNetworkPart remote)
            : this(remote, null)
        { }

        private SynapseConnection(IRemoteNetworkPart remote, IDisposable owned)
        {
            this.remote = remote ?? throw new ArgumentNullException(nameof(remote));
            this.owned = owned;
            this.batch = new SynapseBatch();
            this.padlock = new object();
        }

        /// <summary>
        /// Connects to a <see cref="RemoteNetworkHost"/>; the connection owns the peer and closes it on dispose
        /// </summary>
        public static async Task<SynapseConnection> ConnectAsync(IRpcTransport transport,
                                                                 EndPoint endPoint,
                                                                 RpcOptions options = null,
                                                                 CancellationToken cancellationToken = default)
        {
            RpcPeer<NoService, IRemoteNetworkPart> peer = await RemoteNetworkPeer.ConnectAsync(transport, endPoint, options, cancellationToken).ConfigureAwait(false);
            return new SynapseConnection(peer.Remote, peer);
        }

        public static async Task<SynapseConnection> ConnectAsync(IRpcTransport transport,
                                                                 string host,
                                                                 int port,
                                                                 RpcOptions options = null,
                                                                 CancellationToken cancellationToken = default)
        {
            RpcPeer<NoService, IRemoteNetworkPart> peer = await RemoteNetworkPeer.ConnectAsync(transport, host, port, options, cancellationToken).ConfigureAwait(false);
            return new SynapseConnection(peer.Remote, peer);
        }

        public IRemoteNetworkPart Remote => this.remote;

        public IReadOnlyList<IRemoteSynapse> Synapses => this.batch.Synapses;

        /// <summary>
        /// Shape of the remote part, asked once
        /// </summary>
        public RemoteNetworkInfo Info
        {
            get
            {
                lock (this.padlock)
                {
                    return this.info ?? (this.info = this.remote.GetInfo());
                }
            }
        }

        /// <summary>
        /// Outputs of the remote part from the last completed propagation, null before the first one
        /// </summary>
        public Quad[] Values { get; private set; }

        /// <summary>
        /// value - target of every remote output from the last backpropagation
        /// </summary>
        public Quad[] Errors { get; private set; }

        public Quad Loss { get; private set; }

        /// <summary>
        /// Number of completed propagations, so a caller can tell that its propagation really crossed the wire
        /// </summary>
        public long PropagationCount { get; private set; }

        /// <summary>
        /// True while some, but not all, attached synapses have delivered a value
        /// </summary>
        public bool HasPendingSignals => this.batch.HasPendingSignals;

        public void Attach(IRemoteSynapse synapse)
        {
            this.batch.Attach(synapse);
        }

        public void Propagate(RemotePropagationDTO remoteData)
        {
            if (!this.batch.Add(remoteData, out RemotePropagationDTO[] complete)) return;

            lock (this.padlock)
            {
                RemoteNetworkInfo shape = this.info ?? (this.info = this.remote.GetInfo());
                if (shape != null && shape.InputCount != complete.Length)
                    throw new InvalidOperationException($"{complete.Length} remote synapses are attached but the remote part expects {shape.InputCount} inputs");

                RemotePropagationResult result = this.remote.Propagate(complete);
                if (result == null || result.Values == null) throw new InvalidOperationException("The remote part answered without values");

                this.Values = ToQuads(result.Values);
                this.PropagationCount++;
            }
        }

        /// <summary>
        /// Sends the targets to the remote part, stores its loss and errors and feeds the returned gradient
        /// into every attached synapse, which passes it on to its neuron
        /// </summary>
        public Quad Backpropagate(Quad[] targets)
        {
            if (targets == null) throw new ArgumentNullException(nameof(targets));

            lock (this.padlock)
            {
                if (this.Values == null) throw new InvalidOperationException("Propagate must complete before Backpropagate");
                if (this.batch.HasPendingSignals) throw new InvalidOperationException("Cannot backpropagate while a propagation is in progress");

                double[] remoteTargets = new double[targets.Length];
                for (int i = 0; i < targets.Length; i++)
                {
                    remoteTargets[i] = targets[i].ToDouble(null);
                }

                RemoteBackpropagationResult result = this.remote.Backpropagate(remoteTargets);
                if (result == null) throw new InvalidOperationException("The remote part answered without a result");

                this.Loss = (Quad)result.Loss;
                this.Errors = ToQuads(result.Errors);

                RemoteBackpropagationDTO[] gradients = result.Gradients ?? Array.Empty<RemoteBackpropagationDTO>();
                if (gradients.Length != this.batch.Count)
                    throw new InvalidOperationException($"The remote part returned {gradients.Length} gradients for {this.batch.Count} synapses");

                foreach (RemoteBackpropagationDTO gradient in gradients)
                {
                    int index = this.batch.IndexOf(gradient.Id);
                    if (index < 0) throw new InvalidOperationException($"The remote part returned a gradient for unknown synapse {gradient.Id}");

                    this.batch.Synapses[index].Backpropagate(gradient.ToEventArgs());
                }

                return this.Loss;
            }
        }

        public void SetLearningRate(float learningRate)
        {
            this.remote.SetLearningRateAsync(learningRate).GetAwaiter().GetResult();
        }

        public void Dispose()
        {
            this.owned?.Dispose();
        }

        private static Quad[] ToQuads(double[] values)
        {
            if (values == null) return Array.Empty<Quad>();

            Quad[] quads = new Quad[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                quads[i] = (Quad)values[i];
            }

            return quads;
        }
    }
}
