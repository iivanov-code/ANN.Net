using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using ANN.Net.Remote.Abstactions.Interfaces;
using RPC.Library;
using RPC.Library.Abstractions;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Runs one <see cref="PartNetwork"/> and talks to the other parts. Every delivery runs the graph on the
    /// calling thread; whatever the graph wants to send is queued and sent once that delivery has fully
    /// unwound, so no weight update is still pending when another part learns that a layer is complete.
    /// Each part hosts <see cref="IPartitionPeer"/>; parts with a higher index connect to the ones below them,
    /// or <see cref="ConnectInProcess"/> wires two nodes of the same process through the thread pool
    /// </summary>
    public sealed class PartitionNode : IDisposable
    {
        private readonly IPartitionPeer[] peers;
        private readonly ConcurrentQueue<Action> outbox;
        private readonly List<IDisposable> connections;
        private RpcHost<IPartitionPeer, IPartitionPeer> host;

        public PartitionNode(NetworkPartition partition, int part)
            : this(new PartNetwork(partition, part))
        { }

        public PartitionNode(PartNetwork network)
        {
            this.Network = network ?? throw new ArgumentNullException(nameof(network));
            this.peers = new IPartitionPeer[network.Partition.Parts];
            this.outbox = new ConcurrentQueue<Action>();
            this.connections = new List<IDisposable>();

            network.ActivationsReady = (toPart, layer, values) => this.Enqueue(toPart, peer => peer.DeliverActivations(this.Part, layer, ToDoubles(values)));
            network.GradientsReady = (toPart, layer, values) => this.Enqueue(toPart, peer => peer.DeliverGradients(this.Part, layer, ToDoubles(values)));
            network.OutputsReady = values => this.EnqueueToCoordinator(values, (peer, doubles) => peer.DeliverOutputs(this.Part, doubles), this.OutputsArrived);
            network.InputGradientsReady = values => this.EnqueueToCoordinator(values, (peer, doubles) => peer.DeliverInputGradients(this.Part, doubles), this.InputGradientsArrived);
        }

        /// <summary>
        /// On the coordinator: a part's slice of the last layer, including the coordinator's own
        /// </summary>
        public event Action<int, Quad[]> OutputsArrived;

        /// <summary>
        /// On the coordinator: a part's partial gradient with respect to the section inputs, including the coordinator's own
        /// </summary>
        public event Action<int, Quad[]> InputGradientsArrived;

        /// <summary>
        /// A delivery threw here or in another part
        /// </summary>
        public event Action<Exception> Failed;

        public PartNetwork Network { get; }

        public int Part => this.Network.Part;

        public int Parts => this.Network.Partition.Parts;

        public bool IsCoordinator => this.Part == 0;

        public EndPoint LocalEndPoint => this.host?.LocalEndPoint;

        /// <summary>
        /// True once every other part can be reached
        /// </summary>
        public bool IsConnected => Enumerable.Range(0, this.Parts).All(p => p == this.Part || this.peers[p] != null);

        public IPartitionPeer Peer(int part)
        {
            if (part < 0 || part >= this.Parts) throw new ArgumentOutOfRangeException(nameof(part));
            if (part == this.Part) throw new ArgumentException("A part does not message itself", nameof(part));

            return this.peers[part] ?? throw new RemotePartitionException(part, "not connected");
        }

        /// <summary>
        /// Accepts connections from the parts with a higher index
        /// </summary>
        public async Task StartAsync(IRpcTransport transport, EndPoint endPoint, RpcOptions options = null, CancellationToken cancellationToken = default)
        {
            if (transport == null) throw new ArgumentNullException(nameof(transport));
            if (endPoint == null) throw new ArgumentNullException(nameof(endPoint));
            if (this.host != null) throw new InvalidOperationException("Node already started");

            RpcHost<IPartitionPeer, IPartitionPeer> rpcHost = new RpcHost<IPartitionPeer, IPartitionPeer>(transport, peer => new PartitionPeerService(this, peer.Remote), options);
            rpcHost.AcceptFailed += (sender, error) => this.Failed?.Invoke(error);

            await rpcHost.StartAsync(endPoint, cancellationToken).ConfigureAwait(false);
            this.host = rpcHost;
        }

        /// <summary>
        /// Connects to the node of <paramref name="remotePart"/>, which must have started, and introduces this part to it
        /// </summary>
        public async Task ConnectAsync(int remotePart, IRpcTransport transport, EndPoint endPoint, RpcOptions options = null, CancellationToken cancellationToken = default)
        {
            if (transport == null) throw new ArgumentNullException(nameof(transport));
            if (endPoint == null) throw new ArgumentNullException(nameof(endPoint));
            if (remotePart == this.Part) throw new ArgumentException("A part does not connect to itself", nameof(remotePart));

            RpcPeer<IPartitionPeer, IPartitionPeer> peer = await RpcPeer.ConnectAsync<IPartitionPeer, IPartitionPeer>(transport, endPoint, new PartitionPeerService(this, null), options, cancellationToken)
                                                                        .ConfigureAwait(false);
            peer.UnhandledError += (sender, error) => this.Failed?.Invoke(error);
            this.connections.Add(peer);

            this.RegisterPeer(remotePart, peer.Remote);
            await peer.Remote.HelloAsync(this.Part).ConfigureAwait(false);
        }

        public Task ConnectAsync(int remotePart, IRpcTransport transport, string host, int port, RpcOptions options = null, CancellationToken cancellationToken = default)
        {
            EndPoint endPoint = IPAddress.TryParse(host, out IPAddress address)
                ? new IPEndPoint(address, port)
                : (EndPoint)new DnsEndPoint(host, port);

            return this.ConnectAsync(remotePart, transport, endPoint, options, cancellationToken);
        }

        /// <summary>
        /// Wires two nodes of the same process; deliveries hop through the thread pool like messages would
        /// </summary>
        public void ConnectInProcess(PartitionNode other)
        {
            if (other == null) throw new ArgumentNullException(nameof(other));
            if (other.Part == this.Part) throw new ArgumentException("A part does not connect to itself", nameof(other));

            this.RegisterPeer(other.Part, new ThreadPoolPeer(other));
            other.RegisterPeer(this.Part, new ThreadPoolPeer(this));
        }

        public void SetLearningRate(float learningRate)
        {
            this.Network.SetLearningRate(learningRate);
        }

        public void Dispose()
        {
            foreach (IDisposable connection in this.connections)
            {
                connection.Dispose();
            }

            this.host?.Dispose();
        }

        internal void RegisterPeer(int part, IPartitionPeer peer)
        {
            if (part < 0 || part >= this.Parts) throw new ArgumentOutOfRangeException(nameof(part));
            if (part == this.Part) throw new ArgumentException("A part does not message itself", nameof(part));

            this.peers[part] = peer ?? throw new ArgumentNullException(nameof(peer));
        }

        // deliveries, in wire form; each one runs the graph and then flushes whatever the graph queued

        internal void DeliverInputs(double[] values)
        {
            this.Run(() => this.Network.DeliverInputs(ToQuads(values)));
        }

        internal void DeliverActivations(int fromPart, int layer, double[] slice)
        {
            this.Run(() => this.Network.DeliverActivations(fromPart, layer, ToQuads(slice)));
        }

        internal void DeliverOutputs(int fromPart, double[] slice)
        {
            this.OutputsArrived?.Invoke(fromPart, ToQuads(slice));
        }

        internal void DeliverOutputGradients(double[] slice)
        {
            this.Run(() => this.Network.DeliverOutputGradients(ToQuads(slice)));
        }

        internal void DeliverGradients(int fromPart, int layer, double[] slice)
        {
            this.Run(() => this.Network.DeliverGradients(fromPart, layer, ToQuads(slice)));
        }

        internal void DeliverInputGradients(int fromPart, double[] values)
        {
            this.InputGradientsArrived?.Invoke(fromPart, ToQuads(values));
        }

        internal void ReportFailure(int fromPart, string message)
        {
            this.Failed?.Invoke(new RemotePartitionException(fromPart, message));
        }

        /// <summary>
        /// Coordinator only: feeds its own part directly
        /// </summary>
        internal void DeliverInputsLocal(Quad[] values)
        {
            this.Run(() => this.Network.DeliverInputs(values));
        }

        internal void DeliverOutputGradientsLocal(Quad[] slice)
        {
            this.Run(() => this.Network.DeliverOutputGradients(slice));
        }

        private void Run(Action deliver)
        {
            try
            {
                deliver();
            }
            catch (Exception ex)
            {
                this.Fail(ex);
            }
            finally
            {
                this.FlushOutbox();
            }
        }

        private void FlushOutbox()
        {
            while (this.outbox.TryDequeue(out Action send))
            {
                try
                {
                    send();
                }
                catch (Exception ex)
                {
                    this.Fail(ex);
                }
            }
        }

        private void Enqueue(int toPart, Action<IPartitionPeer> send)
        {
            this.outbox.Enqueue(() => send(this.Peer(toPart)));
        }

        private void EnqueueToCoordinator(Quad[] values, Action<IPartitionPeer, double[]> send, Action<int, Quad[]> local)
        {
            if (this.IsCoordinator)
            {
                this.outbox.Enqueue(() => local?.Invoke(this.Part, values));
            }
            else
            {
                double[] doubles = ToDoubles(values);
                this.outbox.Enqueue(() => send(this.Peer(0), doubles));
            }
        }

        private void Fail(Exception ex)
        {
            Exception failure = ex is RemotePartitionException ? ex : new RemotePartitionException(this.Part, ex.Message, ex);
            this.Failed?.Invoke(failure);

            if (!this.IsCoordinator && this.peers[0] != null)
            {
                try
                {
                    this.peers[0].ReportFailure(this.Part, ex.Message);
                }
                catch (Exception reportError)
                {
                    this.Failed?.Invoke(reportError);
                }
            }
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

        private static double[] ToDoubles(Quad[] values)
        {
            double[] doubles = new double[values.Length];
            for (int i = 0; i < values.Length; i++)
            {
                doubles[i] = values[i].ToDouble(null);
            }

            return doubles;
        }

        /// <summary>
        /// The service one connection exposes to its remote part; forwards to the node and handles the introduction
        /// </summary>
        private sealed class PartitionPeerService : IPartitionPeer
        {
            private readonly PartitionNode node;
            private readonly IPartitionPeer remote;

            public PartitionPeerService(PartitionNode node, IPartitionPeer remote)
            {
                this.node = node;
                this.remote = remote;
            }

            public Task HelloAsync(int part)
            {
                if (this.remote == null) throw new InvalidOperationException("Only the accepting side takes introductions");

                this.node.RegisterPeer(part, this.remote);
                return Task.CompletedTask;
            }

            public void DeliverInputs(double[] values) => this.node.DeliverInputs(values);

            public void DeliverActivations(int fromPart, int layer, double[] slice) => this.node.DeliverActivations(fromPart, layer, slice);

            public void DeliverOutputs(int fromPart, double[] slice) => this.node.DeliverOutputs(fromPart, slice);

            public void DeliverOutputGradients(double[] slice) => this.node.DeliverOutputGradients(slice);

            public void DeliverGradients(int fromPart, int layer, double[] slice) => this.node.DeliverGradients(fromPart, layer, slice);

            public void DeliverInputGradients(int fromPart, double[] values) => this.node.DeliverInputGradients(fromPart, values);

            public void ReportFailure(int fromPart, string message) => this.node.ReportFailure(fromPart, message);

            public Task SetLearningRateAsync(float learningRate)
            {
                this.node.SetLearningRate(learningRate);
                return Task.CompletedTask;
            }
        }

        /// <summary>
        /// In-process stand-in for a connection: every one-way call hops through the thread pool
        /// </summary>
        private sealed class ThreadPoolPeer : IPartitionPeer
        {
            private readonly PartitionNode target;

            public ThreadPoolPeer(PartitionNode target)
            {
                this.target = target;
            }

            public Task HelloAsync(int part) => Task.CompletedTask;

            public void DeliverInputs(double[] values) => Task.Run(() => this.target.DeliverInputs(values));

            public void DeliverActivations(int fromPart, int layer, double[] slice) => Task.Run(() => this.target.DeliverActivations(fromPart, layer, slice));

            public void DeliverOutputs(int fromPart, double[] slice) => Task.Run(() => this.target.DeliverOutputs(fromPart, slice));

            public void DeliverOutputGradients(double[] slice) => Task.Run(() => this.target.DeliverOutputGradients(slice));

            public void DeliverGradients(int fromPart, int layer, double[] slice) => Task.Run(() => this.target.DeliverGradients(fromPart, layer, slice));

            public void DeliverInputGradients(int fromPart, double[] values) => Task.Run(() => this.target.DeliverInputGradients(fromPart, values));

            public void ReportFailure(int fromPart, string message) => Task.Run(() => this.target.ReportFailure(fromPart, message));

            public Task SetLearningRateAsync(float learningRate)
            {
                this.target.SetLearningRate(learningRate);
                return Task.CompletedTask;
            }
        }
    }
}
