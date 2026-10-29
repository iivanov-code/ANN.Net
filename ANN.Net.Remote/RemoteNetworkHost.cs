using System;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Remote.Abstactions.Interfaces;
using RPC.Library;
using RPC.Library.Abstractions;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Serves one <see cref="RemoteNetworkPart"/> over an RPC transport, so a <see cref="SplitNetwork"/> in
    /// another process can connect its remote synapses to it
    /// </summary>
    public sealed class RemoteNetworkHost : IDisposable
    {
        private readonly RpcHost<IRemoteNetworkPart, NoService> host;

        private RemoteNetworkHost(IRemoteNetworkPart part, RpcHost<IRemoteNetworkPart, NoService> host)
        {
            this.Part = part;
            this.host = host;
        }

        public IRemoteNetworkPart Part { get; }

        public EndPoint LocalEndPoint => this.host.LocalEndPoint;

        public int ConnectedPeers => this.host.Peers.Count;

        /// <summary>
        /// Serves a <see cref="RemoteNetworkPart"/> or a <see cref="PartitionCoordinator"/>
        /// </summary>
        public static async Task<RemoteNetworkHost> StartAsync(IRemoteNetworkPart part,
                                                               IRpcTransport transport,
                                                               EndPoint endPoint,
                                                               RpcOptions options = null,
                                                               CancellationToken cancellationToken = default)
        {
            if (part == null) throw new ArgumentNullException(nameof(part));
            if (transport == null) throw new ArgumentNullException(nameof(transport));
            if (endPoint == null) throw new ArgumentNullException(nameof(endPoint));

            RpcHost<IRemoteNetworkPart, NoService> host = new RpcHost<IRemoteNetworkPart, NoService>(transport, part, options);
            try
            {
                await host.StartAsync(endPoint, cancellationToken).ConfigureAwait(false);
            }
            catch
            {
                host.Dispose();
                throw;
            }

            return new RemoteNetworkHost(part, host);
        }

        /// <summary>
        /// Serves a network built by <see cref="NetworkFactory"/> or <see cref="RemoteNetworkFactory.BuildTail(Abstractions.Settings.NetworkSettings, Abstractions.Settings.RecurrentLayerSettings[], int)"/>
        /// </summary>
        public static Task<RemoteNetworkHost> StartAsync(INetwork network,
                                                         IRpcTransport transport,
                                                         EndPoint endPoint,
                                                         RpcOptions options = null,
                                                         CancellationToken cancellationToken = default)
        {
            return StartAsync(RemoteNetworkPart.Create(network), transport, endPoint, options, cancellationToken);
        }

        public static Task<RemoteNetworkHost> StartAsync(INetwork network,
                                                         IRpcTransport transport,
                                                         int port,
                                                         RpcOptions options = null,
                                                         CancellationToken cancellationToken = default)
        {
            return StartAsync(network, transport, new IPEndPoint(IPAddress.Any, port), options, cancellationToken);
        }

        public Task StopAsync()
        {
            return this.host.StopAsync();
        }

        public void Dispose()
        {
            this.host.Dispose();
        }
    }
}
