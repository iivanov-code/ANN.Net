using System;
using System.Net;
using System.Threading;
using System.Threading.Tasks;
using ANN.Net.Remote.Abstactions.Interfaces;
using RPC.Library;
using RPC.Library.Abstractions;

namespace ANN.Net.Remote
{
    /// <summary>
    /// Connects to a <see cref="RemoteNetworkHost"/>; the peer's <c>Remote</c> is the served network part
    /// </summary>
    public static class RemoteNetworkPeer
    {
        public static Task<RpcPeer<NoService, IRemoteNetworkPart>> ConnectAsync(IRpcTransport transport,
                                                                               EndPoint endPoint,
                                                                               RpcOptions options = null,
                                                                               CancellationToken cancellationToken = default)
        {
            if (transport == null) throw new ArgumentNullException(nameof(transport));
            if (endPoint == null) throw new ArgumentNullException(nameof(endPoint));

            return RpcPeer.ConnectAsync<NoService, IRemoteNetworkPart>(transport, endPoint, null, options, cancellationToken);
        }

        public static Task<RpcPeer<NoService, IRemoteNetworkPart>> ConnectAsync(IRpcTransport transport,
                                                                               string host,
                                                                               int port,
                                                                               RpcOptions options = null,
                                                                               CancellationToken cancellationToken = default)
        {
            if (transport == null) throw new ArgumentNullException(nameof(transport));

            return RpcPeer.ConnectAsync<NoService, IRemoteNetworkPart>(transport, host, port, null, options, cancellationToken);
        }
    }
}
