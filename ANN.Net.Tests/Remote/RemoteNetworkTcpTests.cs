using System;
using System.Diagnostics;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.Remote;
using ANN.Net.Tests.Transformers;
using RPC.Library;
using RPC.Library.Transports.Tcp;
using Xunit;

namespace ANN.Net.Tests.Remote
{
    /// <summary>
    /// The tail served by <see cref="RemoteNetworkHost"/> over TCP on the loopback interface, the head connected through RPC
    /// </summary>
    public class RemoteNetworkTcpTests
    {
        private sealed class Link : IDisposable
        {
            public RemoteNetworkHost Host { get; set; }

            public SplitNetwork Split { get; set; }

            public Network Tail { get; set; }

            public void Dispose()
            {
                this.Split?.Dispose();
                this.Host?.Dispose();
            }
        }

        private static async Task<Link> StartAsync(NetworkSettings whole, RecurrentLayerSettings[] layers, int boundary)
        {
            Link link = new Link();
            link.Tail = (Network)RemoteNetworkFactory.BuildTail(whole, layers, boundary);
            link.Host = await RemoteNetworkHost.StartAsync(link.Tail, new TcpTransport(), new IPEndPoint(IPAddress.Loopback, 0));

            SynapseConnection connection = await SynapseConnection.ConnectAsync(new TcpTransport(), link.Host.LocalEndPoint);
            link.Split = RemoteNetworkFactory.BuildHead(whole, layers, boundary, connection);
            return link;
        }

        private static Task<Link> StartDefaultAsync()
        {
            NetworkSettings whole = SplitHelpers.Whole(3, 2, ActivationTypes.Identity, inputBias: true);
            RecurrentLayerSettings[] layers = { SplitHelpers.Layer(4, ActivationTypes.HyperbolicTangens, bias: true), SplitHelpers.Layer(3, ActivationTypes.Sigmoid) };
            return StartAsync(whole, layers, 0);
        }

        [Fact]
        public async Task Propagate_OverTcp_ReturnsTheTailOutputs()
        {
            using (Link link = await StartDefaultAsync())
            {
                Quad[] outputs = link.Split.Propagate(new Quad[] { 0.5f, -1.25f, 0.75f });

                Assert.Equal(2, outputs.Length);
                Quad[] expected = link.Tail.Propagate(link.Split.Head.Values);
                Assert.Equal(expected.Select(v => (float)v), outputs.Select(v => (float)v));
                Assert.Equal(1, link.Split.Connection.PropagationCount);
            }
        }

        [Fact]
        public async Task Backpropagate_OverTcp_InputGradientsMatchFiniteDifferences()
        {
            using (Link link = await StartDefaultAsync())
            {
                Quad[] inputs = { 0.4f, -0.8f, 1.1f };
                Quad[] targets = { 0.25f, -0.5f };

                double[] numeric = new double[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    Quad original = inputs[i];
                    inputs[i] = original + (Quad)GradientChecker.EPSILON;
                    double plus = SplitHelpers.Loss(link.Split.Propagate(inputs), targets);
                    inputs[i] = original - (Quad)GradientChecker.EPSILON;
                    double minus = SplitHelpers.Loss(link.Split.Propagate(inputs), targets);
                    inputs[i] = original;
                    numeric[i] = (plus - minus) / (2 * GradientChecker.EPSILON);
                }

                link.Split.Propagate(inputs);
                link.Split.Backpropagate(targets);
                Quad[] analytic = link.Split.InputGradients;

                for (int i = 0; i < inputs.Length; i++)
                {
                    GradientChecker.AssertClose(numeric[i], analytic[i], $"input gradient [{i}]");
                }
            }
        }

        [Fact]
        public async Task Backpropagate_OverTcp_TrainsBothSides()
        {
            using (Link link = await StartDefaultAsync())
            {
                Quad[] inputs = { 0.5f, -0.5f, 0.25f };
                Quad[] targets = { 0.8f, -0.3f };
                float[] headBefore = SplitHelpers.TrainableSynapses((Network)link.Split.Head).Select(s => (float)s.Weight).ToArray();
                float[] tailBefore = SplitHelpers.TrainableSynapses(link.Tail).Select(s => (float)s.Weight).ToArray();

                link.Split.SetLearningRate(0.1f);
                double initialLoss = SplitHelpers.Loss(link.Split.Propagate(inputs), targets);

                for (int epoch = 0; epoch < 300; epoch++)
                {
                    link.Split.Propagate(inputs);
                    link.Split.Backpropagate(targets);
                }

                double finalLoss = SplitHelpers.Loss(link.Split.Propagate(inputs), targets);

                Assert.True(finalLoss < initialLoss / 10, $"loss went from {initialLoss} to {finalLoss}");
                Assert.NotEqual(headBefore, SplitHelpers.TrainableSynapses((Network)link.Split.Head).Select(s => (float)s.Weight));
                Assert.NotEqual(tailBefore, SplitHelpers.TrainableSynapses(link.Tail).Select(s => (float)s.Weight));
            }
        }

        [Fact]
        public async Task Backpropagate_OverTcp_WithTheWrongNumberOfTargets_SurfacesTheRemoteError()
        {
            using (Link link = await StartDefaultAsync())
            {
                link.Split.Propagate(new Quad[] { 0.1f, 0.2f, 0.3f });

                RpcRemoteException error = Assert.Throws<RpcRemoteException>(() => link.Split.Backpropagate(new Quad[] { 1, 2, 3 }));

                Assert.Equal(typeof(ArgumentException).FullName, error.RemoteType);

                // the connection survives a failed call
                Assert.Equal(2, link.Split.Propagate(new Quad[] { 0.1f, 0.2f, 0.3f }).Length);
            }
        }

        [Fact]
        public async Task Host_SeesTheConnectedHeadUntilItIsDisposed()
        {
            using (Link link = await StartDefaultAsync())
            {
                await WaitUntilAsync(() => link.Host.ConnectedPeers == 1);
                Assert.Equal(1, link.Host.ConnectedPeers);

                link.Split.Dispose();

                await WaitUntilAsync(() => link.Host.ConnectedPeers == 0);
                Assert.Equal(0, link.Host.ConnectedPeers);
            }
        }

        private static async Task WaitUntilAsync(Func<bool> condition)
        {
            Stopwatch watch = Stopwatch.StartNew();
            while (!condition() && watch.Elapsed < TimeSpan.FromSeconds(10))
            {
                await Task.Delay(20);
            }
        }
    }
}
