using System;
using System.Linq;
using System.Net;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.Remote;
using ANN.Net.Remote.Abstactions.Interfaces;
using ANN.Net.Tests.Transformers;
using RPC.Library;
using RPC.Library.Transports.Tcp;
using Xunit;

namespace ANN.Net.Tests.Remote
{
    /// <summary>
    /// Simulation of four machines on the loopback interface, every one on a random free port:
    /// machine 1 runs the input layer and the first hidden layer (depth head), machines 2 and 3 run half of
    /// hidden layers 2 and 3 each (breadth parts, machine 2 also coordinates), machine 4 runs the last hidden
    /// layer and the output layer (depth tail)
    /// </summary>
    public class FourMachineSimulationTests
    {
        private sealed class Simulation : IDisposable
        {
            public SplitNetwork Machine1Head;
            public PartitionNode Machine2Part;
            public PartitionCoordinator Machine2Coordinator;
            public RemoteNetworkHost Machine2Host;
            public PartitionNode Machine3Part;
            public Network Machine4Tail;
            public RemoteNetworkHost Machine4Host;

            public void Dispose()
            {
                this.Machine1Head?.Dispose();
                this.Machine2Host?.Dispose();
                this.Machine2Coordinator?.Dispose();
                this.Machine3Part?.Dispose();
                this.Machine2Part?.Dispose();
                this.Machine4Host?.Dispose();
            }
        }

        private static readonly IPEndPoint ANY_FREE_PORT = new IPEndPoint(IPAddress.Loopback, 0);

        private static async Task<Simulation> StartAsync()
        {
            // the whole network, described identically on every machine
            NetworkSettings whole = SplitHelpers.Whole(3, 2, ActivationTypes.Identity, inputBias: true);
            RecurrentLayerSettings[] hidden =
            {
                SplitHelpers.Layer(4, ActivationTypes.HyperbolicTangens, bias: true),   // machine 1
                SplitHelpers.Layer(6, ActivationTypes.Sigmoid, bias: true),             // machines 2 + 3, three neurons each
                SplitHelpers.Layer(4, ActivationTypes.HyperbolicTangens, bias: true),   // machines 2 + 3, two neurons each
                SplitHelpers.Layer(3, ActivationTypes.Sigmoid, bias: true)              // machine 4
            };

            Simulation simulation = new Simulation();
            TcpTransport tcp = new TcpTransport();

            // machine 4: the tail, served over TCP
            simulation.Machine4Tail = (Network)RemoteNetworkFactory.BuildTail(whole, hidden, 2);
            simulation.Machine4Host = await RemoteNetworkHost.StartAsync(simulation.Machine4Tail, tcp, ANY_FREE_PORT);

            // machines 2 and 3: the two parts of hidden layers 1 and 2; part 0 accepts, part 1 connects
            NetworkPartition partition = RemoteNetworkFactory.Partition(whole, hidden, 1, 2, 2);
            simulation.Machine2Part = new PartitionNode(partition, 0);
            await simulation.Machine2Part.StartAsync(tcp, ANY_FREE_PORT);

            simulation.Machine3Part = new PartitionNode(partition, 1);
            await simulation.Machine3Part.ConnectAsync(0, tcp, simulation.Machine2Part.LocalEndPoint);

            // machine 2 also coordinates: it connects to machine 4 and serves the section to machine 1
            RpcPeer<NoService, IRemoteNetworkPart> toMachine4 = await RemoteNetworkPeer.ConnectAsync(tcp, simulation.Machine4Host.LocalEndPoint);
            simulation.Machine2Coordinator = new PartitionCoordinator(simulation.Machine2Part, toMachine4.Remote, toMachine4);
            simulation.Machine2Host = await RemoteNetworkHost.StartAsync(simulation.Machine2Coordinator, tcp, ANY_FREE_PORT);

            // machine 1: the head, connected to machine 2
            SynapseConnection toMachine2 = await SynapseConnection.ConnectAsync(tcp, simulation.Machine2Host.LocalEndPoint);
            simulation.Machine1Head = RemoteNetworkFactory.BuildHead(whole, hidden, 0, toMachine2);

            Assert.True(simulation.Machine2Part.IsConnected);
            Assert.True(simulation.Machine3Part.IsConnected);
            return simulation;
        }

        [Fact]
        public async Task Propagate_AcrossFourMachines_ReturnsTheTailOutputsAndIsStable()
        {
            using (Simulation simulation = await StartAsync())
            {
                Quad[] first = simulation.Machine1Head.Propagate(new Quad[] { 0.5f, -1.25f, 0.75f });
                Quad[] second = simulation.Machine1Head.Propagate(new Quad[] { 0.5f, -1.25f, 0.75f });

                Assert.Equal(2, first.Length);
                for (int i = 0; i < first.Length; i++)
                {
                    Assert.Equal((float)first[i], (float)second[i], 5);
                }
                Assert.Equal(2, simulation.Machine1Head.Connection.PropagationCount);
            }
        }

        [Fact]
        public async Task Backpropagate_AcrossFourMachines_InputGradientsMatchFiniteDifferences()
        {
            using (Simulation simulation = await StartAsync())
            {
                Quad[] inputs = { 0.4f, -0.8f, 1.1f };
                Quad[] targets = { 0.25f, -0.5f };

                double[] numeric = new double[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    Quad original = inputs[i];
                    inputs[i] = original + (Quad)GradientChecker.EPSILON;
                    double plus = SplitHelpers.Loss(simulation.Machine1Head.Propagate(inputs), targets);
                    inputs[i] = original - (Quad)GradientChecker.EPSILON;
                    double minus = SplitHelpers.Loss(simulation.Machine1Head.Propagate(inputs), targets);
                    inputs[i] = original;
                    numeric[i] = (plus - minus) / (2 * GradientChecker.EPSILON);
                }

                simulation.Machine1Head.Propagate(inputs);
                simulation.Machine1Head.Backpropagate(targets);
                Quad[] analytic = simulation.Machine1Head.InputGradients;

                for (int i = 0; i < inputs.Length; i++)
                {
                    GradientChecker.AssertClose(numeric[i], analytic[i], $"input gradient [{i}]");
                }
            }
        }

        [Fact]
        public async Task Train_AcrossFourMachines_LossDropsAndEveryMachineLearns()
        {
            using (Simulation simulation = await StartAsync())
            {
                Quad[][] inputs = { new Quad[] { 0, 0, 1 }, new Quad[] { 0, 1, 0 }, new Quad[] { 1, 0, 0 } };
                Quad[][] targets = { new Quad[] { 0.8f, -0.2f }, new Quad[] { -0.4f, 0.6f }, new Quad[] { 0.1f, 0.9f } };

                float[] machine1Before = SplitHelpers.TrainableSynapses((Network)simulation.Machine1Head.Head).Select(s => (float)s.Weight).ToArray();
                float[] machine2Before = simulation.Machine2Part.Network.Synapses.Select(s => (float)s.Synapse.Weight).ToArray();
                float[] machine3Before = simulation.Machine3Part.Network.Synapses.Select(s => (float)s.Synapse.Weight).ToArray();
                float[] machine4Before = SplitHelpers.TrainableSynapses(simulation.Machine4Tail).Select(s => (float)s.Weight).ToArray();

                simulation.Machine1Head.SetLearningRate(0.2f);
                double initial = inputs.Select((x, i) => SplitHelpers.Loss(simulation.Machine1Head.Propagate(x), targets[i])).Sum();

                for (int epoch = 0; epoch < 200; epoch++)
                {
                    for (int s = 0; s < inputs.Length; s++)
                    {
                        simulation.Machine1Head.Propagate(inputs[s]);
                        simulation.Machine1Head.Backpropagate(targets[s]);
                    }
                }

                double final = inputs.Select((x, i) => SplitHelpers.Loss(simulation.Machine1Head.Propagate(x), targets[i])).Sum();

                Assert.True(final < initial / 10, $"loss went from {initial} to {final}");
                Assert.NotEqual(machine1Before, SplitHelpers.TrainableSynapses((Network)simulation.Machine1Head.Head).Select(s => (float)s.Weight));
                Assert.NotEqual(machine2Before, simulation.Machine2Part.Network.Synapses.Select(s => (float)s.Synapse.Weight));
                Assert.NotEqual(machine3Before, simulation.Machine3Part.Network.Synapses.Select(s => (float)s.Synapse.Weight));
                Assert.NotEqual(machine4Before, SplitHelpers.TrainableSynapses(simulation.Machine4Tail).Select(s => (float)s.Weight));
            }
        }

        [Fact]
        public async Task Backpropagate_AcrossFourMachines_WrongTargetCountSurfacesTheRemoteError()
        {
            using (Simulation simulation = await StartAsync())
            {
                simulation.Machine1Head.Propagate(new Quad[] { 0.1f, 0.2f, 0.3f });

                RpcRemoteException error = Assert.Throws<RpcRemoteException>(() => simulation.Machine1Head.Backpropagate(new Quad[] { 1, 2, 3 }));

                Assert.Equal(typeof(RpcRemoteException).FullName, error.RemoteType);
                Assert.Equal(2, simulation.Machine1Head.Propagate(new Quad[] { 0.1f, 0.2f, 0.3f }).Length);
            }
        }
    }
}
