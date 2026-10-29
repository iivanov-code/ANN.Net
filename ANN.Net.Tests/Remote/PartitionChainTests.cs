using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.Remote;
using ANN.Net.Tests.Transformers;
using Xunit;

namespace ANN.Net.Tests.Remote
{
    /// <summary>
    /// Four stages in one process: a head with whole layers, two parts with half layers wired through the
    /// thread pool, and a tail with whole layers behind an in-process <see cref="RemoteNetworkPart"/>
    /// </summary>
    public class PartitionChainTests
    {
        internal sealed class Chain : IDisposable
        {
            public SplitNetwork Head;
            public PartitionNode[] Nodes;
            public PartitionCoordinator Coordinator;
            public Network Tail;

            public IEnumerable<BaseSynapse> AllSynapses()
            {
                return SplitHelpers.TrainableSynapses((Network)this.Head.Head)
                                   .Concat(this.Nodes.SelectMany(n => n.Network.Synapses.Select(s => s.Synapse)))
                                   .Concat(SplitHelpers.TrainableSynapses(this.Tail));
            }

            public void Dispose()
            {
                this.Head?.Dispose();
                this.Coordinator?.Dispose();
                foreach (PartitionNode node in this.Nodes ?? Array.Empty<PartitionNode>()) node.Dispose();
            }
        }

        internal static Chain Build(NetworkSettings whole, RecurrentLayerSettings[] hidden, int firstLayer, int lastLayer, int parts)
        {
            Chain chain = new Chain();
            chain.Tail = (Network)RemoteNetworkFactory.BuildTail(whole, hidden, lastLayer);

            NetworkPartition partition = RemoteNetworkFactory.Partition(whole, hidden, firstLayer, lastLayer, parts);
            chain.Nodes = Enumerable.Range(0, parts).Select(p => new PartitionNode(partition, p)).ToArray();
            for (int p = 1; p < parts; p++)
            {
                for (int q = 0; q < p; q++) chain.Nodes[p].ConnectInProcess(chain.Nodes[q]);
            }

            chain.Coordinator = new PartitionCoordinator(chain.Nodes[0], RemoteNetworkPart.Create(chain.Tail));
            chain.Head = RemoteNetworkFactory.BuildHead(whole, hidden, firstLayer - 1, new SynapseConnection(chain.Coordinator));
            return chain;
        }

        private static Chain BuildDefault(int parts = 2)
        {
            NetworkSettings whole = SplitHelpers.Whole(3, 2, ActivationTypes.Identity, inputBias: true);
            RecurrentLayerSettings[] hidden =
            {
                SplitHelpers.Layer(4, ActivationTypes.HyperbolicTangens, bias: true),
                SplitHelpers.Layer(6, ActivationTypes.Sigmoid, bias: true),
                SplitHelpers.Layer(4, ActivationTypes.HyperbolicTangens, bias: true),
                SplitHelpers.Layer(3, ActivationTypes.Sigmoid, bias: true)
            };

            return Build(whole, hidden, 1, 2, parts);
        }

        [Fact]
        public void Chain_HasTheExpectedShape()
        {
            using (Chain chain = BuildDefault())
            {
                Assert.Equal(3, chain.Head.InputCount);
                Assert.Equal(4, chain.Head.BoundaryCount);
                Assert.Equal(2, chain.Head.OutputCount);
                Assert.Equal(4, chain.Coordinator.Partition.InputWidth);
                Assert.Equal(4, chain.Coordinator.Partition.OutputWidth);
                Assert.Equal(3, chain.Nodes[0].Network.Slice(0).Count);
                Assert.Equal(3, chain.Nodes[1].Network.Slice(0).Count);
                Assert.Equal(2, chain.Nodes[0].Network.Slice(1).Count);
                Assert.True(chain.Nodes.All(n => n.IsConnected));
                Assert.Single(chain.Tail.HiddenLayers);
            }
        }

        [Theory]
        [InlineData(2)]
        [InlineData(3)]
        public void Propagate_Twice_GivesTheSameOutput(int parts)
        {
            using (Chain chain = BuildDefault(parts))
            {
                Quad[] first = chain.Head.Propagate(new Quad[] { 0.5f, -1.25f, 0.75f });
                Quad[] second = chain.Head.Propagate(new Quad[] { 0.5f, -1.25f, 0.75f });

                Assert.Equal(2, first.Length);
                for (int i = 0; i < first.Length; i++)
                {
                    // slices arrive in any order, so a neuron may add its inputs in another order than last time: equal up to float rounding
                    Assert.Equal((float)first[i], (float)second[i], 5);
                }
            }
        }

        [Fact]
        public void Backpropagate_InputGradientsMatchFiniteDifferences()
        {
            using (Chain chain = BuildDefault())
            {
                Quad[] inputs = { 0.4f, -0.8f, 1.1f };
                Quad[] targets = { 0.25f, -0.5f };

                double[] numeric = new double[inputs.Length];
                for (int i = 0; i < inputs.Length; i++)
                {
                    Quad original = inputs[i];
                    inputs[i] = original + (Quad)GradientChecker.EPSILON;
                    double plus = SplitHelpers.Loss(chain.Head.Propagate(inputs), targets);
                    inputs[i] = original - (Quad)GradientChecker.EPSILON;
                    double minus = SplitHelpers.Loss(chain.Head.Propagate(inputs), targets);
                    inputs[i] = original;
                    numeric[i] = (plus - minus) / (2 * GradientChecker.EPSILON);
                }

                chain.Head.Propagate(inputs);
                chain.Head.Backpropagate(targets);
                Quad[] analytic = chain.Head.InputGradients;

                for (int i = 0; i < inputs.Length; i++)
                {
                    GradientChecker.AssertClose(numeric[i], analytic[i], $"input gradient [{i}]");
                }
            }
        }

        [Theory]
        [InlineData(2)]
        [InlineData(3)]
        public void Backpropagate_WeightUpdatesOnAllFourStagesMatchFiniteDifferences(int parts)
        {
            using (Chain chain = BuildDefault(parts))
            {
                Quad[] inputs = { 0.4f, -0.8f, 1.1f };
                Quad[] targets = { 0.25f, -0.5f };
                BaseSynapse[] synapses = chain.AllSynapses().ToArray();
                Assert.Equal((3 + 1) * 4 + (4 + 1) * 6 + (6 + 1) * 4 + (4 + 1) * 3 + (3 + 1) * 2, synapses.Length);

                double[] numeric = new double[synapses.Length];
                for (int s = 0; s < synapses.Length; s++)
                {
                    Quad original = synapses[s].Weight;
                    synapses[s].Weight = original + (Quad)GradientChecker.EPSILON;
                    double plus = SplitHelpers.Loss(chain.Head.Propagate(inputs), targets);
                    synapses[s].Weight = original - (Quad)GradientChecker.EPSILON;
                    double minus = SplitHelpers.Loss(chain.Head.Propagate(inputs), targets);
                    synapses[s].Weight = original;
                    numeric[s] = (plus - minus) / (2 * GradientChecker.EPSILON);
                }

                float[] before = synapses.Select(s => (float)s.Weight).ToArray();
                chain.Head.SetLearningRate(1f);

                chain.Head.Propagate(inputs);
                chain.Head.Backpropagate(targets);

                for (int s = 0; s < synapses.Length; s++)
                {
                    GradientChecker.AssertClose(numeric[s], before[s] - (float)synapses[s].Weight, $"weight gradient of synapse {s}");
                }
            }
        }

        [Fact]
        public void Train_RegressionTargets_LossDropsAndEveryStageLearns()
        {
            using (Chain chain = BuildDefault())
            {
                Quad[][] inputs = { new Quad[] { 0, 0, 1 }, new Quad[] { 0, 1, 0 }, new Quad[] { 1, 0, 0 } };
                Quad[][] targets = { new Quad[] { 0.8f, -0.2f }, new Quad[] { -0.4f, 0.6f }, new Quad[] { 0.1f, 0.9f } };
                float[] headBefore = SplitHelpers.TrainableSynapses((Network)chain.Head.Head).Select(s => (float)s.Weight).ToArray();
                float[][] partsBefore = chain.Nodes.Select(n => n.Network.Synapses.Select(s => (float)s.Synapse.Weight).ToArray()).ToArray();
                float[] tailBefore = SplitHelpers.TrainableSynapses(chain.Tail).Select(s => (float)s.Weight).ToArray();

                chain.Head.SetLearningRate(0.2f);
                double initial = inputs.Select((x, i) => SplitHelpers.Loss(chain.Head.Propagate(x), targets[i])).Sum();

                for (int epoch = 0; epoch < 300; epoch++)
                {
                    for (int s = 0; s < inputs.Length; s++)
                    {
                        chain.Head.Propagate(inputs[s]);
                        chain.Head.Backpropagate(targets[s]);
                    }
                }

                double final = inputs.Select((x, i) => SplitHelpers.Loss(chain.Head.Propagate(x), targets[i])).Sum();

                Assert.True(final < initial / 10, $"loss went from {initial} to {final}");
                Assert.NotEqual(headBefore, SplitHelpers.TrainableSynapses((Network)chain.Head.Head).Select(s => (float)s.Weight));
                for (int p = 0; p < chain.Nodes.Length; p++)
                {
                    Assert.NotEqual(partsBefore[p], chain.Nodes[p].Network.Synapses.Select(s => (float)s.Synapse.Weight));
                }

                Assert.NotEqual(tailBefore, SplitHelpers.TrainableSynapses(chain.Tail).Select(s => (float)s.Weight));
            }
        }

        [Fact]
        public void Backpropagate_BeforePropagate_Throws()
        {
            using (Chain chain = BuildDefault())
            {
                Assert.Throws<InvalidOperationException>(() => chain.Head.Backpropagate(new Quad[] { 0, 0 }));
            }
        }
    }
}
