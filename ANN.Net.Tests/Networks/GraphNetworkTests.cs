using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Networks with a free-form acyclic topology, as produced by NEAT, built by <see cref="NetworkFactory.BuildGraph"/>
    /// </summary>
    public class GraphNetworkTests
    {
        private static GraphNetworkSettings Linear(IEnumerable<GraphNode> nodes, IEnumerable<GraphConnection> connections)
        {
            return new GraphNetworkSettings
            {
                Nodes = nodes.ToList(),
                Connections = connections.ToList(),
                HiddenActivationType = ActivationTypes.Identity,
                OutputActivationType = ActivationTypes.Identity
            };
        }

        private static GraphNode Node(uint id, NodeType type) => new GraphNode(id, type);

        private static GraphConnection Link(uint id, uint from, uint to, float weight, bool enabled = true) => new GraphConnection(id, from, to, weight, enabled);

        [Fact]
        public void Propagate_FollowsConnectionsThatSkipLayers()
        {
            // 0,1 inputs, 2 bias, 3 and 4 hidden (4 is fed by 3), 5 output fed by everything
            IGraphNetwork network = NetworkFactory.BuildGraph(Linear(
                new[] { Node(0, NodeType.INPUT), Node(1, NodeType.INPUT), Node(2, NodeType.BIAS), Node(3, NodeType.HIDDEN), Node(4, NodeType.HIDDEN), Node(5, NodeType.OUTPUT) },
                new[]
                {
                    Link(0, 0, 3, 2), Link(1, 1, 3, 1), Link(2, 3, 4, 3), Link(3, 0, 5, 1), Link(4, 3, 5, -1), Link(5, 4, 5, 0.5f), Link(6, 2, 5, 0.25f)
                }));

            // h3 = 2*x0 + x1, h4 = 3*h3, out = x0 - h3 + 0.5*h4 + 0.25
            float x0 = 1.5f, x1 = -2f;
            float h3 = 2 * x0 + x1;
            float expected = x0 - h3 + 0.5f * 3 * h3 + 0.25f;

            Assert.Equal(expected, network.Propagate(x0, x1)[0], 4);
            Assert.Equal(expected, network.Propagate(x0, x1)[0], 4);
        }

        [Fact]
        public void Propagate_KeepsTheInputAndOutputOrderOfTheNodes()
        {
            IGraphNetwork network = NetworkFactory.BuildGraph(Linear(
                new[] { Node(9, NodeType.OUTPUT), Node(7, NodeType.INPUT), Node(8, NodeType.OUTPUT), Node(6, NodeType.INPUT) },
                new[] { Link(0, 7, 9, 1), Link(1, 6, 8, 1) }));

            Assert.Equal(new[] { 3f, 4f }, network.Propagate(3, 4).Select(x => (float)x));
        }

        [Fact]
        public void Build_PrunesDisabledConnectionsAndDeadBranches()
        {
            IGraphNetwork network = NetworkFactory.BuildGraph(Linear(
                new[] { Node(0, NodeType.INPUT), Node(1, NodeType.HIDDEN), Node(2, NodeType.HIDDEN), Node(3, NodeType.HIDDEN), Node(4, NodeType.OUTPUT) },
                new[]
                {
                    Link(0, 0, 4, 2),
                    Link(1, 0, 1, 1),               // 1 is a dead end: its only way out is disabled
                    Link(2, 1, 4, 1, enabled: false),
                    Link(3, 2, 4, 1),               // 2 has no inputs, it would never fire
                    Link(4, 0, 3, 1),               // 3 feeds 1 only, so it cannot reach the output either
                    Link(5, 3, 1, 1)
                }));

            Assert.Empty(network.HiddenNodeIds);
            Assert.Equal(new uint[] { 0 }, network.ConnectionWeights.Keys);
            Assert.Equal(6f, network.Propagate(3)[0], 4);

            // backpropagation must not wait for the pruned branches either
            Assert.Equal(0f, (float)network.Backpropagate(6), 4);
        }

        [Fact]
        public void Build_RejectsCyclesAndInvalidConnections()
        {
            GraphNode[] nodes = { Node(0, NodeType.INPUT), Node(1, NodeType.HIDDEN), Node(2, NodeType.HIDDEN), Node(3, NodeType.OUTPUT) };

            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildGraph(Linear(nodes, new[] { Link(0, 0, 1, 1), Link(1, 1, 2, 1), Link(2, 2, 1, 1), Link(3, 2, 3, 1) })));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildGraph(Linear(nodes, new[] { Link(0, 3, 1, 1) })));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildGraph(Linear(nodes, new[] { Link(0, 1, 0, 1) })));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildGraph(Linear(nodes, new[] { Link(0, 0, 42, 1) })));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildGraph(Linear(nodes, new[] { Link(0, 0, 3, 1), Link(0, 0, 1, 1) })));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildGraph(Linear(nodes.Append(Node(0, NodeType.HIDDEN)), Array.Empty<GraphConnection>())));

            // a disabled connection that would close a cycle is fine, it is left out
            NetworkFactory.BuildGraph(Linear(nodes, new[] { Link(0, 0, 1, 1), Link(1, 1, 2, 1), Link(2, 2, 1, 1, enabled: false), Link(3, 2, 3, 1) }));
        }

        [Fact]
        public void Backpropagate_TrainsTheGraphAndExposesTheWeights()
        {
            IGraphNetwork network = NetworkFactory.BuildGraph(Linear(
                new[] { Node(0, NodeType.INPUT), Node(1, NodeType.INPUT), Node(2, NodeType.BIAS), Node(3, NodeType.OUTPUT) },
                new[] { Link(10, 0, 3, 0.1f), Link(11, 1, 3, 0.1f), Link(12, 2, 3, 0.1f) }));

            foreach (LearningRateOptimizer optimizer in network.Optimizers)
            {
                optimizer.LearningRate = 0.05f;
            }

            var random = new Random(1);
            for (int i = 0; i < 2000; i++)
            {
                float x0 = (float)random.NextDouble(), x1 = (float)random.NextDouble();
                network.Propagate(x0, x1);
                network.Backpropagate(2 * x0 - x1 + 0.5f);
            }

            IReadOnlyDictionary<uint, Quad> weights = network.ConnectionWeights;
            Assert.Equal(2f, weights[10], 1);
            Assert.Equal(-1f, weights[11], 1);
            Assert.Equal(0.5f, weights[12], 1);
        }
    }
}
