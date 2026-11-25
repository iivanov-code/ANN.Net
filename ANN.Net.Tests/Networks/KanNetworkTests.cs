using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Compact;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Kolmogorov-Arnold networks built as graph networks by <see cref="NetworkFactory.BuildKan"/>
    /// </summary>
    public class KanNetworkTests
    {
        private static KanSettings Settings(params int[] sizes) => new KanSettings { LayerSizes = sizes, Seed = 7 };

        private static void SetLearningRate(IGraphNetwork network, float rate)
        {
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = rate;
        }

        private static float Spline(float x)
        {
            Quad value = x;
            return Utils.NetworkUtils.GetActivation(ActivationTypes.CubicBSpline).Activation(ref value);
        }

        [Theory]
        [InlineData(KanBasisTypes.BSpline, true, 8 + 1)]
        [InlineData(KanBasisTypes.BSpline, false, 8)]
        [InlineData(KanBasisTypes.Gaussian, true, 6 + 1)]
        public void Network_HasOneCoefficientPerEdgeAndFeature(KanBasisTypes basis, bool withBase, int features)
        {
            KanSettings settings = Settings(3, 4, 2);
            settings.Basis = basis;
            settings.BaseActivation = withBase ? ActivationTypes.SWISH : null;
            IKanNetwork network = NetworkFactory.BuildKan(settings);

            // grid of 5 intervals: 5 + 3 cubic B-splines or 5 + 1 Gaussians, and the base function
            Assert.Equal(((3 * 4) + (4 * 2)) * features, network.ParameterCount);
            Assert.Equal(new[] { 3, 4, 2 }, network.LayerSizes);
            Assert.Equal(2, network.Propagate(0.1f, 0.2f, 0.3f).Length);
        }

        [Fact]
        public void TrainableGrid_AlsoCountsThePlaceOfEveryBasisFunction()
        {
            KanSettings settings = Settings(2, 1);
            settings.TrainableGrid = true;
            IKanNetwork network = NetworkFactory.BuildKan(settings);

            // 2 · 9 coefficients, and the scale and shift of the 2 · 8 basis functions
            Assert.Equal((2 * 9) + (2 * 8 * 2), network.ParameterCount);
        }

        [Fact]
        public void BSplines_AddUpToOneOnTheGrid()
        {
            KanSettings settings = Settings(1, 1);
            settings.BaseActivation = null;
            settings.GridMinimum = -2;
            settings.GridMaximum = 3;
            GraphNetworkSettings graph = NetworkFactory.KanGraph(settings);
            // the output connections are the coefficients: all 1 gives the sum of the basis functions
            var outputs = new HashSet<uint>(graph.Nodes.Where(x => x.Type == NodeType.OUTPUT).Select(x => x.Id));
            foreach (GraphConnection connection in graph.Connections.Where(x => outputs.Contains(x.ToNode))) connection.Weight = 1;
            IGraphNetwork network = NetworkFactory.BuildGraph(graph);

            for (float x = -2; x <= 3; x += 0.125f)
            {
                Assert.Equal(1f, network.Propagate(x)[0], 5);
            }
        }

        [Fact]
        public void EvaluateEdge_IsTheWeightedSumOfTheFeatures()
        {
            KanSettings settings = Settings(1, 1);
            settings.GridSize = 4;
            IKanNetwork network = NetworkFactory.BuildKan(settings);
            GraphNetworkSettings graph = NetworkFactory.KanGraph(settings);

            // nodes: input 0, bias 1, base 2, splines 3 to 9, output 10; centres -1.5 to 1.5, h = 0.5
            Dictionary<uint, Quad> weights = network.ConnectionWeights.ToDictionary(x => x.Key, x => x.Value);
            float Coefficient(uint node) => weights[graph.Connections.Single(x => x.FromNode == node && x.ToNode == 10).Id];

            const float x = 0.3f;
            float expected = Coefficient(2) * x / (1 + MathF.Exp(-x));
            for (uint node = 3; node <= 9; node++)
            {
                float centre = -1.5f + ((node - 3) * 0.5f);
                expected += Coefficient(node) * Spline((x - centre) / 0.5f);
            }

            Assert.Equal(expected, network.EvaluateEdge(0, 0, 0, x), 5);
            Assert.Equal(expected, network.Propagate(x)[0], 5);
        }

        [Fact]
        public void Propagate_AddsUpTheEdgeFunctionsLayerByLayer()
        {
            KanSettings settings = Settings(2, 3, 2);
            settings.SplineInitScale = 0.5f;
            IKanNetwork network = NetworkFactory.BuildKan(settings);

            var random = new Random(3);
            for (int sample = 0; sample < 10; sample++)
            {
                Quad[] input = { (Quad)((random.NextDouble() * 2) - 1), (Quad)((random.NextDouble() * 2) - 1) };
                var hidden = new Quad[3];
                for (int j = 0; j < 3; j++) hidden[j] = network.EvaluateEdge(0, 0, j, input[0]) + network.EvaluateEdge(0, 1, j, input[1]);

                Quad[] actual = network.Propagate(input);
                for (int k = 0; k < 2; k++)
                {
                    float expected = Enumerable.Range(0, 3).Sum(j => (float)network.EvaluateEdge(1, j, k, hidden[j]));
                    Assert.Equal(expected, actual[k], 4);
                }
            }
        }

        [Fact]
        public void Training_ChangesOnlyTheCoefficients_UnlessTheGridIsTrainable()
        {
            foreach (bool trainableGrid in new[] { false, true })
            {
                KanSettings settings = Settings(2, 2, 1);
                settings.TrainableGrid = trainableGrid;
                IKanNetwork network = NetworkFactory.BuildKan(settings);
                GraphNetworkSettings graph = NetworkFactory.KanGraph(settings);
                Dictionary<uint, Quad> before = network.ConnectionWeights.ToDictionary(x => x.Key, x => x.Value);

                SetLearningRate(network, 0.05f);
                for (int i = 0; i < 20; i++)
                {
                    network.Propagate(0.3f, -0.6f);
                    network.Backpropagate(1);
                }

                foreach (GraphConnection connection in graph.Connections)
                {
                    bool changed = network.ConnectionWeights[connection.Id] != before[connection.Id];
                    if (!connection.Trainable) Assert.False(changed, $"Connection {connection.Id} is fixed");
                }

                Assert.Equal(trainableGrid, graph.Connections.Any(x => x.Trainable && network.ConnectionWeights[x.Id] != before[x.Id]
                                                                     && graph.Nodes.Single(n => n.Id == x.FromNode).Type == NodeType.BIAS));
            }
        }

        [Fact]
        public void BackpropagateGradient_MatchesTheFiniteDifferenceOfTheInputs()
        {
            KanSettings settings = Settings(2, 3, 1);
            settings.SplineInitScale = 0.5f;
            settings.OptimizerType = OptimizerTypes.Normal;
            IKanNetwork network = NetworkFactory.BuildKan(settings);
            SetLearningRate(network, 0);

            Quad[] input = { 0.2f, -0.45f };
            network.Propagate(input);
            Quad[] gradient = network.BackpropagateGradient(new Quad[] { 1 });

            for (int i = 0; i < 2; i++)
            {
                const float step = 1e-3f;
                Quad[] up = (Quad[])input.Clone(), down = (Quad[])input.Clone();
                up[i] += step;
                down[i] -= step;
                float numeric = (network.Propagate(up)[0] - network.Propagate(down)[0]) / (2 * step);
                Assert.Equal(numeric, gradient[i], 2);
            }
        }

        [Fact]
        public void OneLayer_LearnsASumOfFunctionsOfOneVariable()
        {
            // sin(πx) + y² is a sum of one function of each input: exactly what one KAN layer computes
            KanSettings settings = Settings(2, 1);
            IKanNetwork network = NetworkFactory.BuildKan(settings);
            SetLearningRate(network, 0.02f);

            float Target(float x, float y) => MathF.Sin(MathF.PI * x) + (y * y);
            float error = Train(network, Target, epochs: 150);

            Assert.True(error < 0.001f, $"Mean squared error {error}");
            // the edges learned the two functions, up to a constant that moves from one edge to the other
            float shift = network.EvaluateEdge(0, 0, 0, 0);
            for (float v = -0.9f; v <= 0.9f; v += 0.3f)
            {
                Assert.InRange((float)(network.EvaluateEdge(0, 0, 0, v) - shift) - MathF.Sin(MathF.PI * v), -0.1f, 0.1f);
                Assert.InRange((float)(network.EvaluateEdge(0, 1, 0, v) + shift) - (v * v), -0.1f, 0.1f);
            }
        }

        [Fact]
        public void TwoLayers_LearnAProduct()
        {
            // x·y = ((x + y)² - (x - y)²) / 4 needs a second layer
            KanSettings settings = Settings(2, 2, 1);
            settings.GridMinimum = -1.5f;
            settings.GridMaximum = 1.5f;
            IKanNetwork network = NetworkFactory.BuildKan(settings);
            SetLearningRate(network, 0.01f);

            float error = Train(network, (x, y) => x * y, epochs: 600);

            Assert.True(error < 0.003f, $"Mean squared error {error}");
        }

        private static float Train(IKanNetwork network, Func<float, float, float> target, int epochs)
        {
            var random = new Random(11);
            var samples = Enumerable.Range(0, 100).Select(_ => ((float)((random.NextDouble() * 2) - 1), (float)((random.NextDouble() * 2) - 1))).ToArray();
            for (int epoch = 0; epoch < epochs; epoch++)
            {
                foreach ((float x, float y) in samples.OrderBy(_ => random.Next()))
                {
                    network.Propagate(x, y);
                    network.Backpropagate(target(x, y));
                }
            }

            var test = new Random(5);
            float error = 0;
            const int tests = 200;
            for (int i = 0; i < tests; i++)
            {
                float x = (float)((test.NextDouble() * 1.8) - 0.9), y = (float)((test.NextDouble() * 1.8) - 0.9);
                float difference = network.Propagate(x, y)[0] - target(x, y);
                error += difference * difference / tests;
            }

            return error;
        }

        [Fact]
        public void SavedNetwork_LoadsAsAKolmogorovArnoldNetworkWithFixedGrid()
        {
            KanSettings settings = Settings(2, 3, 1);
            settings.Basis = KanBasisTypes.Gaussian;
            IKanNetwork network = NetworkFactory.BuildKan(settings);
            SetLearningRate(network, 0.02f);
            network.Propagate(0.5f, 0.1f);
            network.Backpropagate(1);

            using var stream = new MemoryStream();
            ((ANNetwork)network).Save(stream);
            stream.Position = 0;
            var loaded = Assert.IsAssignableFrom<IKanNetwork>(ANNetwork.Load(stream));

            Assert.Equal(network.LayerSizes, loaded.LayerSizes);
            Assert.Equal(network.ParameterCount, loaded.ParameterCount);
            Assert.Equal((float)network.Propagate(0.2f, -0.7f)[0], loaded.Propagate(0.2f, -0.7f)[0], 6);
            Assert.Equal((float)network.EvaluateEdge(1, 2, 0, 0.4f), loaded.EvaluateEdge(1, 2, 0, 0.4f), 6);

            // the grid stays fixed after loading
            GraphNetworkSettings graph = NetworkFactory.KanGraph(settings);
            Dictionary<uint, Quad> before = loaded.ConnectionWeights.ToDictionary(x => x.Key, x => x.Value);
            SetLearningRate(loaded, 0.05f);
            loaded.Propagate(0.5f, 0.1f);
            loaded.Backpropagate(2);
            Assert.All(graph.Connections.Where(x => !x.Trainable), x => Assert.Equal(before[x.Id], loaded.ConnectionWeights[x.Id]));
            Assert.Contains(graph.Connections, x => x.Trainable && loaded.ConnectionWeights[x.Id] != before[x.Id]);
        }

        [Fact]
        public void CompactNetwork_ComputesTheSameOutputs()
        {
            IKanNetwork network = NetworkFactory.BuildKan(Settings(3, 4, 2));
            ICompactNetwork compact = NetworkCompactor.Compact(network);

            Quad[] input = { 0.1f, -0.8f, 0.55f };
            Quad[] expected = network.Propagate(input);
            Quad[] actual = compact.Propagate(input);
            for (int i = 0; i < 2; i++) Assert.Equal((float)expected[i], actual[i], 5);
        }

        [Fact]
        public void SameSeed_BuildsTheSameNetwork()
        {
            IKanNetwork first = NetworkFactory.BuildKan(Settings(2, 2, 1));
            IKanNetwork second = NetworkFactory.BuildKan(Settings(2, 2, 1));

            Assert.Equal(first.ConnectionWeights, second.ConnectionWeights);
        }

        [Fact]
        public void InvalidSettings_AreRejected()
        {
            Assert.Throws<ArgumentNullException>(() => NetworkFactory.BuildKan(null));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildKan(Settings(2)));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildKan(Settings(2, 0, 1)));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildKan(new KanSettings { LayerSizes = new[] { 2, 1 }, GridSize = 0 }));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildKan(new KanSettings { LayerSizes = new[] { 2, 1 }, GridMinimum = 1, GridMaximum = 1 }));
            Assert.Throws<ArgumentException>(() => NetworkFactory.BuildKan(new KanSettings { LayerSizes = new[] { 2, 1 }, SplineInitScale = -1 }));
            Assert.Throws<NotSupportedException>(() => NetworkFactory.BuildKan(new KanSettings { LayerSizes = new[] { 2, 1 }, BaseActivation = ActivationTypes.SoftMax }));

            IKanNetwork network = NetworkFactory.BuildKan(Settings(2, 1));
            Assert.Throws<ArgumentOutOfRangeException>(() => network.EvaluateEdge(1, 0, 0, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => network.EvaluateEdge(0, 2, 0, 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => network.EvaluateEdge(0, 0, 1, 0));
        }
    }
}
