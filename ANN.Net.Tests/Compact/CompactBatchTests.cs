using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using Xunit;

namespace ANN.Net.Tests.Compact
{
    /// <summary>
    /// Batches, spans and parallel convolutions give exactly the outputs of one sample at a time
    /// </summary>
    public class CompactBatchTests
    {
        private static float[][] Samples(int width, int count, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, count)
                             .Select(i => Enumerable.Range(0, width).Select(j => (i + j) % 7 == 0 ? 0f : (float)((random.NextDouble() * 4) - 2)).ToArray())
                             .ToArray();
        }

        private static int[] Bits(IEnumerable<float> values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

        private static void AssertBatchMatchesSingle(ICompactNetwork network, float[][] samples)
        {
            float[][] batch = network.PropagateBatch(samples);
            Assert.Equal(samples.Length, batch.Length);

            var span = new float[network.OutputCount];
            for (int i = 0; i < samples.Length; i++)
            {
                float[] single = network.Propagate(samples[i].Select(x => (Quad)x).ToArray()).Select(x => (float)x).ToArray();
                network.Propagate(samples[i], span);
                Assert.Equal(Bits(single), Bits(batch[i]));
                Assert.Equal(Bits(single), Bits(span));
            }
        }

        private static INetwork Layered(bool normalize, bool softMax) => NetworkFactory.BuildMatrixFFN(new NetworkSettings
        {
            Seed = 2,
            IdentityGenerator = new IncrementalGenerator(),
            ShouldNormalize = normalize,
            MinMaxValues = normalize ? Enumerable.Range(0, 70).Select(i => Tuple.Create<Quad, Quad>(-2 - (i % 3), 2 + (i % 5))).ToList() : null,
            OutputApplySoftMax = softMax,
            InputSettings = new InputLayerSettings { NeuronsCount = 70, HasBiasNeuron = true, ActivationType = normalize ? ActivationTypes.Sigmoid : ActivationTypes.Identity },
            OutputSettings = new OutputLayerSettings { NeuronsCount = 9, ActivationType = ActivationTypes.Sigmoid, InitializationType = WeightInitType.Xavier }
        },
        new RecurrentLayerSettings
        {
            NeuronsCount = 45, LayersCount = 2, CellType = NeuronType.Normal, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
            ActivationType = ActivationTypes.RELU, HasBiasNeuron = true, InitializationType = WeightInitType.Xavier
        });

        [Theory]
        [InlineData(WeightPrecision.Single, false, false, 0.0)]
        [InlineData(WeightPrecision.Half, true, false, 0.0)]
        [InlineData(WeightPrecision.Int8, false, true, 0.0)]
        [InlineData(WeightPrecision.Single, true, true, 0.8)]
        [InlineData(WeightPrecision.Int8, false, false, 0.9)]
        public void LayeredBatch_GivesTheOutputsOfOneSampleAtATime(WeightPrecision precision, bool normalize, bool softMax, double prune)
        {
            ICompactNetwork network = NetworkCompactor.Compact(Layered(normalize, softMax), new CompactOptions { Precision = precision, PruneFraction = prune });

            // more than one block of samples, and a block that is not full
            AssertBatchMatchesSingle(network, Samples(70, 77, 1));
            AssertBatchMatchesSingle(network, Samples(70, 1, 2));
            Assert.Empty(network.PropagateBatch(Array.Empty<float[]>()));
        }

        [Fact]
        public void GraphBatch_GivesTheOutputsOfOneSampleAtATime()
        {
            var random = new Random(3);
            var settings = new GraphNetworkSettings { OutputApplySoftMax = true };
            for (uint i = 0; i < 3; i++) settings.Nodes.Add(new GraphNode(i, NodeType.INPUT, i == 1 ? ActivationTypes.ArcusTangens : null));
            settings.Nodes.Add(new GraphNode(3, NodeType.BIAS));
            for (uint i = 0; i < 25; i++) settings.Nodes.Add(new GraphNode(10 + i, NodeType.HIDDEN, i % 2 == 0 ? ActivationTypes.RELU : ActivationTypes.Sigmoid));
            for (uint i = 0; i < 2; i++) settings.Nodes.Add(new GraphNode(50 + i, NodeType.OUTPUT));
            List<uint> order = settings.Nodes.Select(x => x.Id).ToList();
            for (uint c = 0; c < 90; c++)
            {
                int from = random.Next(order.Count - 2), to = random.Next(Math.Max(from + 1, 4), order.Count);
                settings.Connections.Add(new GraphConnection(c, order[from], order[to], (float)((random.NextDouble() * 2) - 1)));
            }

            foreach (WeightPrecision precision in Enum.GetValues<WeightPrecision>())
            {
                AssertBatchMatchesSingle(NetworkCompactor.CompactGraph(settings, new CompactOptions { Precision = precision }), Samples(3, 40, 4));
            }
        }

        [Theory]
        [InlineData(WeightPrecision.Single)]
        [InlineData(WeightPrecision.Int8)]
        public void ConvolutionalBatch_AndParallelConvolutions_GiveTheOutputsOfOneSampleAtATime(WeightPrecision precision)
        {
            // the first convolution is large enough to spread over several threads
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputChannels = 3, InputHeight = 40, InputWidth = 40, OutputDimension = 6, OutputApplySoftMax = true, Seed = 5,
                Layers =
                {
                    new ConvolutionLayerSettings { Filters = 32, KernelSize = 5, Padding = 2, UseBias = false, ActivationType = ActivationTypes.Identity },
                    new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.RELU },
                    new PoolingLayerSettings { Type = PoolingType.Max, Size = 2 },
                    new ResidualBlockSettings { Filters = 16, Stride = 2 },
                    new GlobalAveragePoolingLayerSettings()
                }
            });
            network.Training = false;
            ICompactNetwork compact = NetworkCompactor.Compact(network, new CompactOptions { Precision = precision });

            AssertBatchMatchesSingle(compact, Samples(3 * 40 * 40, 5, 6));

            // several callers at once
            float[][] samples = Samples(3 * 40 * 40, 8, 7);
            float[][] expected = samples.Select(x => compact.Propagate(x.Select(v => (Quad)v).ToArray()).Select(v => (float)v).ToArray()).ToArray();
            var actual = new float[samples.Length][];
            Parallel.For(0, samples.Length, i => actual[i] = compact.PropagateBatch(new[] { samples[i] })[0]);
            for (int i = 0; i < samples.Length; i++) Assert.Equal(Bits(expected[i]), Bits(actual[i]));
        }

        [Fact]
        public void PropagateBatch_RejectsSamplesOfTheWrongSize()
        {
            ICompactNetwork network = NetworkCompactor.Compact(Layered(false, false));
            Assert.Throws<ArgumentException>(() => network.PropagateBatch(new[] { new float[70], new float[69] }));
            Assert.Throws<ArgumentException>(() => network.Propagate(new float[70], new float[8]));
        }
    }
}
