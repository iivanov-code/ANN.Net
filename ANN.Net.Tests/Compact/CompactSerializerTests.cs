using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using Xunit;

namespace ANN.Net.Tests.Compact
{
    public class CompactSerializerTests
    {
        private static float[][] Samples(int width, int count, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, count).Select(_ => Enumerable.Range(0, width).Select(_ => (float)((random.NextDouble() * 4) - 2)).ToArray()).ToArray();
        }

        private static int[] Bits(IEnumerable<float> values) => values.Select(BitConverter.SingleToInt32Bits).ToArray();

        private static byte[] Save(ICompactNetwork network)
        {
            using var stream = new MemoryStream();
            CompactNetworkSerializer.Save(network, stream);
            return stream.ToArray();
        }

        private static ICompactNetwork RoundTrip(ICompactNetwork network, int seed)
        {
            byte[] file = Save(network);
            ICompactNetwork loaded = CompactNetworkSerializer.Load(new MemoryStream(file));

            Assert.Equal(network.GetType(), loaded.GetType());
            Assert.Equal(network.InputCount, loaded.InputCount);
            Assert.Equal(network.OutputCount, loaded.OutputCount);
            Assert.Equal(network.ParameterCount, loaded.ParameterCount);
            Assert.Equal(network.SizeInBytes, loaded.SizeInBytes);
            Assert.Equal(network.Precision, loaded.Precision);
            Assert.InRange(file.Length, network.SizeInBytes * 0.9, (network.SizeInBytes * 1.1) + 1024);

            float[][] samples = Samples(network.InputCount, 6, seed);
            float[][] expected = network.PropagateBatch(samples), actual = loaded.PropagateBatch(samples);
            for (int i = 0; i < samples.Length; i++) Assert.Equal(Bits(expected[i]), Bits(actual[i]));
            return loaded;
        }

        private static INetwork Layered() => NetworkFactory.BuildMatrixFFN(new NetworkSettings
        {
            Seed = 4,
            IdentityGenerator = new IncrementalGenerator(),
            ShouldNormalize = true,
            MinMaxValues = Enumerable.Range(0, 30).Select(i => Tuple.Create<Quad, Quad>(-3, 2 + i)).ToList(),
            OutputApplySoftMax = true,
            InputSettings = new InputLayerSettings { NeuronsCount = 30, HasBiasNeuron = true, ActivationType = ActivationTypes.SoftSign },
            OutputSettings = new OutputLayerSettings { NeuronsCount = 5, ActivationType = ActivationTypes.Sigmoid, InitializationType = WeightInitType.Xavier }
        },
        new RecurrentLayerSettings
        {
            NeuronsCount = 40, LayersCount = 2, CellType = NeuronType.Normal, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
            ActivationType = ActivationTypes.RELU, HasBiasNeuron = true, InitializationType = WeightInitType.Xavier
        });

        private static GraphNetworkSettings Graph()
        {
            var random = new Random(5);
            var settings = new GraphNetworkSettings();
            for (uint i = 0; i < 3; i++) settings.Nodes.Add(new GraphNode(i, NodeType.INPUT, i == 2 ? ActivationTypes.HyperbolicTangens : null));
            settings.Nodes.Add(new GraphNode(3, NodeType.BIAS));
            for (uint i = 0; i < 15; i++) settings.Nodes.Add(new GraphNode(10 + i, NodeType.HIDDEN, i % 2 == 0 ? ActivationTypes.GELU : ActivationTypes.RELU));
            for (uint i = 0; i < 3; i++) settings.Nodes.Add(new GraphNode(40 + i, NodeType.OUTPUT));
            List<uint> order = settings.Nodes.Select(x => x.Id).ToList();
            for (uint c = 0; c < 50; c++)
            {
                int from = random.Next(order.Count - 3), to = random.Next(Math.Max(from + 1, 4), order.Count);
                settings.Connections.Add(new GraphConnection(c, order[from], order[to], (float)((random.NextDouble() * 2) - 1)));
            }

            return settings;
        }

        private static IConvolutionalNetwork Convolutional()
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputChannels = 2, InputHeight = 14, InputWidth = 12, OutputDimension = 4, OutputApplySoftMax = true, Seed = 6,
                Layers =
                {
                    new ConvolutionLayerSettings { Filters = 6, KernelHeight = 3, KernelWidth = 5, Padding = 1, ActivationType = ActivationTypes.RELU },
                    new BatchNormalizationLayerSettings { ActivationType = ActivationTypes.HyperbolicTangens }, // after an activation: not folded
                    new PoolingLayerSettings { Type = PoolingType.Average, Size = 2, Stride = 2, Padding = 1 },
                    new ResidualBlockSettings { Filters = 8, Stride = 2 },
                    new ResidualBlockSettings { Filters = 8, Stride = 1 },
                    new GlobalAveragePoolingLayerSettings(),
                    new DenseLayerSettings { NeuronsCount = 7, ActivationType = ActivationTypes.Sigmoid }
                }
            });
            network.Training = false;
            return network;
        }

        public static TheoryData<WeightPrecision, double> Formats() => new TheoryData<WeightPrecision, double>
        {
            { WeightPrecision.Single, 0 }, { WeightPrecision.Half, 0 }, { WeightPrecision.Int8, 0 },
            { WeightPrecision.Single, 0.8 }, { WeightPrecision.Half, 0.9 }, { WeightPrecision.Int8, 0.95 }
        };

        [Theory]
        [MemberData(nameof(Formats))]
        public void Layered_RoundTripsBitForBit(WeightPrecision precision, double prune)
        {
            RoundTrip(NetworkCompactor.Compact(Layered(), new CompactOptions { Precision = precision, PruneFraction = prune }), 1);
        }

        [Theory]
        [MemberData(nameof(Formats))]
        public void Graph_RoundTripsBitForBit(WeightPrecision precision, double prune)
        {
            RoundTrip(NetworkCompactor.CompactGraph(Graph(), new CompactOptions { Precision = precision, PruneFraction = prune }), 2);
        }

        [Theory]
        [MemberData(nameof(Formats))]
        public void Convolutional_RoundTripsBitForBit(WeightPrecision precision, double prune)
        {
            RoundTrip(NetworkCompactor.Compact(Convolutional(), new CompactOptions { Precision = precision, PruneFraction = prune }), 3);
        }

        [Fact]
        public void SaveToFile_AndLoad()
        {
            ICompactNetwork network = NetworkCompactor.Compact(Layered(), new CompactOptions { Precision = WeightPrecision.Int8 });
            string file = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N") + "." + CompactNetworkSerializer.FileExtension);
            try
            {
                CompactNetworkSerializer.Save(network, file);
                ICompactNetwork loaded = CompactNetworkSerializer.Load(file);
                float[] sample = Samples(30, 1, 7)[0];
                Assert.Equal(Bits(network.PropagateBatch(new[] { sample })[0]), Bits(loaded.PropagateBatch(new[] { sample })[0]));
            }
            finally
            {
                File.Delete(file);
            }
        }

        public static TheoryData<int> Networks() => new TheoryData<int> { 0, 1, 2 };

        /// <summary>
        /// A damaged file is rejected with InvalidDataException, or loads into a network that still runs
        /// </summary>
        [Theory]
        [MemberData(nameof(Networks))]
        public void DamagedFiles_AreRejectedOrStillRun(int kind)
        {
            ICompactNetwork network = kind switch
            {
                0 => NetworkCompactor.Compact(Layered(), new CompactOptions { Precision = WeightPrecision.Int8, PruneFraction = 0.9 }),
                1 => NetworkCompactor.CompactGraph(Graph()),
                _ => NetworkCompactor.Compact(Convolutional(), new CompactOptions { Precision = WeightPrecision.Half })
            };
            byte[] file = Save(network);
            float[] sample = Samples(network.InputCount, 1, 8)[0];

            void Check(byte[] damaged, string what)
            {
                ICompactNetwork loaded;
                try
                {
                    loaded = CompactNetworkSerializer.Load(new MemoryStream(damaged));
                }
                catch (InvalidDataException)
                {
                    return;
                }
                catch (Exception exception)
                {
                    Assert.Fail($"{what}: {exception.GetType().Name} {exception.Message}");
                    return;
                }

                try
                {
                    loaded.PropagateBatch(new[] { new float[loaded.InputCount] });
                    if (loaded.InputCount == sample.Length) loaded.PropagateBatch(new[] { sample });
                }
                catch (Exception exception)
                {
                    Assert.Fail($"{what} loaded but does not run: {exception.GetType().Name} {exception.Message}");
                }
            }

            for (int length = 0; length < file.Length; length += Math.Max(1, file.Length / 300))
            {
                Check(file.Take(length).ToArray(), $"cut at {length}");
            }

            var random = new Random(9);
            for (int i = 0; i < 400; i++)
            {
                byte[] damaged = (byte[])file.Clone();
                int position = random.Next(damaged.Length);
                damaged[position] = (byte)random.Next(256);
                Check(damaged, $"byte {position} = {damaged[position]}");
            }
        }
    }
}
