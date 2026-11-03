using System;
using System.IO;
using System.Linq;
using System.Text;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Tests.Networks;
using Xunit;

namespace ANN.Net.Tests.Convolutional
{
    /// <summary>
    /// Saving, loading and ONNX export of convolutional networks
    /// </summary>
    public class ConvolutionalPersistenceTests
    {
        private static ConvolutionalNetworkSettings Settings(bool softMax) => new ConvolutionalNetworkSettings
        {
            InputChannels = 2,
            InputHeight = 9,
            InputWidth = 8,
            Layers =
            {
                new ConvolutionLayerSettings { Filters = 4, KernelSize = 3, Padding = 1, ActivationType = ActivationTypes.RELU },
                new PoolingLayerSettings { Type = PoolingType.Max, Size = 2 },
                new ConvolutionLayerSettings { Filters = 3, KernelHeight = 2, KernelWidth = 3, Stride = 2, Padding = 1, ActivationType = ActivationTypes.HyperbolicTangens },
                new PoolingLayerSettings { Type = PoolingType.Average, Size = 2, Stride = 1 },
                new DenseLayerSettings { NeuronsCount = 6, ActivationType = ActivationTypes.Sigmoid }
            },
            OutputDimension = 3,
            OutputApplySoftMax = softMax,
            OutputActivationType = ActivationTypes.SoftSign
        };

        private static Quad[][] Images(int count, int seed)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, count).Select(_ => Enumerable.Range(0, 2 * 9 * 8).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray()).ToArray();
        }

        private static void Train(INetwork network, Quad[][] images, int steps)
        {
            for (int i = 0; i < steps; i++)
            {
                network.Propagate(images[i % images.Length]);
                network.Backpropagate(new Quad[] { 0.1f, 0.8f, 0.1f });
            }
        }

        private static void AssertSameOutputs(INetwork expected, INetwork actual, Quad[][] images)
        {
            foreach (Quad[] image in images)
            {
                Assert.Equal(expected.Propagate(image).Select(x => (float)x), actual.Propagate(image).Select(x => (float)x));
            }
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void RoundTrip_GivesTheSameOutputsAndKeepsTraining(bool softMax)
        {
            IConvolutionalNetwork original = NetworkFactory.BuildConvolutional(Settings(softMax));
            foreach (LearningRateOptimizer optimizer in original.Optimizers) optimizer.LearningRate = 0.05f;
            Quad[][] images = Images(6, 1);
            Train(original, images, 20);

            using var stream = new MemoryStream();
            NetworkSerializer.Save(original, stream);
            stream.Position = 0;
            IConvolutionalNetwork loaded = NetworkSerializer.Load<IConvolutionalNetwork>(stream);

            AssertSameOutputs(original, loaded, images);
            Assert.All(loaded.Optimizers, x => Assert.Equal(0.05f, (float)x.LearningRate));

            Train(original, images, 10);
            Train(loaded, images, 10);
            AssertSameOutputs(original, loaded, images);
        }

        [Fact]
        public void RoundTrip_KeepsDifferentLearningRates()
        {
            IConvolutionalNetwork original = NetworkFactory.BuildConvolutional(Settings(false));
            float[] rates = original.Optimizers.Select((_, i) => (i % 7) / 10f).ToArray();
            int index = 0;
            foreach (LearningRateOptimizer optimizer in original.Optimizers) optimizer.LearningRate = rates[index++];

            using var stream = new MemoryStream();
            NetworkSerializer.Save(original, stream);
            stream.Position = 0;
            IConvolutionalNetwork loaded = NetworkSerializer.Load<IConvolutionalNetwork>(stream);

            Assert.Equal(rates, loaded.Optimizers.Select(x => (float)x.LearningRate));
        }

        [Fact]
        public void SaveToFile_LoadsThroughANNetwork()
        {
            IConvolutionalNetwork original = NetworkFactory.BuildConvolutional(Settings(true));
            string directory = Path.Combine(Path.GetTempPath(), Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(directory);

            try
            {
                original.Save(directory, "cnn");
                INetwork loaded = ANNetwork.Load(Path.Combine(directory, "cnn.ann"));

                Assert.IsAssignableFrom<IConvolutionalNetwork>(loaded);
                AssertSameOutputs(original, loaded, Images(3, 2));
            }
            finally
            {
                Directory.Delete(directory, recursive: true);
            }
        }

        [Fact]
        public void Load_RejectsDamagedLayers()
        {
            using var stream = new MemoryStream();
            NetworkSerializer.Save(NetworkFactory.BuildConvolutional(Settings(false)), stream);
            string json = Encoding.UTF8.GetString(stream.ToArray());

            string withoutValues = json.Replace("\"values\": [", "\"values\": [ 1.5, ", StringComparison.Ordinal);
            Assert.Throws<InvalidDataException>(() => NetworkSerializer.Load(new MemoryStream(Encoding.UTF8.GetBytes(withoutValues))));
        }

        [Theory]
        [InlineData(false)]
        [InlineData(true)]
        public void OnnxExport_MatchesPropagate(bool softMax)
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(Settings(softMax));
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.05f;
            Quad[][] images = Images(5, 3);
            Train(network, images, 15);

            float[][] samples = images.Select(x => x.Select(v => (float)v).ToArray()).ToArray();
            float[][] actual = OnnxExportTests.Run(OnnxExporter.Export(network), samples);

            for (int i = 0; i < samples.Length; i++)
            {
                float[] expected = network.Propagate(images[i]).Select(x => (float)x).ToArray();
                Assert.Equal(expected.Length, actual[i].Length);
                for (int j = 0; j < expected.Length; j++)
                {
                    Assert.Equal(expected[j], actual[i][j], 4);
                }
            }
        }
    }
}
