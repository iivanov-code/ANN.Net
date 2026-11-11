using System;
using System.Linq;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Seeded initial weights make networks reproducible
    /// </summary>
    public class SeedTests
    {
        private static readonly Quad[] Input = Enumerable.Range(0, 16).Select(i => (Quad)((i % 5) - 2)).ToArray();

        private static float[] Layered(int? seed)
        {
            INetwork network = NetworkFactory.BuildFFN(new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                Seed = seed,
                InputSettings = new InputLayerSettings { NeuronsCount = 16, ActivationType = ActivationTypes.Identity },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.Identity }
            },
            new RecurrentLayerSettings { NeuronsCount = 6, LayersCount = 2, CellType = NeuronType.Normal, ActivationType = ActivationTypes.HyperbolicTangens });
            return network.Propagate(Input).Select(x => (float)x).ToArray();
        }

        private static float[] Convolutional(int? seed)
        {
            IConvolutionalNetwork network = NetworkFactory.BuildConvolutional(new ConvolutionalNetworkSettings
            {
                InputHeight = 4,
                InputWidth = 4,
                Seed = seed,
                Layers = { new ConvolutionLayerSettings { Filters = 3, KernelSize = 3, Padding = 1 } },
                OutputDimension = 3
            });
            return network.Propagate(Input).Select(x => (float)x).ToArray();
        }

        private static float[] Transformer(int? seed)
        {
            ISequenceNetwork network = NetworkFactory.BuildTransformer(new TransformerSettings
            {
                InputDimension = 4,
                ModelDimension = 8,
                HeadsCount = 2,
                FeedForwardDimension = 8,
                LayersCount = 1,
                OutputDimension = 3,
                MaxSequenceLength = 4,
                Seed = seed
            });
            Quad[][] sequence = Enumerable.Range(0, 4).Select(t => Input.Skip(t * 4).Take(4).ToArray()).ToArray();
            return network.Propagate(sequence).SelectMany(x => x).Select(x => (float)x).ToArray();
        }

        public static TheoryData<string> Networks() => new TheoryData<string> { "layered", "convolutional", "transformer" };

        private static Func<int?, float[]> Builder(string name) => name switch
        {
            "layered" => Layered,
            "convolutional" => Convolutional,
            _ => Transformer
        };

        [Theory]
        [MemberData(nameof(Networks))]
        public void SameSeed_BuildsTheSameNetwork(string name)
        {
            Func<int?, float[]> build = Builder(name);

            Assert.Equal(build(7), build(7));
            Assert.NotEqual(build(7), build(8));
            Assert.NotEqual(build(null), build(null));
        }

        [Fact]
        public void SeededNetworksBuiltInParallel_DoNotInterfere()
        {
            float[][] sequential = Enumerable.Range(0, 8).Select(i => Transformer(i)).ToArray();

            float[][] parallel = Task.WhenAll(Enumerable.Range(0, 8).Select(i => Task.Run(() => Transformer(i)))).GetAwaiter().GetResult();

            for (int i = 0; i < 8; i++)
            {
                Assert.Equal(sequential[i], parallel[i]);
            }
        }

        [Fact]
        public void Seed_OnlyAppliesWhileTheNetworkIsBuilt()
        {
            Layered(3);

            // random numbers drawn afterwards come from the shared source again
            Assert.NotEqual(Layered(null), Layered(null));
        }
    }
}
