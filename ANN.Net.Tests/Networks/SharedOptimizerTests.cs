using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.IdentityGenerators;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Plain gradient descent keeps no state, so the synapses of a layer share one optimizer
    /// </summary>
    public class SharedOptimizerTests
    {
        private static Network Build(OptimizerTypes type) => (Network)NetworkFactory.BuildFFN(new NetworkSettings
        {
            IdentityGenerator = new IncrementalGenerator(),
            Seed = 1,
            InputSettings = new InputLayerSettings { NeuronsCount = 5, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true, OptimizerType = type },
            OutputSettings = new OutputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity, OptimizerType = type }
        },
        new RecurrentLayerSettings { NeuronsCount = 4, LayersCount = 2, CellType = NeuronType.Normal, ActivationType = ActivationTypes.HyperbolicTangens, HasBiasNeuron = true, OptimizerType = type });

        private static int Synapses(Network network) => network.InputNeurons.Cast<INeuron>()
            .Concat(network.HiddenLayers.SelectMany(x => x.Cast<INeuron>()))
            .Sum(x => x.Outputs.Count());

        [Fact]
        public void Normal_SharesOneOptimizerPerLayer()
        {
            Network network = Build(OptimizerTypes.Normal);

            Assert.Equal(3, network.Optimizers.Count);
            Assert.Equal(3, network.InputNeurons.Cast<INeuron>().SelectMany(x => x.Outputs).Cast<BaseSynapse>().Select(x => x.Optimizer)
                                   .Concat(network.HiddenLayers.SelectMany(x => x).SelectMany(x => x.Outputs).Cast<BaseSynapse>().Select(x => x.Optimizer))
                                   .Distinct().Count());
        }

        [Fact]
        public void OptimizersWithState_KeepOnePerSynapse()
        {
            Network network = Build(OptimizerTypes.Adam);

            Assert.Equal(Synapses(network), network.Optimizers.Count);
        }

        [Fact]
        public void LoadedNetwork_SharesNormalOptimizersWithTheSameLearningRate()
        {
            Network original = Build(OptimizerTypes.Normal);
            foreach (LearningRateOptimizer optimizer in original.Optimizers) optimizer.LearningRate = 0.05f;

            using var stream = new MemoryStream();
            NetworkSerializer.Save(original, stream);
            stream.Position = 0;
            var loaded = (Network)NetworkSerializer.Load(stream);

            LearningRateOptimizer shared = Assert.Single(loaded.Optimizers);
            Assert.Equal(0.05f, (float)shared.LearningRate);

            Quad[] input = { 0.1f, -0.4f, 0.8f, 0.3f, -1f };
            for (int i = 0; i < 10; i++)
            {
                original.Propagate(input);
                original.Backpropagate(new Quad[] { 0.5f, -0.5f });
                loaded.Propagate(input);
                loaded.Backpropagate(new Quad[] { 0.5f, -0.5f });
            }

            Assert.Equal(original.Propagate(input).Select(x => (float)x), loaded.Propagate(input).Select(x => (float)x));
        }
    }
}
