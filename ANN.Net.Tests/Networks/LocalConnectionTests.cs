using System;
using System.Collections.Generic;
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
    /// Convolutional and deconvolutional hidden layers of the neuron-graph network (1-D, shared kernel weights)
    /// </summary>
    public class LocalConnectionTests
    {
        internal static Network Build(ushort inputs, LayerConnectionType type, int kernel, int stride, int padding, int filters = 1,
                                     bool inputBias = false, ActivationTypes hiddenActivation = ActivationTypes.Identity,
                                     OptimizerTypes optimizer = OptimizerTypes.Normal, ushort outputs = 1)
        {
            return (Network)NetworkFactory.BuildFFN(new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                InputSettings = new InputLayerSettings { NeuronsCount = inputs, ActivationType = ActivationTypes.Identity, HasBiasNeuron = inputBias },
                OutputSettings = new OutputLayerSettings { NeuronsCount = outputs, ActivationType = ActivationTypes.Identity, OptimizerType = optimizer }
            },
            new RecurrentLayerSettings
            {
                LayersCount = 1,
                CellType = NeuronType.Normal,
                PrevLayerConnectionType = type,
                KernelSize = kernel,
                Stride = stride,
                Padding = padding,
                FiltersCount = filters,
                ActivationType = hiddenActivation,
                OptimizerType = optimizer
            });
        }

        private static List<INeuron> Inputs(Network network) => network.InputNeurons.Cast<INeuron>().ToList();

        private static List<IHiddenNeuron> Hidden(Network network) => network.HiddenLayers.Single().ToList();

        /// <summary>
        /// Input positions feeding every hidden neuron ("b" for the bias neuron)
        /// </summary>
        private static List<string> Sources(Network network)
        {
            List<INeuron> inputs = Inputs(network);
            return Hidden(network).Select(h => string.Join(",", h.Inputs.Cast<BaseSynapse>().Select(s => s.Input is Neurons.BiasNeuron ? "b" : inputs.IndexOf(s.Input).ToString(System.Globalization.CultureInfo.InvariantCulture)))).ToList();
        }

        [Fact]
        public void Convolution_ConnectsSlidingWindowsWithSharedKernels()
        {
            Network network = Build(6, LayerConnectionType.Convolutional, kernel: 3, stride: 1, padding: 0, filters: 2);

            Assert.Equal(new[] { "0,1,2", "1,2,3", "2,3,4", "3,4,5", "0,1,2", "1,2,3", "2,3,4", "3,4,5" }, Sources(network));

            // every position of a filter uses the same three weight objects, the second filter its own
            var kernels = Hidden(network).Select(h => h.Inputs.Cast<SharedWeightSynapse>().Select(s => s.SharedWeight).ToArray()).ToList();
            for (int o = 1; o < 4; o++) Assert.Equal(kernels[0], kernels[o]);
            Assert.Empty(kernels[0].Intersect(kernels[4]));
            Assert.All(kernels[0], x => Assert.Equal(4, x.UsersCount));
        }

        [Fact]
        public void Convolution_HonoursStrideAndPadding()
        {
            Network network = Build(6, LayerConnectionType.Convolutional, kernel: 3, stride: 2, padding: 1);

            // positions (6 + 2 - 3) / 2 + 1 = 3; the padded positions -1 and 6 add nothing
            Assert.Equal(new[] { "0,1", "1,2,3", "3,4,5" }, Sources(network));
        }

        [Fact]
        public void Convolution_FeedsTheBiasThroughOneSharedWeightPerFilter()
        {
            Network network = Build(5, LayerConnectionType.Convolutional, kernel: 2, stride: 1, padding: 0, filters: 2, inputBias: true);

            Assert.Equal(new[] { "0,1,b", "1,2,b", "2,3,b", "3,4,b", "0,1,b", "1,2,b", "2,3,b", "3,4,b" }, Sources(network));
            var biasWeights = Hidden(network).Select(h => h.Inputs.Cast<SharedWeightSynapse>().Last().SharedWeight).ToList();
            Assert.Single(biasWeights.Take(4).Distinct());
            Assert.Single(biasWeights.Skip(4).Distinct());
            Assert.NotSame(biasWeights[0], biasWeights[4]);
        }

        [Fact]
        public void Deconvolution_SpreadsEveryInputOverAKernelOfOutputs()
        {
            Network network = Build(3, LayerConnectionType.Deconvolutional, kernel: 3, stride: 2, padding: 0);

            // (3 - 1) * 2 + 3 = 7 outputs; input i reaches outputs 2i .. 2i + 2
            Assert.Equal(new[] { "0", "0", "0,1", "1", "1,2", "2", "2" }, Sources(network));

            Network padded = Build(3, LayerConnectionType.Deconvolutional, kernel: 3, stride: 1, padding: 1);
            Assert.Equal(new[] { "0,1", "0,1,2", "1,2" }, Sources(padded));
        }

        [Fact]
        public void Build_RejectsSettingsThatLeaveNeuronsWithoutInput()
        {
            Assert.Throws<ArgumentException>(() => Build(6, LayerConnectionType.Convolutional, kernel: 3, stride: 1, padding: 3));
            Assert.Throws<ArgumentException>(() => Build(2, LayerConnectionType.Convolutional, kernel: 5, stride: 1, padding: 1));
            Assert.Throws<ArgumentException>(() => Build(6, LayerConnectionType.Convolutional, kernel: 3, stride: 0, padding: 0));
            Assert.Throws<ArgumentException>(() => Build(4, LayerConnectionType.Deconvolutional, kernel: 2, stride: 3, padding: 0));
            Assert.Throws<ArgumentException>(() => Build(4, LayerConnectionType.Convolutional, kernel: 2, stride: 1, padding: 0, filters: 0));
        }

        [Fact]
        public void Convolution_ComputesTheConvolutionAndTheSummedKernelGradient()
        {
            Network network = Build(5, LayerConnectionType.Convolutional, kernel: 2, stride: 1, padding: 0);
            float[] x = { 0.5f, -1f, 2f, 0.25f, -0.75f };
            List<IHiddenNeuron> hidden = Hidden(network);
            SharedWeight[] kernel = hidden[0].Inputs.Cast<SharedWeightSynapse>().Select(s => s.SharedWeight).ToArray();
            float w0 = kernel[0].Value, w1 = kernel[1].Value;
            float[] v = hidden.Select(h => (float)h.Outputs.Cast<BaseSynapse>().Single().Weight).ToArray();

            float[] h = Enumerable.Range(0, 4).Select(o => (w0 * x[o]) + (w1 * x[o + 1])).ToArray();
            float y = h.Zip(v, (a, b) => a * b).Sum();
            Assert.Equal(y, (float)network.Propagate(x.Select(z => (Quad)z).ToArray())[0], 1e-4f);

            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 1;
            const float target = 0.3f;
            network.Backpropagate(new Quad[] { target });

            // dL/dw_k = e * Σ_o v_o * x[o + k], summed over the positions that share the weight
            float e = y - target;
            Assert.Equal(w0 - (e * Enumerable.Range(0, 4).Sum(o => v[o] * x[o])), (float)kernel[0].Value, 1e-4f);
            Assert.Equal(w1 - (e * Enumerable.Range(0, 4).Sum(o => v[o] * x[o + 1])), (float)kernel[1].Value, 1e-4f);
            Assert.All(hidden[2].Inputs.Cast<BaseSynapse>(), s => Assert.Equal((float)((SharedWeightSynapse)s).SharedWeight.Value, (float)s.Weight));

            // the input gradients are computed with the weights before the update
            float[] inputGradient = new float[5];
            for (int o = 0; o < 4; o++)
            {
                inputGradient[o] += e * v[o] * w0;
                inputGradient[o + 1] += e * v[o] * w1;
            }

            Assert.Equal(inputGradient, network.InputGradients.Select(g => (float)g).ToArray(), new FloatComparer(1e-4f));
        }

        private sealed class FloatComparer : IEqualityComparer<float>
        {
            private readonly float tolerance;

            public FloatComparer(float tolerance) => this.tolerance = tolerance;

            public bool Equals(float x, float y) => Math.Abs(x - y) <= tolerance;

            public int GetHashCode(float obj) => 0;
        }

        /// <summary>
        /// 10 values with a [1, -1] edge at a random place for label 1, or no edge for label 0
        /// </summary>
        internal static (Quad[] Signal, float Label) Edge(Random random)
        {
            var signal = Enumerable.Range(0, 10).Select(_ => (Quad)((random.NextDouble() - 0.5) * 0.2)).ToArray();
            float label = random.Next(2);
            if (label > 0)
            {
                int position = random.Next(9);
                signal[position] += 1;
                signal[position + 1] -= 1;
            }

            return (signal, label);
        }

        [Fact]
        public void Convolution_LearnsATranslationInvariantPattern()
        {
            Network network = Build(10, LayerConnectionType.Convolutional, kernel: 2, stride: 1, padding: 0, filters: 4,
                                    inputBias: true, hiddenActivation: ActivationTypes.RELU);

            // a shared kernel weight sums the gradients of all 9 positions, so it needs a smaller rate than a plain weight
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 0.005f;

            var random = new Random(3);
            for (int i = 0; i < 6000; i++)
            {
                var (signal, label) = Edge(random);
                network.Propagate(signal);
                network.Backpropagate(new Quad[] { label });
            }

            var test = new Random(4);
            int correct = Enumerable.Range(0, 200).Count(_ =>
            {
                var (signal, label) = Edge(test);
                return Math.Abs((float)network.Propagate(signal)[0] - label) < 0.5f;
            });

            Assert.True(correct >= 190, $"{correct} of 200 correct");
        }

        [Theory]
        [InlineData(LayerConnectionType.Convolutional)]
        [InlineData(LayerConnectionType.Deconvolutional)]
        public void SavedNetwork_KeepsItsWeightsShared(LayerConnectionType type)
        {
            Network original = Build(6, type, kernel: 3, stride: type == LayerConnectionType.Convolutional ? 1 : 2, padding: 1, filters: 2,
                                     inputBias: true, hiddenActivation: ActivationTypes.HyperbolicTangens, outputs: 2);
            foreach (LearningRateOptimizer optimizer in original.Optimizers) optimizer.LearningRate = 0.05f;

            using var stream = new MemoryStream();
            NetworkSerializer.Save(original, stream);
            stream.Position = 0;
            INetwork loaded = NetworkSerializer.Load(stream);

            var random = new Random(5);
            Quad[][] samples = Enumerable.Range(0, 4).Select(_ => Enumerable.Range(0, 6).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray()).ToArray();
            void Train(INetwork network)
            {
                for (int i = 0; i < 20; i++)
                {
                    network.Propagate(samples[i % 4]);
                    network.Backpropagate(0.2f, -0.4f);
                }
            }

            // a network whose weights were no longer shared would drift away from the original while training
            Train(original);
            Train(loaded);
            foreach (Quad[] sample in samples)
            {
                Assert.Equal(original.Propagate(sample).Select(x => (float)x), loaded.Propagate(sample).Select(x => (float)x));
            }

            float[][] actual = OnnxExportTests.Run(OnnxExporter.Export(original), samples.Select(x => x.Select(v => (float)v).ToArray()).ToArray());
            for (int i = 0; i < samples.Length; i++)
            {
                float[] expected = original.Propagate(samples[i]).Select(x => (float)x).ToArray();
                for (int j = 0; j < expected.Length; j++) Assert.Equal(expected[j], actual[i][j], 1e-4f * Math.Max(1, Math.Abs(expected[j])));
            }
        }
    }
}
