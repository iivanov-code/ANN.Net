using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.IdentityGenerators;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// Matrix networks compute and train exactly like the neuron-graph networks of the same settings and seed
    /// </summary>
    public class MatrixNetworkTests
    {
        public sealed class Case
        {
            public OptimizerTypes Optimizer { get; init; } = OptimizerTypes.Normal;
            public bool Normalize { get; init; }
            public bool SoftMax { get; init; }
            public bool InputBias { get; init; } = true;
            public bool HiddenBias { get; init; } = true;
            public int HiddenLayers { get; init; } = 2;
            public WeightInitType Initialization { get; init; } = WeightInitType.Xavier;
            public ActivationTypes InputActivation { get; init; } = ActivationTypes.Identity;
            public LossFunctionTypes Loss { get; init; } = LossFunctionTypes.Squared;

            public override string ToString() =>
                $"{Optimizer} normalize={Normalize} softmax={SoftMax} bias={InputBias}/{HiddenBias} layers={HiddenLayers} {Initialization} {InputActivation} {Loss}";
        }

        public static TheoryData<Case> Cases() => new TheoryData<Case>
        {
            new Case(),
            new Case { Optimizer = OptimizerTypes.Adam },
            new Case { Optimizer = OptimizerTypes.Nesterov, Normalize = true },
            new Case { Optimizer = OptimizerTypes.AmsGrad, SoftMax = true, InputActivation = ActivationTypes.Sigmoid },
            new Case { Optimizer = OptimizerTypes.RMSprop, InputBias = false, HiddenBias = false, Initialization = WeightInitType.Random },
            new Case { Optimizer = OptimizerTypes.Nadam, HiddenLayers = 0, Loss = LossFunctionTypes.Absolute },
            new Case { Optimizer = OptimizerTypes.Momentum, HiddenBias = false, SoftMax = true, Normalize = true },
            new Case { Optimizer = OptimizerTypes.AdaMax, InputBias = false, HiddenLayers = 1 },
        };

        private static NetworkSettings Settings(Case c, int seed = 7) => new NetworkSettings
        {
            Seed = seed,
            IdentityGenerator = new IncrementalGenerator(),
            LossType = c.Loss,
            ShouldNormalize = c.Normalize,
            MinMaxValues = c.Normalize
                ? new List<Tuple<Quad, Quad>> { Tuple.Create<Quad, Quad>(-3, 3), Tuple.Create<Quad, Quad>(-4, 2), Tuple.Create<Quad, Quad>(-1, 5), Tuple.Create<Quad, Quad>(0, 1) }
                : null,
            OutputApplySoftMax = c.SoftMax,
            InputSettings = new InputLayerSettings { NeuronsCount = 4, ActivationType = c.InputActivation, HasBiasNeuron = c.InputBias },
            OutputSettings = new OutputLayerSettings
            {
                NeuronsCount = 3, ActivationType = ActivationTypes.Sigmoid, OptimizerType = c.Optimizer, InitializationType = c.Initialization
            }
        };

        private static RecurrentLayerSettings[] Hidden(Case c)
        {
            if (c.HiddenLayers == 0) return Array.Empty<RecurrentLayerSettings>();

            return new[]
            {
                new RecurrentLayerSettings
                {
                    NeuronsCount = 6,
                    LayersCount = (ushort)c.HiddenLayers,
                    CellType = NeuronType.Normal,
                    PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                    ActivationType = ActivationTypes.HyperbolicTangens,
                    HasBiasNeuron = c.HiddenBias,
                    OptimizerType = c.Optimizer,
                    InitializationType = c.Initialization
                }
            };
        }

        private static INetwork Graph(Case c, int seed = 7)
        {
            RecurrentLayerSettings[] hidden = Hidden(c);
            return NetworkFactory.BuildFFN(Settings(c, seed), hidden.Length == 0 ? null : hidden[0]);
        }

        private static void SetLearningRate(IEnumerable<LearningRateOptimizer> optimizers, float rate)
        {
            foreach (LearningRateOptimizer optimizer in optimizers) optimizer.LearningRate = rate;
        }

        private static Quad[][] Samples(int count, int seed, int width)
        {
            var random = new Random(seed);
            return Enumerable.Range(0, count).Select(_ => Enumerable.Range(0, width).Select(_ => (Quad)((random.NextDouble() * 4) - 2)).ToArray()).ToArray();
        }

        private static int[] Bits(IEnumerable<Quad> values) => values.Select(x => BitConverter.SingleToInt32Bits(x)).ToArray();

        private static void TrainTogether(INetwork expected, INetwork actual, int steps)
        {
            Quad[][] inputs = Samples(5, 3, 4);
            Quad[][] targets = Samples(5, 4, 3).Select(x => x.Select(v => (Quad)((v / 4) + 0.5f)).ToArray()).ToArray();

            for (int step = 0; step < steps; step++)
            {
                Quad[] input = inputs[step % inputs.Length];
                Assert.Equal(Bits(expected.Propagate(input)), Bits(actual.Propagate(input)));

                Quad[] target = targets[step % targets.Length];
                Assert.Equal(BitConverter.SingleToInt32Bits(expected.Backpropagate(target)), BitConverter.SingleToInt32Bits(actual.Backpropagate(target)));
                Assert.Equal(Bits(expected.Errors), Bits(actual.Errors));
                Assert.Equal(Bits(expected.InputGradients), Bits(actual.InputGradients));
            }

            foreach (Quad[] input in inputs)
            {
                Assert.Equal(Bits(expected.Propagate(input)), Bits(actual.Propagate(input)));
            }
        }

        [Theory]
        [MemberData(nameof(Cases))]
        public void BuildMatrixFFN_TrainsBitIdenticallyToBuildFFN(Case c)
        {
            INetwork graph = Graph(c);
            IMatrixNetwork matrix = NetworkFactory.BuildMatrixFFN(Settings(c), Hidden(c));
            SetLearningRate(((Network)graph).Optimizers, 0.05f);
            SetLearningRate(matrix.Optimizers, 0.05f);

            TrainTogether(graph, matrix, 40);
        }

        [Theory]
        [MemberData(nameof(Cases))]
        public void ToMatrix_CopiesTheWeightsOfATrainedNetwork(Case c)
        {
            INetwork graph = Graph(c);
            SetLearningRate(((Network)graph).Optimizers, 0.05f);
            Quad[][] inputs = Samples(4, 5, 4);
            foreach (Quad[] input in inputs)
            {
                graph.Propagate(input);
                graph.Backpropagate(0.2f, 0.5f, 0.7f);
            }

            IMatrixNetwork matrix = NetworkFactory.ToMatrix(graph);
            Assert.All(matrix.Optimizers, x => Assert.Equal(0.05f, (float)x.LearningRate));

            foreach (Quad[] input in inputs)
            {
                Assert.Equal(Bits(graph.Propagate(input)), Bits(matrix.Propagate(input)));
            }

            // optimizers without state continue identically
            if (c.Optimizer == OptimizerTypes.Normal) TrainTogether(graph, matrix, 20);
        }

        [Fact]
        public void ParameterCount_CountsWeightsAndBiases()
        {
            IMatrixNetwork matrix = NetworkFactory.BuildMatrixFFN(Settings(new Case()), Hidden(new Case()));

            Assert.Equal((5 * 6) + (7 * 6) + (7 * 3), matrix.ParameterCount);
            Assert.Equal(6, matrix.Optimizers.Count);
        }

        [Theory]
        [MemberData(nameof(Cases))]
        public void SaveAndLoad_GiveTheDocumentOfTheNeuronGraphNetwork(Case c)
        {
            INetwork graph = Graph(c);
            IMatrixNetwork matrix = NetworkFactory.BuildMatrixFFN(Settings(c), Hidden(c));
            SetLearningRate(((Network)graph).Optimizers, 0.03f);
            SetLearningRate(matrix.Optimizers, 0.03f);

            // same weights, layers and optimizers: the graph network loads from the matrix network's file
            using var stream = new MemoryStream();
            NetworkSerializer.Save(matrix, stream);
            stream.Position = 0;
            INetwork loaded = NetworkSerializer.Load(stream);

            IMatrixNetwork loadedMatrix = Assert.IsAssignableFrom<IMatrixNetwork>(loaded);
            Assert.All(loadedMatrix.Optimizers, x => Assert.Equal(0.03f, (float)x.LearningRate));

            Serialization.NetworkDocument fromGraph = NetworkSerializer.ToDocument(graph);
            Serialization.NetworkDocument fromMatrix = NetworkSerializer.ToDocument(matrix);
            Assert.Equal(fromGraph.Neurons.Select(x => (x.Type, x.Activation, x.ApplyActivation, string.Join(",", x.Inputs ?? new List<int>()))),
                         fromMatrix.Neurons.Select(x => (x.Type, x.Activation, x.ApplyActivation, string.Join(",", x.Inputs ?? new List<int>()))));
            Assert.Equal(fromGraph.Synapses.Select(x => (x.From, x.To, BitConverter.SingleToInt32Bits(x.Weight), x.Optimizer, x.LearningRate, x.Recurrent)),
                         fromMatrix.Synapses.Select(x => (x.From, x.To, BitConverter.SingleToInt32Bits(x.Weight), x.Optimizer, x.LearningRate, x.Recurrent)));
            Assert.Equal(fromGraph.InputLayer, fromMatrix.InputLayer);
            Assert.Equal(fromGraph.HiddenLayers, fromMatrix.HiddenLayers);
            Assert.Equal(fromGraph.OutputLayer, fromMatrix.OutputLayer);

            TrainTogether(graph, loaded, 10);
        }

        [Fact]
        public void OnnxExportAndCompact_MatchPropagate()
        {
            var c = new Case { Optimizer = OptimizerTypes.Adam, Normalize = true, SoftMax = true, InputActivation = ActivationTypes.Sigmoid };
            IMatrixNetwork matrix = NetworkFactory.BuildMatrixFFN(Settings(c), Hidden(c));
            SetLearningRate(matrix.Optimizers, 0.05f);
            Quad[][] inputs = Samples(6, 8, 4);
            foreach (Quad[] input in inputs)
            {
                matrix.Propagate(input);
                matrix.Backpropagate(0.2f, 0.5f, 0.3f);
            }

            float[][] onnx = OnnxExportTests.Run(OnnxExporter.Export(matrix), inputs.Select(x => x.Select(v => (float)v).ToArray()).ToArray());
            ICompactNetwork compact = NetworkCompactor.Compact(matrix);
            Assert.Equal(matrix.ParameterCount + (2 * 4), compact.ParameterCount); // plus the minimum and range of every input

            for (int i = 0; i < inputs.Length; i++)
            {
                float[] expected = matrix.Propagate(inputs[i]).Select(x => (float)x).ToArray();
                float[] fromCompact = compact.Propagate(inputs[i]).Select(x => (float)x).ToArray();
                for (int j = 0; j < expected.Length; j++)
                {
                    Assert.Equal(expected[j], onnx[i][j], 1e-5f);
                    Assert.Equal(expected[j], fromCompact[j], 1e-5f);
                }
            }
        }

        [Fact]
        public void BuildMatrixFFN_RejectsRecurrentAndUnconnectedLayers()
        {
            var lstm = Hidden(new Case());
            lstm[0].CellType = NeuronType.LSTM;
            Assert.Throws<NotSupportedException>(() => NetworkFactory.BuildMatrixFFN(Settings(new Case()), lstm));

            var straight = Hidden(new Case());
            straight[0].PrevLayerConnectionType = LayerConnectionType.Straight;
            Assert.Throws<NotSupportedException>(() => NetworkFactory.BuildMatrixFFN(Settings(new Case()), straight));
        }
    }
}
