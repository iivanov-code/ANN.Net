using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Matrix;
using Xunit;

namespace ANN.Net.Tests.LossFunctions
{
    public class LossFunctionTests
    {
        private static ILossFunction Loss(LossFunctionTypes type, LossOptions options = null) => NetworkFactory.BuildLossFunction(type, options);

        private static List<ErrorResult> Results(params (double Target, double Predicted)[] values) =>
            values.Select(x => new ErrorResult { Target = (Quad)x.Target, Predicted = (Quad)x.Predicted }).ToList();

        private static readonly (double T, double P)[] Regression = { (1, 0.5), (2, 3), (0.5, 0.25), (-1, 1.5) };
        private static readonly (double T, double P)[] Probabilities = { (1, 0.9), (0, 0.2), (1, 0.6), (0, 0.45) };
        private static readonly (double T, double P)[] Distribution = { (0.7, 0.5), (0, 0.2), (0.3, 0.3) };
        private static readonly (double T, double P)[] Signs = { (1, 0.3), (-1, 0.5), (1, 2), (-1, -0.2) };
        private static readonly (double T, double P)[] Positive = { (1, 0.5), (2, 3), (0.5, 0.25), (4, 1.5) };

        private static double Mean((double T, double P)[] data, Func<double, double, double> term) => data.Average(x => term(x.T, x.P));

        private static double Sum((double T, double P)[] data, Func<double, double, double> term) => data.Sum(x => term(x.T, x.P));

        private static double Huber(double e, double delta) => Math.Abs(e) <= delta ? e * e / 2 : delta * (Math.Abs(e) - (delta / 2));

        /// <summary>
        /// Every loss against its textbook formula, written out here independently of the implementation
        /// </summary>
        public static TheoryData<LossFunctionTypes, string, double> Values => new TheoryData<LossFunctionTypes, string, double>
        {
            { LossFunctionTypes.Difference, nameof(Regression), Sum(Regression, (t, p) => p - t) },
            { LossFunctionTypes.Squared, nameof(Regression), Sum(Regression, (t, p) => (p - t) * (p - t) / 2) },
            { LossFunctionTypes.Absolute, nameof(Regression), Mean(Regression, (t, p) => Math.Abs(p - t)) },
            { LossFunctionTypes.MeanSquared, nameof(Regression), Mean(Regression, (t, p) => (p - t) * (p - t)) },
            { LossFunctionTypes.Huber, nameof(Regression), Mean(Regression, (t, p) => Huber(p - t, 1)) },
            { LossFunctionTypes.SmoothL1, nameof(Regression), Mean(Regression, (t, p) => Huber(p - t, 1)) },
            { LossFunctionTypes.PseudoHuber, nameof(Regression), Mean(Regression, (t, p) => Math.Sqrt(1 + ((p - t) * (p - t))) - 1) },
            { LossFunctionTypes.LogCosh, nameof(Regression), Mean(Regression, (t, p) => Math.Log(Math.Cosh(p - t))) },
            { LossFunctionTypes.MeanAbsolutePercentage, nameof(Regression), 100 * Mean(Regression, (t, p) => Math.Abs((p - t) / t)) },
            { LossFunctionTypes.MeanSquaredLogarithmic, nameof(Positive), Mean(Positive, (t, p) => Math.Pow(Math.Log(1 + p) - Math.Log(1 + t), 2)) },
            { LossFunctionTypes.Poisson, nameof(Positive), Mean(Positive, (t, p) => p - (t * Math.Log(p))) },
            { LossFunctionTypes.CrossEntropy, nameof(Distribution), Sum(Distribution, (t, p) => -t * Math.Log(p)) },
            { LossFunctionTypes.BinaryCrossEntropy, nameof(Probabilities), Mean(Probabilities, (t, p) => -((t * Math.Log(p)) + ((1 - t) * Math.Log(1 - p)))) },
            { LossFunctionTypes.KullbackLeibler, nameof(Distribution), (0.7 * Math.Log(0.7 / 0.5)) + (0.3 * Math.Log(0.3 / 0.3)) },
            { LossFunctionTypes.Focal, nameof(Probabilities), Mean(Probabilities, (t, p) => -((0.25 * t * Math.Pow(1 - p, 2) * Math.Log(p)) + (0.75 * (1 - t) * p * p * Math.Log(1 - p)))) },
            { LossFunctionTypes.Hinge, nameof(Signs), Mean(Signs, (t, p) => Math.Max(0, 1 - (t * p))) },
            { LossFunctionTypes.SquaredHinge, nameof(Signs), Mean(Signs, (t, p) => Math.Pow(Math.Max(0, 1 - (t * p)), 2)) },
            { LossFunctionTypes.SoftMargin, nameof(Signs), Mean(Signs, (t, p) => Math.Log(1 + Math.Exp(-t * p))) },
            { LossFunctionTypes.CosineDistance, nameof(Regression), 1 - (Regression.Sum(x => x.T * x.P) / (Math.Sqrt(Regression.Sum(x => x.T * x.T)) * Math.Sqrt(Regression.Sum(x => x.P * x.P)))) },
        };

        private static (double T, double P)[] Data(string name) => name switch
        {
            nameof(Regression) => Regression,
            nameof(Probabilities) => Probabilities,
            nameof(Distribution) => Distribution,
            nameof(Signs) => Signs,
            nameof(Positive) => Positive,
            _ => throw new ArgumentException(name),
        };

        [Theory]
        [MemberData(nameof(Values))]
        public void EveryLossGivesItsTextbookValue(LossFunctionTypes type, string data, double expected)
        {
            Assert.Equal(expected, Loss(type).CalculateLoss(Results(Data(data))), 5);
        }

        [Fact]
        public void EveryLossTypeHasAValueTest()
        {
            Assert.Equal(Enum.GetValues<LossFunctionTypes>().OrderBy(x => x), Values.Select(x => (LossFunctionTypes)x[0]).OrderBy(x => x));
        }

        [Theory]
        [MemberData(nameof(Values))]
        public void EveryGradientMatchesFiniteDifferences(LossFunctionTypes type, string data, double expected)
        {
            _ = expected;
            ILossFunction loss = Loss(type);
            List<ErrorResult> results = Results(Data(data));
            Quad[] gradient = loss.CalculateGradient(results);
            Assert.Equal(results.Count, gradient.Length);

            const double step = 1e-3;
            for (int i = 0; i < results.Count; i++)
            {
                Quad original = results[i].Predicted;
                results[i].Predicted = (Quad)(original + step);
                double plus = loss.CalculateLoss(results);
                results[i].Predicted = (Quad)(original - step);
                double minus = loss.CalculateLoss(results);
                results[i].Predicted = original;

                double numeric = (plus - minus) / (2 * step);
                if (type == LossFunctionTypes.Difference) numeric = (double)original - results[i].Target;   // trains like Squared
                Assert.True(Math.Abs(numeric - gradient[i]) <= 2e-3 * Math.Max(1, Math.Abs(numeric)),
                            $"{type} output {i}: numeric {numeric} vs analytic {(float)gradient[i]}");
            }
        }

        [Fact]
        public void OptionsSetTheParameters()
        {
            List<ErrorResult> results = Results(Regression);
            var options = new LossOptions { Delta = 0.4f, FocalGamma = 0, FocalAlpha = 0.5f };

            Assert.Equal(Mean(Regression, (t, p) => Huber(p - t, 0.4)), Loss(LossFunctionTypes.Huber, options).CalculateLoss(results), 5);
            Assert.Equal(Mean(Regression, (t, p) => Huber(p - t, 0.4) / 0.4), Loss(LossFunctionTypes.SmoothL1, options).CalculateLoss(results), 5);
            Assert.Equal(Mean(Regression, (t, p) => 0.16 * (Math.Sqrt(1 + Math.Pow((p - t) / 0.4, 2)) - 1)), Loss(LossFunctionTypes.PseudoHuber, options).CalculateLoss(results), 5);

            // without focusing and with even weights the focal loss is half the binary cross entropy
            List<ErrorResult> probabilities = Results(Probabilities);
            Assert.Equal(Loss(LossFunctionTypes.BinaryCrossEntropy).CalculateLoss(probabilities) / 2, Loss(LossFunctionTypes.Focal, options).CalculateLoss(probabilities), 5);
        }

        [Fact]
        public void InvalidOptionsAreRejected()
        {
            Assert.Throws<ArgumentException>(() => Loss(LossFunctionTypes.Huber, new LossOptions { Delta = 0 }));
            Assert.Throws<ArgumentException>(() => Loss(LossFunctionTypes.Huber, new LossOptions { Delta = float.NaN }));
            Assert.Throws<ArgumentException>(() => Loss(LossFunctionTypes.Focal, new LossOptions { FocalGamma = -1 }));
            Assert.Throws<ArgumentException>(() => Loss(LossFunctionTypes.Focal, new LossOptions { FocalAlpha = 1.5f }));
            Assert.Throws<ArgumentException>(() => Loss((LossFunctionTypes)999));
        }

        [Fact]
        public void EdgeValuesStayFinite()
        {
            foreach (LossFunctionTypes type in Enum.GetValues<LossFunctionTypes>())
            {
                ILossFunction loss = Loss(type);
                foreach (List<ErrorResult> results in new[] { Results((1, 1), (0, 0)), Results((0, 1), (1, 0)), Results((0, 0), (0, 0)), Results((1, -1)) })
                {
                    Assert.True(float.IsFinite(loss.CalculateLoss(results)), $"{type} loss");
                    Assert.All(loss.CalculateGradient(results), g => Assert.True(float.IsFinite(g), $"{type} gradient"));
                }

                // no outputs: nothing to add up, except the cosine distance, which has no direction to compare and gives 1
                Assert.Equal(type == LossFunctionTypes.CosineDistance ? 1f : 0f, (float)loss.CalculateLoss(new List<ErrorResult>()), 5);
            }

            Assert.Equal(0, Loss(LossFunctionTypes.CrossEntropy).CalculateLoss(Results((1, 1), (0, 0))), 5);
            Assert.Equal(0, Loss(LossFunctionTypes.KullbackLeibler).CalculateLoss(Results((0, 0.4), (1, 1))), 5);
        }

        /// <summary>
        /// The loss decides what is learned: for the targets 0, 0 and 9 the squared error is smallest at their mean 3
        /// and the absolute error at their median 0
        /// </summary>
        [Theory]
        [InlineData(LossFunctionTypes.Squared, 3)]
        [InlineData(LossFunctionTypes.Absolute, 0)]
        public void TheLossDecidesWhatIsLearned(LossFunctionTypes type, float expected)
        {
            var settings = new NetworkSettings
            {
                Seed = 1, LossType = type,
                InputSettings = new InputLayerSettings { NeuronsCount = 1, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 1, ActivationType = ActivationTypes.Identity }
            };

            foreach (INetwork network in new INetwork[] { NetworkFactory.BuildFFN(settings, (RecurrentLayerSettings)null), NetworkFactory.BuildMatrixFFN(settings) })
            {
                foreach (LearningRateOptimizer optimizer in ((IVectorNetwork)network).Optimizers) optimizer.LearningRate = 0.01f;
                Quad[] targets = { 0, 9, 0 };
                for (int step = 0; step < 6000; step++)
                {
                    network.Propagate(new Quad[] { 0 });
                    network.Backpropagate(targets[step % 3]);
                }

                Assert.Equal(expected, (float)network.Propagate(new Quad[] { 0 })[0], 0.2f);
            }
        }

        public static TheoryData<LossFunctionTypes, bool> NetworkCases()
        {
            var cases = new TheoryData<LossFunctionTypes, bool>();
            foreach (LossFunctionTypes type in new[] { LossFunctionTypes.Squared, LossFunctionTypes.Absolute, LossFunctionTypes.Huber, LossFunctionTypes.CrossEntropy,
                                                       LossFunctionTypes.BinaryCrossEntropy, LossFunctionTypes.KullbackLeibler, LossFunctionTypes.CosineDistance })
            {
                cases.Add(type, false);
                cases.Add(type, true);
            }

            return cases;
        }

        /// <summary>
        /// Every weight of a matrix network changes by the gradient of the loss, through the soft-max (the division by
        /// the sum of the outputs) when it is on
        /// </summary>
        [Theory]
        [MemberData(nameof(NetworkCases))]
        public void NetworkWeightGradientsMatchFiniteDifferencesOfTheLoss(LossFunctionTypes type, bool softMax)
        {
            var network = (MatrixNetwork)NetworkFactory.BuildMatrixFFN(new NetworkSettings
            {
                Seed = 3, LossType = type, OutputApplySoftMax = softMax,
                InputSettings = new InputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.Sigmoid, OptimizerType = OptimizerTypes.Normal }
            }, new RecurrentLayerSettings
            {
                NeuronsCount = 4, LayersCount = 1, ActivationType = ActivationTypes.HyperbolicTangens, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                HasBiasNeuron = true, OptimizerType = OptimizerTypes.Normal
            });

            // the soft-max divides linear outputs by their sum; keep them positive so they are probabilities
            if (softMax) Array.Fill(network.Layers[^1].Bias, (Quad)2);

            Quad[] input = { 0.3f, -0.7f, 0.5f };
            Quad[] target = { 0.2f, 0.7f, 0.1f };
            ILossFunction loss = Loss(type);
            double LossOf() => loss.CalculateLoss(network.Propagate(input).Select((p, i) => new ErrorResult { Predicted = p, Target = target[i] }).ToList());

            var tensors = network.Layers.SelectMany(x => new[] { x.Weights, x.Bias }).Where(x => x != null).ToList();
            var numeric = new List<double[]>();
            const double step = 1e-3;
            foreach (Quad[] tensor in tensors)
            {
                var gradient = new double[tensor.Length];
                for (int i = 0; i < tensor.Length; i++)
                {
                    Quad original = tensor[i];
                    tensor[i] = (Quad)(original + step);
                    double plus = LossOf();
                    tensor[i] = (Quad)(original - step);
                    double minus = LossOf();
                    tensor[i] = original;
                    gradient[i] = (plus - minus) / (2 * step);
                }

                numeric.Add(gradient);
            }

            List<Quad[]> before = tensors.Select(x => (Quad[])x.Clone()).ToList();
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 1;
            network.Propagate(input);
            Assert.Equal(LossOf(), network.Backpropagate(target), 5);

            for (int k = 0; k < tensors.Count; k++)
            {
                for (int i = 0; i < tensors[k].Length; i++)
                {
                    double analytic = before[k][i] - tensors[k][i];
                    Assert.True(Math.Abs(numeric[k][i] - analytic) <= 1e-2 * Math.Max(1, Math.Abs(numeric[k][i])),
                                $"{type} tensor {k} value {i}: numeric {numeric[k][i]} vs analytic {analytic}");
                }
            }
        }

        /// <summary>
        /// Graph, matrix and nested networks use the loss and its options, and keep both when saved
        /// </summary>
        [Fact]
        public void LossOptionsReachEveryNetworkAndSurviveSaving()
        {
            var options = new LossOptions { Delta = 0.05f };
            Quad[] input = { 0.5f, -0.25f };
            Quad[] target = { 0.9f, -0.4f };
            double Expected(Quad[] outputs) => outputs.Select((p, i) => Huber(p - target[i], 0.05)).Average();

            IGraphNetwork graph = NetworkFactory.BuildGraph(new GraphNetworkSettings
            {
                LossType = LossFunctionTypes.Huber, LossOptions = options, OutputActivationType = ActivationTypes.Identity,
                Nodes = { new GraphNode(1, NodeType.INPUT), new GraphNode(2, NodeType.INPUT), new GraphNode(3, NodeType.OUTPUT), new GraphNode(4, NodeType.OUTPUT) },
                Connections = { new GraphConnection(1, 1, 3, 0.4f), new GraphConnection(2, 2, 3, -0.7f), new GraphConnection(3, 1, 4, 0.2f), new GraphConnection(4, 2, 4, 0.9f) }
            });

            IMatrixNetwork matrix = NetworkFactory.BuildMatrixFFN(new NetworkSettings
            {
                Seed = 1, LossType = LossFunctionTypes.Huber, LossOptions = options,
                InputSettings = new InputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity }
            });

            INestedNetwork nested = NetworkFactory.BuildNested(new NestedNetworkSettings
            {
                InputSize = 2, LossType = LossFunctionTypes.Huber, LossOptions = options, Outputs = { 1 },
                Nodes =
                {
                    new NestedNode(1, (IVectorNetwork)NetworkFactory.BuildMatrixFFN(new NetworkSettings
                    {
                        Seed = 2,
                        InputSettings = new InputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity },
                        OutputSettings = new OutputLayerSettings { NeuronsCount = 2, ActivationType = ActivationTypes.Identity }
                    }), NestedNetworkSettings.InputNode)
                }
            });

            foreach (INetwork network in new INetwork[] { graph, matrix, nested })
            {
                using var stream = new System.IO.MemoryStream();
                NetworkSerializer.Save(network, stream);
                stream.Position = 0;
                INetwork loaded = NetworkSerializer.Load(stream);

                foreach (INetwork copy in new[] { network, loaded })
                {
                    Quad[] outputs = copy.Propagate(input);
                    Assert.Equal(Expected(outputs), copy.Backpropagate(target), 5);
                }
            }
        }

        [Theory]
        [MemberData(nameof(NetworkCases))]
        public void NeuronAndMatrixNetworksTrainTheSameOnEveryLoss(LossFunctionTypes type, bool softMax)
        {
            NetworkSettings Settings() => new NetworkSettings
            {
                Seed = 5, LossType = type, OutputApplySoftMax = softMax, IdentityGenerator = new IdentityGenerators.IncrementalGenerator(),
                InputSettings = new InputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.Identity, HasBiasNeuron = true },
                OutputSettings = new OutputLayerSettings { NeuronsCount = 3, ActivationType = ActivationTypes.Sigmoid, OptimizerType = OptimizerTypes.Adam }
            };
            var hidden = new RecurrentLayerSettings
            {
                NeuronsCount = 4, LayersCount = 1, ActivationType = ActivationTypes.HyperbolicTangens, PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                HasBiasNeuron = true, OptimizerType = OptimizerTypes.Adam
            };

            INetwork neurons = NetworkFactory.BuildFFN(Settings(), hidden);
            IMatrixNetwork matrix = NetworkFactory.BuildMatrixFFN(Settings(), hidden);
            foreach (LearningRateOptimizer optimizer in ((IVectorNetwork)neurons).Optimizers.Concat(matrix.Optimizers)) optimizer.LearningRate = 0.01f;

            var random = new Random(2);
            for (int step = 0; step < 20; step++)
            {
                Quad[] input = Enumerable.Range(0, 3).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray();
                Quad[] target = Enumerable.Range(0, 3).Select(_ => (Quad)random.NextDouble()).ToArray();
                Assert.Equal(neurons.Propagate(input).Select(x => (float)x), matrix.Propagate(input).Select(x => (float)x));
                Assert.Equal((float)neurons.Backpropagate(target), (float)matrix.Backpropagate(target));
                Assert.Equal(neurons.Errors.Select(x => (float)x), matrix.Errors.Select(x => (float)x));
            }
        }
    }
}
