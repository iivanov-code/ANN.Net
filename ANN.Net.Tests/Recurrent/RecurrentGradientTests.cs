using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Recurrent;
using ANN.Net.Tests.Transformers;
using Xunit;

namespace ANN.Net.Tests.Recurrent
{
    /// <summary>
    /// Backpropagation through time against central finite differences of the loss, for every weight of a
    /// two-layer network: the weights of the lower layer only get the right gradient if the gradient of its outputs
    /// coming down from the upper layer is right too
    /// </summary>
    public class RecurrentGradientTests
    {
        private const int Steps = 6;

        public static TheoryData<NeuronType, bool, bool> Cases => new TheoryData<NeuronType, bool, bool>
        {
            { NeuronType.LSTM, false, false },
            { NeuronType.LSTM, true, true },
            { NeuronType.LSTMPeepHole, false, false },
            { NeuronType.LSTMPeepHole, true, true },
            { NeuronType.GRU, false, false },
            { NeuronType.GRU, true, true },
            { NeuronType.RNN, false, false },
            { NeuronType.RNN, true, true },
        };

        [Theory]
        [MemberData(nameof(Cases))]
        public void EveryWeightGradientMatchesFiniteDifferences(NeuronType cell, bool softMax, bool lastStepOnly)
        {
            var network = new RecurrentNetwork(new RecurrentNetworkSettings
            {
                InputDimension = 3, HiddenDimension = 4, LayersCount = 2, OutputDimension = 3, CellType = cell,
                OutputApplySoftMax = softMax, OptimizerType = OptimizerTypes.Normal, Seed = 5
            });

            var random = new Random(11);
            Quad[][] inputs = GradientChecker.RandomSequence(random, Steps, 3, 2);
            Quad[][] targets = Enumerable.Range(0, Steps).Select(t => Target(random, softMax)).ToArray();
            if (lastStepOnly)
            {
                for (int t = 0; t < Steps - 1; t++) targets[t] = null;
            }

            // peepholes start at 0, which hides mistakes in their path; give them values
            foreach (RecurrentLayer layer in network.Layers)
            {
                if (layer is LstmLayer { HasPeepholes: true } lstm)
                {
                    Span<float> peepholes = lstm.Peepholes.Weights;
                    for (int i = 0; i < peepholes.Length; i++) peepholes[i] = (float)((random.NextDouble() * 2) - 1);
                }
            }

            var numeric = new List<double[]>();
            foreach (RecurrentTensor tensor in network.Tensors)
            {
                var gradient = new double[tensor.Length];
                for (int i = 0; i < tensor.Length; i++)
                {
                    Quad original = tensor.Values[i];
                    tensor.Values[i] = original + (Quad)GradientChecker.EPSILON;
                    double plus = Loss(network.Propagate(inputs), targets, softMax);
                    tensor.Values[i] = original - (Quad)GradientChecker.EPSILON;
                    double minus = Loss(network.Propagate(inputs), targets, softMax);
                    tensor.Values[i] = original;
                    gradient[i] = (plus - minus) / (2 * GradientChecker.EPSILON);
                }

                numeric.Add(gradient);
            }

            // with learning rate 1 the plain optimizer subtracts exactly the gradient
            List<Quad[]> before = network.Tensors.Select(x => (Quad[])x.Values.Clone()).ToList();
            foreach (LearningRateOptimizer optimizer in network.Optimizers) optimizer.LearningRate = 1;

            network.Propagate(inputs);
            double loss = network.Backpropagate(targets);
            Assert.Equal(Loss(network.Values, targets, softMax), loss, 4);

            for (int k = 0; k < numeric.Count; k++)
            {
                Quad[] after = network.Tensors[k].Values;
                for (int i = 0; i < after.Length; i++)
                {
                    GradientChecker.AssertClose(numeric[k][i], before[k][i] - after[i], $"tensor {k} value {i}");
                }
            }
        }

        private static Quad[] Target(Random random, bool softMax)
        {
            if (!softMax) return Enumerable.Range(0, 3).Select(_ => (Quad)((random.NextDouble() * 2) - 1)).ToArray();

            int k = random.Next(3);
            return Enumerable.Range(0, 3).Select(i => (Quad)(i == k ? 1 : 0)).ToArray();
        }

        private static double Loss(Quad[][] outputs, Quad[][] targets, bool softMax)
        {
            double loss = 0;
            for (int t = 0; t < outputs.Length; t++)
            {
                if (targets[t] == null) continue;
                for (int i = 0; i < outputs[t].Length; i++)
                {
                    double y = outputs[t][i], target = targets[t][i];
                    loss += softMax ? -target * Math.Log(Math.Max(y, 1e-7)) : (y - target) * (y - target) / 2;
                }
            }

            return loss;
        }
    }
}
