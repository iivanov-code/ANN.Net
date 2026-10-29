using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.IdentityGenerators;
using ANN.Net.Tests.Transformers;
using ANN.Net.Utils;
using Xunit;

namespace ANN.Net.Tests.Networks
{
    /// <summary>
    /// The neuron-graph network: objects connected by synapses, built by <see cref="NetworkBuilder"/>
    /// </summary>
    public class NetworkTests
    {
        private static NetworkSettings Settings(ushort inputs,
                                                ushort outputs,
                                                ActivationTypes inputActivation = ActivationTypes.Identity,
                                                ActivationTypes outputActivation = ActivationTypes.Identity,
                                                bool bias = false)
        {
            return new NetworkSettings
            {
                IdentityGenerator = new IncrementalGenerator(),
                LossType = LossFunctionTypes.Squared,
                InputSettings = new InputLayerSettings { NeuronsCount = inputs, ActivationType = inputActivation, HasBiasNeuron = bias, InitializationType = WeightInitType.Xavier },
                OutputSettings = new OutputLayerSettings { NeuronsCount = outputs, ActivationType = outputActivation, HasBiasNeuron = false, InitializationType = WeightInitType.Xavier }
            };
        }

        private static RecurrentLayerSettings Hidden(ushort count, ActivationTypes activation, bool bias = false)
        {
            return new RecurrentLayerSettings
            {
                NeuronsCount = count,
                LayersCount = 1,
                CellType = NeuronType.Normal,
                PrevLayerConnectionType = LayerConnectionType.FullyConnected,
                ActivationType = activation,
                HasBiasNeuron = bias,
                InitializationType = WeightInitType.Xavier
            };
        }

        private static Network Build(NetworkSettings settings, RecurrentLayerSettings hidden, float learningRate)
        {
            Network network = (Network)NetworkFactory.BuildFFN(settings, hidden);
            foreach (LearningRateOptimizer optimizer in network.Optimizers)
            {
                optimizer.LearningRate = learningRate;
            }

            return network;
        }

        private static IEnumerable<BaseSynapse> AllSynapses(Network network)
        {
            IEnumerable<INeuron> neurons = network.InputNeurons.Cast<INeuron>();
            if (network.HiddenLayers != null)
            {
                neurons = neurons.Concat(network.HiddenLayers.SelectMany(layer => layer.Cast<INeuron>()));
            }

            return neurons.SelectMany(n => n.Outputs.Cast<BaseSynapse>());
        }

        private static double Loss(Quad[] values, Quad[] targets)
        {
            double loss = 0;
            for (int i = 0; i < values.Length; i++)
            {
                double difference = (float)values[i] - (float)targets[i];
                loss += difference * difference / 2;
            }

            return loss;
        }

        private static double Sigmoid(double x)
        {
            return 1 / (1 + Math.Exp(-x));
        }

        [Fact]
        public void Propagate_Twice_GivesTheSameOutput()
        {
            Network network = Build(Settings(2, 1), Hidden(3, ActivationTypes.Sigmoid), 0);

            Quad first = network.Propagate(new Quad[] { 1, 1 })[0];
            Quad second = network.Propagate(new Quad[] { 1, 1 })[0];

            Assert.Equal((float)first, (float)second, 6);
        }

        [Fact]
        public void Propagate_OneInputTwoOutputs_EachOutputUsesOnlyItsOwnWeight()
        {
            Network network = Build(Settings(1, 2), null, 0);
            float[] weights = network.InputNeurons[0].Outputs.Cast<BaseSynapse>().Select(s => (float)s.Weight).ToArray();

            Quad[] outputs = network.Propagate(new Quad[] { 0.7f });

            Assert.Equal(0.7f * weights[0], (float)outputs[0], 5);
            Assert.Equal(0.7f * weights[1], (float)outputs[1], 5);
        }

        [Fact]
        public void Propagate_MatchesManualComputation()
        {
            Network network = Build(Settings(2, 1), Hidden(3, ActivationTypes.Sigmoid), 0);
            Quad[] inputs = { 0.5f, -1.25f };

            Quad[] outputs = network.Propagate(inputs);

            double expected = 0;
            IHiddenNeuron[] hidden = network.HiddenLayers.Single().ToArray();
            for (int j = 0; j < hidden.Length; j++)
            {
                double sum = 0;
                for (int i = 0; i < 2; i++)
                {
                    BaseSynapse synapse = network.InputNeurons[i].Outputs.Cast<BaseSynapse>().Single(s => s.Output == hidden[j]);
                    sum += (float)inputs[i] * (float)synapse.Weight;
                }

                BaseSynapse outputSynapse = hidden[j].Outputs.Cast<BaseSynapse>().Single();
                expected += Sigmoid(sum) * (float)outputSynapse.Weight;
            }

            Assert.Equal(expected, (float)outputs[0], 5);
        }

        [Fact]
        public void Propagate_WithBiasNeurons_CompletesAndAddsTheBiasWeights()
        {
            Network network = Build(Settings(2, 1, bias: true), Hidden(2, ActivationTypes.Sigmoid, bias: true), 0);
            Quad[] inputs = { 0.3f, 0.9f };

            Quad[] outputs = network.Propagate(inputs);

            IHiddenNeuron[] hidden = network.HiddenLayers.Single().Where(n => !(n is Neurons.BiasNeuron)).ToArray();
            INeuron hiddenBias = network.HiddenLayers.Single().Single(n => n is Neurons.BiasNeuron);
            INeuron inputBias = network.InputNeurons.Single(n => n is Neurons.BiasNeuron);
            Assert.Equal(2, hidden.Length);

            double expected = (float)hiddenBias.Outputs.Cast<BaseSynapse>().Single().Weight;
            for (int j = 0; j < hidden.Length; j++)
            {
                double sum = (float)inputBias.Outputs.Cast<BaseSynapse>().Single(s => s.Output == hidden[j]).Weight;
                for (int i = 0; i < 2; i++)
                {
                    sum += (float)inputs[i] * (float)network.InputNeurons[i].Outputs.Cast<BaseSynapse>().Single(s => s.Output == hidden[j]).Weight;
                }

                expected += Sigmoid(sum) * (float)hidden[j].Outputs.Cast<BaseSynapse>().Single().Weight;
            }

            Assert.Equal(expected, (float)outputs[0], 5);
        }

        [Fact]
        public void Backpropagate_InputGradientsMatchFiniteDifferences()
        {
            Network network = Build(Settings(3, 2, bias: true), Hidden(4, ActivationTypes.HyperbolicTangens, bias: true), 0);
            Quad[] inputs = { 0.4f, -0.8f, 1.1f };
            Quad[] targets = { 0.25f, -0.5f };

            double[] numeric = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                Quad original = inputs[i];
                inputs[i] = original + (Quad)GradientChecker.EPSILON;
                double plus = Loss(network.Propagate(inputs), targets);
                inputs[i] = original - (Quad)GradientChecker.EPSILON;
                double minus = Loss(network.Propagate(inputs), targets);
                inputs[i] = original;
                numeric[i] = (plus - minus) / (2 * GradientChecker.EPSILON);
            }

            network.Propagate(inputs);
            network.Backpropagate(targets);
            Quad[] analytic = network.InputGradients;

            Assert.Equal(inputs.Length, analytic.Length);
            for (int i = 0; i < inputs.Length; i++)
            {
                GradientChecker.AssertClose(numeric[i], analytic[i], $"input gradient [{i}]");
            }
        }

        [Fact]
        public void Backpropagate_WeightUpdatesMatchFiniteDifferences()
        {
            Network network = Build(Settings(3, 2, bias: true), Hidden(4, ActivationTypes.Sigmoid, bias: true), 0);
            Quad[] inputs = { 0.4f, -0.8f, 1.1f };
            Quad[] targets = { 0.25f, -0.5f };
            BaseSynapse[] synapses = AllSynapses(network).ToArray();
            Assert.Equal((3 + 1) * 4 + (4 + 1) * 2, synapses.Length);

            double[] numeric = new double[synapses.Length];
            for (int s = 0; s < synapses.Length; s++)
            {
                Quad original = synapses[s].Weight;
                synapses[s].Weight = original + (Quad)GradientChecker.EPSILON;
                double plus = Loss(network.Propagate(inputs), targets);
                synapses[s].Weight = original - (Quad)GradientChecker.EPSILON;
                double minus = Loss(network.Propagate(inputs), targets);
                synapses[s].Weight = original;
                numeric[s] = (plus - minus) / (2 * GradientChecker.EPSILON);
            }

            // with learning rate 1 the plain optimizer subtracts exactly the gradient
            float[] before = synapses.Select(s => (float)s.Weight).ToArray();
            foreach (LearningRateOptimizer optimizer in network.Optimizers)
            {
                optimizer.LearningRate = 1f;
            }

            network.Propagate(inputs);
            network.Backpropagate(targets);

            for (int s = 0; s < synapses.Length; s++)
            {
                GradientChecker.AssertClose(numeric[s], before[s] - (float)synapses[s].Weight, $"weight gradient of synapse {s}");
            }
        }

        [Fact]
        public void Backpropagate_WithZeroLearningRate_LeavesTheOutputUnchanged()
        {
            Network network = Build(Settings(2, 1), Hidden(3, ActivationTypes.Sigmoid), 0);
            Quad[] inputs = { 0.2f, 0.4f };

            Quad before = network.Propagate(inputs)[0];
            network.Backpropagate(new Quad[] { 1 });
            Quad after = network.Propagate(inputs)[0];

            Assert.Equal((float)before, (float)after, 6);
        }

        [Fact]
        public void Train_Xor_IsLearned()
        {
            Network network = Build(Settings(2, 1, outputActivation: ActivationTypes.Sigmoid, bias: true), Hidden(8, ActivationTypes.Sigmoid, bias: true), 0.5f);
            Quad[][] inputs = { new Quad[] { 0, 0 }, new Quad[] { 0, 1 }, new Quad[] { 1, 0 }, new Quad[] { 1, 1 } };
            Quad[] targets = { 0, 1, 1, 0 };

            double initialLoss = 0;
            for (int s = 0; s < inputs.Length; s++)
            {
                initialLoss += Loss(network.Propagate(inputs[s]), new[] { targets[s] });
            }

            double loss = initialLoss;
            bool allCorrect = false;
            for (int epoch = 0; epoch < 8000 && !allCorrect; epoch++)
            {
                for (int s = 0; s < inputs.Length; s++)
                {
                    network.Propagate(inputs[s]);
                    network.Backpropagate(new[] { targets[s] });
                }

                if (epoch % 100 == 99)
                {
                    loss = 0;
                    allCorrect = true;
                    for (int s = 0; s < inputs.Length; s++)
                    {
                        Quad output = network.Propagate(inputs[s])[0];
                        loss += Loss(new[] { output }, new[] { targets[s] });
                        allCorrect &= (output > 0.5f) == (targets[s] > 0.5f) && Math.Abs(output - targets[s]) < 0.3f;
                    }
                }
            }

            Assert.True(allCorrect, $"XOR not learned, loss went from {initialLoss} to {loss}");
        }

        [Fact]
        public void Propagate_WrongInputCount_Throws()
        {
            Network network = Build(Settings(2, 1), null, 0);

            Assert.Throws<ArgumentException>(() => network.Propagate(new Quad[] { 1 }));
        }

        [Fact]
        public void HyperbolicTangens_IsTheHyperbolicTangent()
        {
            IActivationFunction tanh = NetworkUtils.GetActivation(ActivationTypes.HyperbolicTangens);
            Quad x = 1;

            Assert.Equal(Math.Tanh(1), (float)tanh.Activation(ref x), 5);
            Assert.Equal(1 - Math.Tanh(1) * Math.Tanh(1), (float)tanh.Derivative(1), 5);
        }

        [Fact]
        public void Sigmoid_DerivativeIsTakenAtThePreActivation()
        {
            IActivationFunction sigmoid = NetworkUtils.GetActivation(ActivationTypes.Sigmoid);
            double s = Sigmoid(0.8);

            Assert.Equal(s * (1 - s), (float)sigmoid.Derivative(0.8f), 5);
        }

        [Fact]
        public void Identity_PassesValuesThrough()
        {
            IActivationFunction identity = NetworkUtils.GetActivation(ActivationTypes.Identity);
            Quad x = -3.5f;

            Assert.Equal(-3.5f, (float)identity.Activation(ref x));
            Assert.Equal(1f, (float)identity.Derivative(-3.5f));
        }
    }
}
