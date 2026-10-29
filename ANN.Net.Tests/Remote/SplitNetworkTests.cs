using System;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.Remote;
using ANN.Net.Tests.Transformers;
using Xunit;

namespace ANN.Net.Tests.Remote
{
    /// <summary>
    /// Head and tail connected in this process through <see cref="RemoteNetworkPart"/>, no sockets involved
    /// </summary>
    public class SplitNetworkTests
    {
        private static SplitNetwork Build(out Network tail, ActivationTypes outputActivation = ActivationTypes.Identity)
        {
            NetworkSettings whole = SplitHelpers.Whole(3, 2, outputActivation, inputBias: true);
            RecurrentLayerSettings[] layers = { SplitHelpers.Layer(4, ActivationTypes.HyperbolicTangens, bias: true), SplitHelpers.Layer(3, ActivationTypes.Sigmoid, bias: true) };
            return SplitHelpers.BuildInProcess(whole, layers, 0, out tail);
        }

        [Fact]
        public void Propagate_ReturnsTheTailOutputsForTheBoundaryValues()
        {
            SplitNetwork split = Build(out Network tail);
            Quad[] inputs = { 0.5f, -1.25f, 0.75f };

            Quad[] outputs = split.Propagate(inputs);

            Assert.Equal(2, outputs.Length);
            Assert.Equal(outputs.Select(v => (float)v), split.Values.Select(v => (float)v));

            Quad[] boundary = split.Head.Values;
            Assert.Equal(4, boundary.Length);
            Quad[] expected = tail.Propagate(boundary);
            for (int i = 0; i < expected.Length; i++)
            {
                Assert.Equal((float)expected[i], (float)outputs[i], 6);
            }
        }

        [Fact]
        public void Propagate_Twice_GivesTheSameOutput()
        {
            SplitNetwork split = Build(out _);

            Quad[] first = split.Propagate(new Quad[] { 1, 1, 1 });
            Quad[] second = split.Propagate(new Quad[] { 1, 1, 1 });

            Assert.Equal(first.Select(v => (float)v), second.Select(v => (float)v));
            Assert.Equal(2, split.Connection.PropagationCount);
        }

        [Fact]
        public void Propagate_WrongInputCount_Throws()
        {
            SplitNetwork split = Build(out _);

            Assert.Throws<ArgumentException>(() => split.Propagate(new Quad[] { 1 }));
        }

        [Fact]
        public void Backpropagate_BeforePropagate_Throws()
        {
            SplitNetwork split = Build(out _);

            Assert.Throws<InvalidOperationException>(() => split.Backpropagate(new Quad[] { 0, 0 }));
        }

        [Fact]
        public void Backpropagate_ReturnsTheTailLossAndErrors()
        {
            SplitNetwork split = Build(out _);
            Quad[] targets = { 0.25f, -0.5f };

            Quad[] values = split.Propagate(new Quad[] { 0.4f, -0.8f, 1.1f });
            Quad loss = split.Backpropagate(targets);

            double expectedLoss = 0;
            for (int i = 0; i < values.Length; i++)
            {
                double error = (float)values[i] - (float)targets[i];
                expectedLoss += error * error;
                Assert.Equal(error, (float)split.Errors[i], 5);
            }

            Assert.Equal(expectedLoss, (float)loss, 4);
        }

        [Fact]
        public void Backpropagate_InputGradientsMatchFiniteDifferences()
        {
            SplitNetwork split = Build(out _);
            Quad[] inputs = { 0.4f, -0.8f, 1.1f };
            Quad[] targets = { 0.25f, -0.5f };

            double[] numeric = new double[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                Quad original = inputs[i];
                inputs[i] = original + (Quad)GradientChecker.EPSILON;
                double plus = SplitHelpers.Loss(split.Propagate(inputs), targets);
                inputs[i] = original - (Quad)GradientChecker.EPSILON;
                double minus = SplitHelpers.Loss(split.Propagate(inputs), targets);
                inputs[i] = original;
                numeric[i] = (plus - minus) / (2 * GradientChecker.EPSILON);
            }

            split.Propagate(inputs);
            split.Backpropagate(targets);
            Quad[] analytic = split.InputGradients;

            Assert.Equal(inputs.Length, analytic.Length);
            for (int i = 0; i < inputs.Length; i++)
            {
                GradientChecker.AssertClose(numeric[i], analytic[i], $"input gradient [{i}]");
            }
        }

        [Fact]
        public void Backpropagate_WeightUpdatesOnBothSidesMatchFiniteDifferences()
        {
            SplitNetwork split = Build(out Network tail);
            Quad[] inputs = { 0.4f, -0.8f, 1.1f };
            Quad[] targets = { 0.25f, -0.5f };

            BaseSynapse[] headSynapses = SplitHelpers.TrainableSynapses((Network)split.Head).ToArray();
            BaseSynapse[] tailSynapses = SplitHelpers.TrainableSynapses(tail).ToArray();
            BaseSynapse[] synapses = headSynapses.Concat(tailSynapses).ToArray();
            Assert.Equal((3 + 1) * 4, headSynapses.Length);
            Assert.Equal((4 + 1) * 3 + (3 + 1) * 2, tailSynapses.Length);

            double[] numeric = new double[synapses.Length];
            for (int s = 0; s < synapses.Length; s++)
            {
                Quad original = synapses[s].Weight;
                synapses[s].Weight = original + (Quad)GradientChecker.EPSILON;
                double plus = SplitHelpers.Loss(split.Propagate(inputs), targets);
                synapses[s].Weight = original - (Quad)GradientChecker.EPSILON;
                double minus = SplitHelpers.Loss(split.Propagate(inputs), targets);
                synapses[s].Weight = original;
                numeric[s] = (plus - minus) / (2 * GradientChecker.EPSILON);
            }

            // with learning rate 1 the plain optimizer subtracts exactly the gradient, on both sides of the boundary
            float[] before = synapses.Select(s => (float)s.Weight).ToArray();
            split.SetLearningRate(1f);

            split.Propagate(inputs);
            split.Backpropagate(targets);

            for (int s = 0; s < synapses.Length; s++)
            {
                string side = s < headSynapses.Length ? "head" : "tail";
                GradientChecker.AssertClose(numeric[s], before[s] - (float)synapses[s].Weight, $"weight gradient of {side} synapse {s}");
            }
        }

        [Fact]
        public void SetLearningRate_Zero_LeavesEverythingUnchanged()
        {
            SplitNetwork split = Build(out Network tail);
            Quad[] inputs = { 0.2f, 0.4f, -0.6f };
            float[] weights = SplitHelpers.TrainableSynapses((Network)split.Head).Concat(SplitHelpers.TrainableSynapses(tail)).Select(s => (float)s.Weight).ToArray();

            split.SetLearningRate(0f);
            Quad[] before = split.Propagate(inputs);
            split.Backpropagate(new Quad[] { 1, -1 });
            Quad[] after = split.Propagate(inputs);

            Assert.Equal(before.Select(v => (float)v), after.Select(v => (float)v));
            Assert.Equal(weights, SplitHelpers.TrainableSynapses((Network)split.Head).Concat(SplitHelpers.TrainableSynapses(tail)).Select(s => (float)s.Weight));
        }

        [Fact]
        public void Train_Xor_IsLearnedAcrossTheBoundary()
        {
            NetworkSettings whole = SplitHelpers.Whole(2, 1, ActivationTypes.Sigmoid, inputBias: true);
            RecurrentLayerSettings[] layers = { SplitHelpers.Layer(8, ActivationTypes.Sigmoid, bias: true) };
            SplitNetwork split = SplitHelpers.BuildInProcess(whole, layers, 0, out _);
            split.SetLearningRate(0.5f);

            Quad[][] inputs = { new Quad[] { 0, 0 }, new Quad[] { 0, 1 }, new Quad[] { 1, 0 }, new Quad[] { 1, 1 } };
            Quad[] targets = { 0, 1, 1, 0 };

            double initialLoss = 0;
            for (int s = 0; s < inputs.Length; s++)
            {
                initialLoss += SplitHelpers.Loss(split.Propagate(inputs[s]), new[] { targets[s] });
            }

            double loss = initialLoss;
            bool allCorrect = false;
            for (int epoch = 0; epoch < 8000 && !allCorrect; epoch++)
            {
                for (int s = 0; s < inputs.Length; s++)
                {
                    split.Propagate(inputs[s]);
                    split.Backpropagate(new[] { targets[s] });
                }

                if (epoch % 100 == 99)
                {
                    loss = 0;
                    allCorrect = true;
                    for (int s = 0; s < inputs.Length; s++)
                    {
                        Quad output = split.Propagate(inputs[s])[0];
                        loss += SplitHelpers.Loss(new[] { output }, new[] { targets[s] });
                        allCorrect &= (output > 0.5f) == (targets[s] > 0.5f) && Math.Abs(output - targets[s]) < 0.3f;
                    }
                }
            }

            Assert.True(allCorrect, $"XOR not learned across the boundary, loss went from {initialLoss} to {loss}");
        }

        [Fact]
        public void Save_IsNotSupported()
        {
            SplitNetwork split = Build(out _);

            Assert.Throws<NotSupportedException>(() => split.Save(".", "split"));
        }
    }
}
