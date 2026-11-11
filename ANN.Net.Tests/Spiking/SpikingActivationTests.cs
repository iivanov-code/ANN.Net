using System;
using System.Linq;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.Spiking;
using Xunit;

namespace ANN.Net.Tests.Spiking
{
    /// <summary>
    /// The firing rate of every spike function follows the activation it stands in for
    /// </summary>
    public class SpikingActivationTests
    {
        private static (int Positive, int Negative) Count(ISpikingActivation activation, float current, int steps)
        {
            var neuron = new SpikingNeuronState();
            int positive = 0, negative = 0;
            for (int i = 0; i < steps; i++)
            {
                int spike = activation.Step(ref neuron, current);
                if (spike > 0) positive++;
                if (spike < 0) negative++;
            }

            return (positive, negative);
        }

        [Theory]
        [InlineData(0.25f, 25)]
        [InlineData(0.5f, 50)]
        [InlineData(1f, 100)]
        [InlineData(3f, 100)]
        [InlineData(0f, 0)]
        [InlineData(-0.5f, 0)]
        public void IntegrateAndFireFiresAtTheReluRate(float current, int expected)
        {
            Assert.Equal((expected, 0), Count(new IntegrateAndFireActivation(), current, 100));
        }

        [Fact]
        public void IntegrateAndFireSpacesItsSpikesEvenly()
        {
            var activation = new IntegrateAndFireActivation();
            var neuron = new SpikingNeuronState();
            int[] spikes = Enumerable.Range(0, 12).Select(_ => activation.Step(ref neuron, 0.25f)).ToArray();

            Assert.Equal(new[] { 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1 }, spikes);
        }

        [Fact]
        public void ThresholdScalesTheRate()
        {
            Assert.Equal((25, 0), Count(new IntegrateAndFireActivation(threshold: 2), 0.5f, 100));
        }

        [Fact]
        public void LeakForgetsInputsThatArriveSlowly()
        {
            // without leak 0.375 per step fires 37 times; with leak 0.5 the potential settles at 0.75 and never fires
            Assert.Equal((37, 0), Count(new IntegrateAndFireActivation(), 0.375f, 100));
            Assert.Equal((0, 0), Count(new IntegrateAndFireActivation(leak: 0.5f), 0.375f, 100));
        }

        [Fact]
        public void ResetToZeroLosesThePartAboveTheThreshold()
        {
            // 0.625 per step: subtracting keeps the rest and fires 62 times, resetting to 0 fires every second step
            Assert.Equal((62, 0), Count(new IntegrateAndFireActivation(), 0.625f, 100));
            Assert.Equal((50, 0), Count(new IntegrateAndFireActivation(resetToZero: true), 0.625f, 100));
        }

        [Fact]
        public void AdaptationRaisesTheThresholdOfBusyNeurons()
        {
            var activation = new IntegrateAndFireActivation(thresholdAdaptation: 0.5f);
            var neuron = new SpikingNeuronState();
            int spikes = 0;
            for (int i = 0; i < 100; i++) spikes += activation.Step(ref neuron, 0.5f);

            Assert.InRange(spikes, 5, 15);
            Assert.True(neuron.Adaptation > 2);

            // a new input starts from rest but keeps the higher threshold
            neuron.Restart();
            Assert.Equal(0, neuron.Potential);
            Assert.True(neuron.Adaptation > 2);
        }

        [Theory]
        [InlineData(0.25f, 0.01f, 25, 0)]
        [InlineData(-50f, 0.01f, 0, 50)]
        [InlineData(-0.25f, 1f, 0, 25)]
        [InlineData(-0.5f, 0.1f, 0, 5)]
        public void SignedIntegrateAndFireFiresAtTheLeakyReluRate(float current, float slope, int positive, int negative)
        {
            Assert.Equal((positive, negative), Count(new SignedIntegrateAndFireActivation(negativeSlope: slope), current, 100));
        }

        [Theory]
        [InlineData(0f)]
        [InlineData(2f)]
        [InlineData(-1.5f)]
        [InlineData(6f)]
        public void SigmoidRateFiresAtTheSigmoidRate(float current)
        {
            const int steps = 200;
            double expected = steps / (1 + Math.Exp(-current));

            Assert.InRange(Count(new SigmoidRateActivation(), current, steps).Positive, expected - 1, expected + 1);
        }

        [Fact]
        public void SigmoidRateWithGainTwoGivesTanh()
        {
            const int steps = 200;
            int spikes = Count(new SigmoidRateActivation(gain: 2), 0.7f, steps).Positive;

            Assert.InRange((2.0 * spikes / steps) - 1, Math.Tanh(0.7) - 0.02, Math.Tanh(0.7) + 0.02);
        }

        [Fact]
        public void PoissonFiresWithTheGivenProbability()
        {
            int spikes = Count(new PoissonActivation(seed: 3), 0.3f, 10_000).Positive;

            Assert.InRange(spikes, 2850, 3150);
            Assert.Equal(spikes, Count(new PoissonActivation(seed: 3), 0.3f, 10_000).Positive);
            Assert.Equal((0, 500), Count(new PoissonActivation(seed: 3), -1f, 500));
        }

        public static TheoryData<string> Activations() => new TheoryData<string> { "if", "lif", "reset", "adapt", "signed", "sigmoid" };

        private static ISpikingActivation Create(string name) => name switch
        {
            "if" => new IntegrateAndFireActivation(),
            "lif" => new IntegrateAndFireActivation(threshold: 0.8f, leak: 0.9f),
            "reset" => new IntegrateAndFireActivation(resetToZero: true),
            "adapt" => new IntegrateAndFireActivation(thresholdAdaptation: 0.1f, adaptationDecay: 0.99f),
            "signed" => new SignedIntegrateAndFireActivation(negativeSlope: 0.2f, leak: 0.95f),
            _ => new SigmoidRateActivation(gain: 1.5f)
        };

        [Theory]
        [MemberData(nameof(Activations))]
        public void LayerStepEqualsNeuronSteps(string name)
        {
            ISpikingActivation layer = Create(name), single = Create(name);
            float[] currents = { 0.3f, -0.2f, 1.4f, 0f, 0.71f, -3f, 0.05f, 0.5f, 2f, -0.6f, 0.9f };
            var layerStates = new SpikingNeuronState[currents.Length];
            var singleStates = new SpikingNeuronState[currents.Length];
            sbyte[] spikes = new sbyte[currents.Length];

            for (int step = 0; step < 50; step++)
            {
                int fired = layer.Step(layerStates, currents, spikes);

                int expectedFired = 0;
                for (int i = 0; i < currents.Length; i++)
                {
                    int spike = single.Step(ref singleStates[i], currents[i]);
                    Assert.Equal(spike, spikes[i]);
                    if (spike != 0) expectedFired++;
                }

                Assert.Equal(expectedFired, fired);
            }

            Assert.Equal(singleStates, layerStates);
        }

        [Fact]
        public void InvalidParametersAreRejected()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new IntegrateAndFireActivation(threshold: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new IntegrateAndFireActivation(leak: 1.5f));
            Assert.Throws<ArgumentOutOfRangeException>(() => new IntegrateAndFireActivation(thresholdAdaptation: -1));
            Assert.Throws<ArgumentOutOfRangeException>(() => new IntegrateAndFireActivation(adaptationDecay: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new SignedIntegrateAndFireActivation(negativeSlope: 0));
            Assert.Throws<ArgumentOutOfRangeException>(() => new SignedIntegrateAndFireActivation(threshold: float.NaN));
            Assert.Throws<ArgumentOutOfRangeException>(() => new SigmoidRateActivation(gain: -1));
        }
    }
}
