using System;
using ANN.Net.Abstractions.Spiking;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Integrate-and-fire neuron that also fires negative spikes, the spiking counterpart of leaky ReLU: it fires a
    /// spike when the potential reaches the threshold, and a negative spike when it falls to -threshold / slope. For a
    /// constant current c it fires at the rate c / threshold when c is positive and at the rate slope * c / threshold
    /// (negative spikes) when c is negative, which is leaky ReLU. A slope of 1 gives the identity
    /// </summary>
    public sealed class SignedIntegrateAndFireActivation : ISpikingActivation
    {
        private readonly float negativeThreshold;

        /// <param name="negativeSlope">Slope for negative currents, greater than 0 and at most 1; 0.01 like ANN.Net's leaky ReLU</param>
        /// <param name="leak">Factor the potential is multiplied by every step before the current is added; 1 keeps it</param>
        public SignedIntegrateAndFireActivation(float threshold = 1, float negativeSlope = 0.01f, float leak = 1)
        {
            if (!(threshold > 0) || !float.IsFinite(threshold)) throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "The threshold must be greater than 0");
            if (!(negativeSlope > 0 && negativeSlope <= 1))
                throw new ArgumentOutOfRangeException(nameof(negativeSlope), negativeSlope, "The slope must be greater than 0 and at most 1");
            if (!(leak > 0 && leak <= 1)) throw new ArgumentOutOfRangeException(nameof(leak), leak, "The leak must be greater than 0 and at most 1");

            Threshold = threshold;
            NegativeSlope = negativeSlope;
            Leak = leak;
            negativeThreshold = -threshold / negativeSlope;
        }

        public float Threshold { get; }

        public float NegativeSlope { get; }

        public float Leak { get; }

        public int Step(ref SpikingNeuronState neuron, float current)
        {
            float potential = (neuron.Potential * Leak) + current;

            if (potential >= Threshold)
            {
                neuron.Potential = potential - Threshold;
                return 1;
            }

            if (potential <= negativeThreshold)
            {
                neuron.Potential = potential - negativeThreshold;
                return -1;
            }

            neuron.Potential = potential;
            return 0;
        }

        public int Step(Span<SpikingNeuronState> neurons, ReadOnlySpan<float> currents, Span<sbyte> spikes)
        {
            float leak = Leak, threshold = Threshold, negative = negativeThreshold;
            int fired = 0;
            for (int i = 0; i < neurons.Length; i++)
            {
                ref SpikingNeuronState neuron = ref neurons[i];
                float potential = (neuron.Potential * leak) + currents[i];
                sbyte spike = 0;

                if (potential >= threshold)
                {
                    potential -= threshold;
                    spike = 1;
                    fired++;
                }
                else if (potential <= negative)
                {
                    potential -= negative;
                    spike = -1;
                    fired++;
                }

                spikes[i] = spike;
                neuron.Potential = potential;
            }

            return fired;
        }

        public override string ToString() => $"Signed integrate-and-fire (threshold {Threshold}, slope {NegativeSlope}, leak {Leak})";
    }
}
