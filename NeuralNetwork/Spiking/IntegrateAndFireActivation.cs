using System;
using ANN.Net.Abstractions.Spiking;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Integrate-and-fire neuron, the spiking counterpart of ReLU: the input current is added to the membrane
    /// potential, and when the potential reaches the threshold the neuron fires and the threshold is subtracted. For a
    /// constant current c it fires at the rate max(0, c) / threshold (at most every step), which is ReLU. With a leak
    /// below 1 the potential decays every step (leaky integrate-and-fire), so only inputs arriving close together add up
    /// </summary>
    public sealed class IntegrateAndFireActivation : ISpikingActivation
    {
        /// <param name="threshold">Potential at which the neuron fires</param>
        /// <param name="leak">Factor the potential is multiplied by every step before the current is added; 1 keeps it</param>
        /// <param name="resetToZero">Set the potential to 0 after a spike instead of subtracting the threshold; biologically
        /// closer, but loses the part above the threshold, so firing rates no longer follow ReLU exactly</param>
        /// <param name="thresholdAdaptation">Added to the neuron's threshold at every spike (homeostasis); 0 turns it off</param>
        /// <param name="adaptationDecay">Factor the added threshold is multiplied by every step</param>
        public IntegrateAndFireActivation(float threshold = 1, float leak = 1, bool resetToZero = false,
                                          float thresholdAdaptation = 0, float adaptationDecay = 1)
        {
            if (!(threshold > 0) || !float.IsFinite(threshold)) throw new ArgumentOutOfRangeException(nameof(threshold), threshold, "The threshold must be greater than 0");
            if (!(leak > 0 && leak <= 1)) throw new ArgumentOutOfRangeException(nameof(leak), leak, "The leak must be greater than 0 and at most 1");
            if (!(thresholdAdaptation >= 0) || !float.IsFinite(thresholdAdaptation))
                throw new ArgumentOutOfRangeException(nameof(thresholdAdaptation), thresholdAdaptation, "The adaptation must be 0 or more");
            if (!(adaptationDecay > 0 && adaptationDecay <= 1))
                throw new ArgumentOutOfRangeException(nameof(adaptationDecay), adaptationDecay, "The decay must be greater than 0 and at most 1");

            Threshold = threshold;
            Leak = leak;
            ResetToZero = resetToZero;
            ThresholdAdaptation = thresholdAdaptation;
            AdaptationDecay = adaptationDecay;
        }

        public float Threshold { get; }

        public float Leak { get; }

        public bool ResetToZero { get; }

        public float ThresholdAdaptation { get; }

        public float AdaptationDecay { get; }

        private bool Adapts => ThresholdAdaptation > 0 || AdaptationDecay < 1;

        public int Step(ref SpikingNeuronState neuron, float current)
        {
            float potential = (neuron.Potential * Leak) + current;
            float threshold = Threshold;

            if (Adapts)
            {
                float adaptation = neuron.Adaptation * AdaptationDecay;
                threshold += adaptation;
                if (potential >= threshold) adaptation += ThresholdAdaptation;
                neuron.Adaptation = adaptation;
            }

            if (potential >= threshold)
            {
                neuron.Potential = ResetToZero ? 0 : potential - threshold;
                return 1;
            }

            neuron.Potential = potential;
            return 0;
        }

        public int Step(Span<SpikingNeuronState> neurons, ReadOnlySpan<float> currents, Span<sbyte> spikes)
        {
            if (Adapts || ResetToZero)
            {
                int count = 0;
                for (int i = 0; i < neurons.Length; i++)
                {
                    int spike = Step(ref neurons[i], currents[i]);
                    spikes[i] = (sbyte)spike;
                    count += spike;
                }

                return count;
            }

            // the plain neuron of converted networks, without calls
            float leak = Leak, threshold = Threshold;
            int fired = 0;
            for (int i = 0; i < neurons.Length; i++)
            {
                ref SpikingNeuronState neuron = ref neurons[i];
                float potential = (neuron.Potential * leak) + currents[i];

                if (potential >= threshold)
                {
                    potential -= threshold;
                    spikes[i] = 1;
                    fired++;
                }
                else
                {
                    spikes[i] = 0;
                }

                neuron.Potential = potential;
            }

            return fired;
        }

        public override string ToString() => $"Integrate-and-fire (threshold {Threshold}, leak {Leak})";
    }
}
