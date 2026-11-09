using System;
using ANN.Net.Abstractions.Spiking;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Spiking counterpart of the sigmoid: the neuron averages its input current since the input started and fires
    /// just often enough that its spike count stays at steps * sigmoid(gain * average). Its firing rate therefore tends
    /// to sigmoid(gain * c) for a constant current c, between 0 and 1 spike per step, without any scaling. It uses
    /// <see cref="SpikingNeuronState.Potential"/> for the summed current, so it does not leak
    /// </summary>
    public sealed class SigmoidRateActivation : ISpikingActivation
    {
        /// <param name="gain">Factor the average current is multiplied by before the sigmoid; 2 gives the rate (tanh + 1) / 2</param>
        public SigmoidRateActivation(float gain = 1)
        {
            if (!(gain > 0) || !float.IsFinite(gain)) throw new ArgumentOutOfRangeException(nameof(gain), gain, "The gain must be greater than 0");

            Gain = gain;
        }

        public float Gain { get; }

        public int Step(ref SpikingNeuronState neuron, float current)
        {
            float total = neuron.Potential + current;
            int steps = neuron.Steps + 1;
            neuron.Potential = total;
            neuron.Steps = steps;

            float target = steps / (1 + MathF.Exp(-Gain * total / steps));
            if (target - neuron.Spikes >= 0.5f)
            {
                neuron.Spikes++;
                return 1;
            }

            return 0;
        }

        public int Step(Span<SpikingNeuronState> neurons, ReadOnlySpan<float> currents, Span<sbyte> spikes)
        {
            int fired = 0;
            for (int i = 0; i < neurons.Length; i++)
            {
                int spike = Step(ref neurons[i], currents[i]);
                spikes[i] = (sbyte)spike;
                fired += spike;
            }

            return fired;
        }

        public override string ToString() => $"Sigmoid rate (gain {Gain})";
    }
}
