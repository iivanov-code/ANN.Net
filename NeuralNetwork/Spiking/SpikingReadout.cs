using System;
using ANN.Net.Compact;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Turns what the output neurons did during one input into output values: either their firing rates, or, for a
    /// converted network, the activation of the average input current they received (the output of the original
    /// network, as estimated from the spikes)
    /// </summary>
    internal sealed class SpikingReadout
    {
        private SpikingReadout(int[] neurons, float[] divisors, CompactActivation[] activations, bool divideBySum)
        {
            Neurons = neurons;
            this.divisors = divisors;
            this.activations = activations;
            this.divideBySum = divideBySum;
        }

        private readonly float[] divisors;
        private readonly CompactActivation[] activations;
        private readonly bool divideBySum;

        /// <summary>
        /// Neuron of every output; -1 for an output that is always 0
        /// </summary>
        public int[] Neurons { get; }

        public bool FromPotential => divisors != null;

        public static SpikingReadout Rates(int firstNeuron, int count)
        {
            int[] neurons = new int[count];
            for (int i = 0; i < count; i++) neurons[i] = firstNeuron + i;
            return new SpikingReadout(neurons, null, null, false);
        }

        /// <param name="divisors">What the input currents of every output were divided by when the weights were scaled</param>
        /// <param name="activations">Activation of the original network for every output</param>
        /// <param name="divideBySum">Divide every output by the sum of all outputs, as the original network does</param>
        public static SpikingReadout Potentials(int[] neurons, float[] divisors, CompactActivation[] activations, bool divideBySum)
        {
            return new SpikingReadout(neurons, divisors, activations, divideBySum);
        }

        /// <param name="currents">Input current of every output summed over all steps, biases included</param>
        /// <param name="spikes">Spikes of every output</param>
        public void Decode(ReadOnlySpan<float> currents, ReadOnlySpan<int> spikes, int steps, Span<float> outputs)
        {
            for (int i = 0; i < Neurons.Length; i++)
            {
                if (Neurons[i] < 0) outputs[i] = 0;
                else if (FromPotential) outputs[i] = activations[i].Apply(divisors[i] * currents[i] / steps);
                else outputs[i] = (float)spikes[i] / steps;
            }

            if (!divideBySum) return;

            // as CompactGraphNetwork and the neuron networks
            Quad total = 0;
            foreach (float value in outputs) total += value;
            for (int i = 0; i < outputs.Length; i++) outputs[i] = (Quad)outputs[i] / total;
        }
    }
}
