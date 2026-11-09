using System.Collections.Generic;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// How a trained network is turned into a spiking network. Every neuron becomes a spiking neuron whose firing
    /// rate follows its activation: ReLU an integrate-and-fire neuron, leaky ReLU and the identity a neuron that also
    /// fires negative spikes, sigmoid and tanh a neuron that fires at the rate sigmoid(input). The weights are scaled
    /// so that the largest activation seen in <see cref="CalibrationInputs"/> fires about once per time step
    /// </summary>
    public sealed class SpikingConversionOptions
    {
        /// <summary>
        /// Typical inputs, for instance a few hundred training samples. They set the input normalization and the
        /// scale of every layer; required unless every layer uses sigmoid or tanh
        /// </summary>
        public IReadOnlyList<float[]> CalibrationInputs { get; set; }

        /// <summary>
        /// Percentile of the activations in <see cref="CalibrationInputs"/> that fires once per time step. Below 100
        /// rare outliers are cut off, and every other value gets a higher, more precise firing rate
        /// </summary>
        public double Percentile { get; set; } = 99.9;

        /// <summary>
        /// Time steps every input is presented for. More steps give outputs closer to the original network's
        /// </summary>
        public int TimeSteps { get; set; } = 64;

        /// <summary>
        /// Normalize the inputs to [-1, 1] and fire negative spikes for negative inputs, instead of normalizing them to
        /// [0, 1]. Better when inputs are spread around 0
        /// </summary>
        public bool SignedInputs { get; set; }
    }
}
