using System;
using System.Collections.Generic;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// A spiking network of any topology, for example the phenotype of a NEAT genome: layers of neurons and any
    /// synapses between them. Neurons are numbered across the layers, layer 0 holding the inputs. A synapse into a later
    /// layer delivers in the same time step, a synapse within a layer or back to an earlier one in the next step, so
    /// cycles are allowed and give the network memory. The outputs are the firing rates of <see cref="Outputs"/>
    /// </summary>
    public sealed class SpikingGraphSettings
    {
        public IList<SpikingGraphLayer> Layers { get; set; } = new List<SpikingGraphLayer>();

        public IList<SpikingSynapse> Synapses { get; set; } = new List<SpikingSynapse>();

        /// <summary>
        /// Current every neuron receives in every step, one value per neuron (0 for the inputs); null for none
        /// </summary>
        public IList<float> Bias { get; set; }

        /// <summary>
        /// Neurons whose firing rates are the outputs, in output order; empty for every neuron of the last layer
        /// </summary>
        public IList<int> Outputs { get; set; } = new List<int>();

        /// <summary>
        /// Normalization of the raw inputs; null uses them as they are, so they should already lie in [0, 1]
        /// </summary>
        public InputScaling InputScaling { get; set; }

        public int TimeSteps { get; set; } = 50;

        /// <summary>
        /// Learning rule of <see cref="ISpikingNetwork.Learn"/>; null uses the default <see cref="StdpOptions"/>
        /// </summary>
        public StdpOptions Plasticity { get; set; }
    }

    public sealed class SpikingGraphLayer
    {
        /// <param name="activation">Spike function of the layer; null uses an integrate-and-fire neuron with threshold 1</param>
        /// <param name="learn">Whether STDP changes the synapses into this layer from earlier layers</param>
        public SpikingGraphLayer(int neuronsCount, ISpikingActivation activation = null, bool learn = false)
        {
            ArgumentOutOfRangeException.ThrowIfLessThan(neuronsCount, 1);

            NeuronsCount = neuronsCount;
            Activation = activation;
            Learn = learn;
        }

        public int NeuronsCount { get; }

        public ISpikingActivation Activation { get; }

        public bool Learn { get; }
    }
}
