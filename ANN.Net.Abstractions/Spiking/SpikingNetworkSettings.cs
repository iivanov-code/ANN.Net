using System.Collections.Generic;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// A layered spiking network built from scratch, for instance to be trained with <see cref="StdpOptions"/>.
    /// Every layer is fully connected (or randomly, see <see cref="SpikingLayerSettings.ConnectionProbability"/>) to
    /// the layer before it; the output is the firing rate of every neuron of the last layer
    /// </summary>
    public sealed class SpikingNetworkSettings
    {
        public int InputCount { get; set; }

        /// <summary>
        /// Normalization of the raw inputs; null uses them as they are, so they should already lie in [0, 1]
        /// </summary>
        public InputScaling InputScaling { get; set; }

        /// <summary>
        /// Turns every normalized input into spikes, the input being the neuron's current in every time step.
        /// Null uses an integrate-and-fire neuron with threshold 1: an input of 0.25 fires every fourth step
        /// </summary>
        public ISpikingActivation InputActivation { get; set; }

        public IList<SpikingLayerSettings> Layers { get; } = new List<SpikingLayerSettings>();

        /// <summary>
        /// Time steps every input is presented for
        /// </summary>
        public int TimeSteps { get; set; } = 50;

        /// <summary>
        /// Seed of the initial weights and random connections
        /// </summary>
        public int Seed { get; set; }

        /// <summary>
        /// Learning rule used by <see cref="ISpikingNetwork.Learn"/>; null uses the default <see cref="StdpOptions"/>
        /// </summary>
        public StdpOptions Plasticity { get; set; }
    }

    public sealed class SpikingLayerSettings
    {
        public int NeuronsCount { get; set; }

        /// <summary>
        /// Spike function of the layer; null uses a leaky integrate-and-fire neuron
        /// </summary>
        public ISpikingActivation Activation { get; set; }

        /// <summary>
        /// Initial weights are drawn uniformly from [<see cref="MinimumWeight"/>, <see cref="MaximumWeight"/>]
        /// </summary>
        public float MinimumWeight { get; set; }

        public float MaximumWeight { get; set; } = 1;

        /// <summary>
        /// Probability of every possible synapse from the previous layer; 1 connects every pair
        /// </summary>
        public double ConnectionProbability { get; set; } = 1;

        /// <summary>
        /// When greater than 0, every neuron of the layer inhibits every other one by this weight in the next time
        /// step: the first neurons to fire silence the rest (winner takes all), so neurons learn different patterns
        /// </summary>
        public float LateralInhibition { get; set; }

        /// <summary>
        /// Whether <see cref="ISpikingNetwork.Learn"/> changes the synapses from the previous layer
        /// </summary>
        public bool Learn { get; set; } = true;
    }
}
