namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// What a spiking neuron remembers between time steps. The network keeps one per neuron in an array and hands it
    /// to the neuron's <see cref="ISpikingActivation"/> every step; the activation decides what the fields mean
    /// </summary>
    public struct SpikingNeuronState
    {
        /// <summary>
        /// Membrane potential: the input collected so far and not yet spent on spikes
        /// </summary>
        public float Potential { get; set; }

        /// <summary>
        /// Extra threshold built up by recent spikes (homeostasis). Kept when the network starts a new input, so a
        /// neuron that fires a lot stays harder to fire
        /// </summary>
        public float Adaptation { get; set; }

        /// <summary>
        /// Time steps since the network started the current input
        /// </summary>
        public int Steps { get; set; }

        /// <summary>
        /// Spikes since the network started the current input, negative spikes subtracted
        /// </summary>
        public int Spikes { get; set; }

        /// <summary>
        /// Clears everything except <see cref="Adaptation"/>
        /// </summary>
        public void Restart()
        {
            Potential = 0;
            Steps = 0;
            Spikes = 0;
        }
    }
}
