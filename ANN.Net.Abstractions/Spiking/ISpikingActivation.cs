using System;

namespace ANN.Net.Abstractions.Spiking
{
    /// <summary>
    /// Spike function of the neurons of one spiking layer: once per time step it adds a neuron's input current to its
    /// <see cref="SpikingNeuronState"/> and decides whether the neuron fires. Every layer of a spiking network has one,
    /// and any implementation can replace another with <see cref="ISpikingNetwork.SetActivation"/>
    /// </summary>
    public interface ISpikingActivation
    {
        /// <summary>
        /// One time step of one neuron
        /// </summary>
        /// <param name="current">Sum of the weights of the synapses whose input neuron fired, plus the bias</param>
        /// <returns>1 for a spike, -1 for a negative spike, 0 for none</returns>
        int Step(ref SpikingNeuronState neuron, float current);

        /// <summary>
        /// One time step of every neuron of a layer. Implementations override it to avoid a call per neuron
        /// </summary>
        /// <param name="spikes">Receives the result of <see cref="Step(ref SpikingNeuronState, float)"/> of every neuron</param>
        /// <returns>Number of neurons that fired</returns>
        int Step(Span<SpikingNeuronState> neurons, ReadOnlySpan<float> currents, Span<sbyte> spikes)
        {
            int fired = 0;
            for (int i = 0; i < neurons.Length; i++)
            {
                int spike = Step(ref neurons[i], currents[i]);
                spikes[i] = (sbyte)spike;
                if (spike != 0) fired++;
            }

            return fired;
        }
    }
}
