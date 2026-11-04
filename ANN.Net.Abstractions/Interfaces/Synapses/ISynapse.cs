using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Abstractions.Interfaces.Synapses
{
    public interface ISynapse
    {
        /// <summary>
        /// Neuron the synapse takes its value from
        /// </summary>
        INeuron Input { get; }

        /// <summary>
        /// Neuron the synapse delivers its value to
        /// </summary>
        INeuron Output { get; }

        void Propagate(NeuronPropagateEventArgs value);

        void Backpropagate(BackpropagateEventArgs errorSignal);
    }
}
