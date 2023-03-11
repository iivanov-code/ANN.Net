using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces.Neurons;

namespace ANN.Net.Abstractions.Interfaces.Synapses
{
    public interface ISynapse
    {
#if DEBUG
        INeuron Input { get; }
        INeuron Output { get; }
#endif

        void Propagate(NeuronPropagateEventArgs value);

        void Backpropagate(BackpropagateEventArgs errorSignal);
    }
}
