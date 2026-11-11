using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Interfaces.Synapses;

namespace ANN.Net.Abstractions.Interfaces
{
    public interface ISynapses : ISynapseCollection<ISynapse>
    {
        void AccumulateError(BackpropagateEventArgs errorSignal);

        void BackpropagateForEach(BackpropagateEventArgs errorSignal);

        void AccountSignal(NeuronPropagateEventArgs value);

        void PropagateForEach(NeuronPropagateEventArgs value);
    }
}
