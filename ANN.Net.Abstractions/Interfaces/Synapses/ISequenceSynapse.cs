using ANN.Net.Abstractions.Arguments;

namespace ANN.Net.Abstractions.Interfaces.Synapses
{
    public interface ISequenceSynapse : ISynapse
    {
        void Propagate(CellPropagateEventArgs values);
    }
}
