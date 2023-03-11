using ANN.Net.Abstractions.Arguments;

namespace ANN.Net.Abstractions.Interfaces.Neurons
{
    public interface ILSTMCell : IHiddenNeuron
    {
        void Propagate(CellPropagateEventArgs value);
    }
}
