namespace ANN.Net.Abstractions.Interfaces.Neurons
{
    public interface IOutputNeuron : INeuron
    {
        Quad Value { get; }
    }
}
