namespace ANN.Net.Abstractions.Interfaces.Neurons
{
    public interface IInputNeuron : INeuron
    {
        /// <summary>
        /// Gradient of the loss with respect to the raw input value, from the last backpropagation
        /// </summary>
        Quad Gradient { get; }
    }
}
