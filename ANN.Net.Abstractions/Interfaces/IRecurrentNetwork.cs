using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Recurrent sequence network built by <c>NetworkFactory.BuildRecurrent</c>. Every call of
    /// <see cref="ISequenceNetwork.Propagate"/> starts from a zero state, and <see cref="ISequenceNetwork.Backpropagate"/>
    /// trains through the whole sequence (backpropagation through time). A target vector may be null for steps
    /// whose output does not matter, such as every step but the last when a sequence is classified.
    /// <see cref="Step"/> runs one step at a time and keeps the state between calls, for inputs that arrive one by one.
    /// A network keeps what it computed last, so it must not be used by several threads at once
    /// </summary>
    public interface IRecurrentNetwork : ISequenceNetwork
    {
        NeuronType CellType { get; }

        int HiddenDimension { get; }

        int LayersCount { get; }

        /// <summary>
        /// Number of weights and biases
        /// </summary>
        long ParameterCount { get; }

        /// <summary>
        /// One step from the state the previous step left; the first step after <see cref="ResetState"/> starts from zero.
        /// Gives what <see cref="ISequenceNetwork.Propagate"/> gives at the same position of the sequence
        /// </summary>
        Quad[] Step(Quad[] input);

        /// <summary>
        /// Sets the state <see cref="Step"/> continues from back to zero
        /// </summary>
        void ResetState();
    }
}
