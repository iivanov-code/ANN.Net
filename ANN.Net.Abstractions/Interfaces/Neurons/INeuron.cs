using ANN.Net.Abstractions.Arguments;

namespace ANN.Net.Abstractions.Interfaces.Neurons
{
    public interface INeuron
    {
        uint ID { get; }

        /// <summary>
        /// Propagate new signal value
        /// </summary>
        /// <param name="value"></param>
        void Propagate(NeuronPropagateEventArgs value);

        ISynapses Outputs { get; }
        ISynapses Inputs { get; }

        /// <summary>
        /// BackpropagateError
        /// </summary>
        /// <param name="eWeightedSignal">ERROR SIGNAL multiplied by WEIGHT</param>
        /// <param name="errorSignal">ERROR SIGNAL ONLY</param>
        /// <param name="output">Axon from which the error came from</param>
        void Backpropagate(BackpropagateEventArgs errorSignal);
    }
}
