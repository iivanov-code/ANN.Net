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

        /// <summary>
        /// Output of the last propagation, after activation
        /// </summary>
        Quad Value { get; }

        ISynapses Outputs { get; }
        ISynapses Inputs { get; }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="errorSignal"></param>
        void Backpropagate(BackpropagateEventArgs errorSignal);
    }
}
