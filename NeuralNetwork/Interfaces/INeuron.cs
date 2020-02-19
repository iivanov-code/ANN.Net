using System;

namespace NeuralNetwork.Interfaces
{
    public interface INeuron
    {
#if DEBUG
        Guid ID { get; }
#endif

        /// <summary>
        /// Propagate new signal value
        /// </summary>
        /// <param name="value"></param>
        void Propagate(float value);
        ISynapseCollection<ISynapse> Outputs { get; }
        ISynapseCollection<ISynapse> Inputs { get; }
        /// <summary>
        /// BackpropagateError
        /// </summary>
        /// <param name="eWeightedSignal">ERROR SIGNAL multiplied by WEIGHT</param>
        /// <param name="errorSignal">ERROR SIGNAL ONLY</param>
        /// <param name="output">Axon from which the error came from</param>
        void Backpropagate(float errorSignal, float eWeightedSignal = 0, Action<float> updateWeight = null);

    }
}
