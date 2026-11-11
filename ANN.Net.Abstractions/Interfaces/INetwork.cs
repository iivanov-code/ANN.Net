using System;
using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
{
    public interface INetwork
    {
        Quad[] Propagate(params Quad[] values);

        /// <summary>
        /// Trains on the targets of the last propagation with the gradient of the network's loss, and returns the loss
        /// </summary>
        Quad Backpropagate(params Quad[] targets);

        Quad[] Values { get; }
        /// <summary>
        /// Gradient of the loss with respect to every output of the last backpropagation; predicted - target for the
        /// squared loss
        /// </summary>
        Quad[] Errors { get; }

        /// <summary>
        /// Gradient of the loss with respect to every input value, from the last backpropagation
        /// </summary>
        Quad[] InputGradients { get; }
        IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues { get; }

        void Save(string fullDirPath, string fileName);
    }
}
