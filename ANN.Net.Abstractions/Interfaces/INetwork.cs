using System;
using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
{
    public interface INetwork
    {
        Quad[] Propagate(params Quad[] values);

        Quad Backpropagate(params Quad[] targets);

        Quad[] Values { get; }
        Quad[] Errors { get; }

        /// <summary>
        /// Gradient of the loss with respect to every input value, from the last backpropagation
        /// </summary>
        Quad[] InputGradients { get; }
        IReadOnlyList<Tuple<Quad, Quad>> MinMaxValues { get; }

        void Save(string fullDirPath, string fileName);
    }
}
