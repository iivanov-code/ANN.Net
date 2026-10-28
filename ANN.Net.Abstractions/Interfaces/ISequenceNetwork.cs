using System.Collections.Generic;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Network that maps a sequence of vectors to a sequence of vectors of the same length
    /// </summary>
    public interface ISequenceNetwork
    {
        /// <summary>
        /// Propagate a whole sequence; one output vector is produced for every input vector
        /// </summary>
        Quad[][] Propagate(Quad[][] sequence);

        /// <summary>
        /// Backpropagate the error against the targets of the last propagated sequence
        /// and update the weights. Returns the total error
        /// </summary>
        Quad Backpropagate(Quad[][] targets);

        Quad[][] Values { get; }

        Quad[][] Errors { get; }

        ICollection<LearningRateOptimizer> Optimizers { get; }
    }
}
