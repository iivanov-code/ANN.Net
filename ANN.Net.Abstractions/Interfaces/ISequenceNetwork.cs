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

        /// <summary>
        /// One step of mini-batch gradient descent on several threads: the gradients of all sequences are computed with
        /// the weights before the step, averaged, and every weight is updated once. Returns the mean error. The result
        /// does not depend on the number of processors (see <see cref="Settings.BatchOptions"/>), and a batch of one
        /// sequence trains exactly like Propagate and Backpropagate.
        /// Networks that cannot train in batches throw <see cref="System.NotSupportedException"/>
        /// </summary>
        Quad TrainBatch(IReadOnlyList<Quad[][]> sequences, IReadOnlyList<Quad[][]> targets, Settings.BatchOptions options = null) =>
            throw new System.NotSupportedException($"A {GetType().Name} cannot train in batches");

        Quad[][] Values { get; }

        Quad[][] Errors { get; }

        ICollection<LearningRateOptimizer> Optimizers { get; }

        /// <summary>
        /// Size of every input vector
        /// </summary>
        int InputDimension { get; }

        /// <summary>
        /// Size of every output vector
        /// </summary>
        int OutputDimension { get; }

        /// <summary>
        /// Longest sequence the network accepts
        /// </summary>
        int MaxSequenceLength { get; }

        /// <summary>
        /// True when every output vector is a probability distribution
        /// </summary>
        bool OutputApplySoftMax { get; }
    }
}
