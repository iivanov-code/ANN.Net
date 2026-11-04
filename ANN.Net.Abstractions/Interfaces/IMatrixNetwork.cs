using System.Collections.Generic;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Layered network that keeps its weights in one matrix per layer instead of neuron and synapse objects, built by
    /// <c>NetworkFactory.BuildMatrixFFN</c> or converted with <c>NetworkFactory.ToMatrix</c>. It computes and trains
    /// exactly like the neuron-graph network of the same settings
    /// </summary>
    public interface IMatrixNetwork : INetwork, IVectorNetwork
    {
        /// <summary>
        /// Outputs for the input values; one method for <see cref="INetwork"/> and <see cref="IVectorNetwork"/>
        /// </summary>
        new Quad[] Propagate(params Quad[] values);

        /// <summary>
        /// One optimizer per weight matrix and per bias vector, to set the learning rate before training
        /// </summary>
        new ICollection<LearningRateOptimizer> Optimizers { get; }

        /// <summary>
        /// Number of weights and biases
        /// </summary>
        new long ParameterCount { get; }

        /// <summary>
        /// One step of mini-batch gradient descent on several threads: the gradients of all samples are computed with
        /// the weights before the step, averaged, and every weight is updated once. Returns the mean loss. The result
        /// does not depend on the number of processors (see <see cref="BatchOptions"/>), and a batch of one sample
        /// trains exactly like Propagate and Backpropagate
        /// </summary>
        Quad TrainBatch(IReadOnlyList<Quad[]> inputs, IReadOnlyList<Quad[]> targets, BatchOptions options = null);
    }
}
