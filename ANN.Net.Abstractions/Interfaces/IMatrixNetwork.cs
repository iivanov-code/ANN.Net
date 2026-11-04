using System.Collections.Generic;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Layered network that keeps its weights in one matrix per layer instead of neuron and synapse objects, built by
    /// <c>NetworkFactory.BuildMatrixFFN</c> or converted with <c>NetworkFactory.ToMatrix</c>. It computes and trains
    /// exactly like the neuron-graph network of the same settings
    /// </summary>
    public interface IMatrixNetwork : INetwork
    {
        /// <summary>
        /// One optimizer per weight matrix and per bias vector, to set the learning rate before training
        /// </summary>
        ICollection<LearningRateOptimizer> Optimizers { get; }

        /// <summary>
        /// Number of weights and biases
        /// </summary>
        long ParameterCount { get; }
    }
}
