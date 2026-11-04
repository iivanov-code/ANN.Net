using System.Collections.Generic;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Network that maps a vector to a vector and can pass a gradient back through itself, so it can sit inside a node
    /// of a nested network (<c>NetworkFactory.BuildNested</c>) and be trained there. Matrix, neuron, graph,
    /// convolutional and nested networks are vector networks; <c>NetworkFactory.AsVectorNetwork</c> turns a
    /// transformer into one
    /// </summary>
    public interface IVectorNetwork
    {
        int InputSize { get; }

        int OutputSize { get; }

        /// <summary>
        /// Number of trainable values
        /// </summary>
        long ParameterCount { get; }

        ICollection<LearningRateOptimizer> Optimizers { get; }

        Quad[] Propagate(params Quad[] values);

        /// <summary>
        /// Takes the gradient of the loss with respect to the outputs of the last propagation, updates the weights and
        /// returns the gradient with respect to its inputs
        /// </summary>
        Quad[] BackpropagateGradient(Quad[] outputGradients);
    }
}
