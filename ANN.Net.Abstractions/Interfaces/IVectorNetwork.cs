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

        /// <summary>
        /// One training step on a loss that depends on the outputs of all samples together, such as a ranking or a
        /// triplet loss: every input is propagated, <paramref name="loss"/> gives the gradient with respect to the
        /// outputs of every sample, and the weights move by the gradient of that loss (the sum over the samples, not the
        /// mean). Returns the loss before the step.
        /// Matrix and convolutional networks compute every sample with the weights before the step, on several threads
        /// (see <see cref="Settings.BatchOptions"/>), and update every weight once. The other networks apply the
        /// gradients of the samples one after the other; every sample is propagated again just before its update, so
        /// later samples see the weights already changed by earlier ones
        /// </summary>
        Quad TrainOutputs(IReadOnlyList<Quad[]> inputs, BatchLoss loss, Settings.BatchOptions options = null)
        {
            System.ArgumentNullException.ThrowIfNull(inputs);
            System.ArgumentNullException.ThrowIfNull(loss);
            _ = options;

            var outputs = new Quad[inputs.Count][];
            for (int i = 0; i < outputs.Length; i++) outputs[i] = (Quad[])Propagate(inputs[i]).Clone();

            var gradients = new Quad[outputs.Length][];
            Quad value = loss(outputs, gradients);
            for (int i = 0; i < outputs.Length; i++)
            {
                if (gradients[i] == null) throw new System.InvalidOperationException($"The loss gave no gradient for sample {i}");
                Propagate(inputs[i]);
                BackpropagateGradient(gradients[i]);
            }

            return value;
        }
    }
}
