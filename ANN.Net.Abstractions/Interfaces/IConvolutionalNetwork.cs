using System.Collections.Generic;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Convolutional network built by <c>NetworkFactory.BuildConvolutional</c>. <see cref="INetwork.Propagate"/> takes one
    /// image as a flat array of <see cref="InputChannels"/> x <see cref="InputHeight"/> x <see cref="InputWidth"/> values in
    /// channel, row, column order; <see cref="INetwork.Backpropagate"/> trains on that image and returns its loss
    /// </summary>
    public interface IConvolutionalNetwork : INetwork, IVectorNetwork
    {
        /// <summary>
        /// Outputs for the input values; one method for <see cref="INetwork"/> and <see cref="IVectorNetwork"/>
        /// </summary>
        new Quad[] Propagate(params Quad[] values);

        /// <summary>
        /// Optimizers of the weights and biases, one per weight or bias array of a layer, to set the learning rate
        /// before training
        /// </summary>
        new ICollection<LearningRateOptimizer> Optimizers { get; }

        /// <summary>
        /// Number of trainable values (weights, biases, batch normalization scales and shifts)
        /// </summary>
        new long ParameterCount { get; }

        int InputChannels { get; }

        int InputHeight { get; }

        int InputWidth { get; }

        int OutputDimension { get; }

        bool OutputApplySoftMax { get; }

        /// <summary>
        /// Training mode (the default): batch normalization uses the statistics of the current image and updates its
        /// running averages. Set to false for inference, where the running averages are used, as in an ONNX export
        /// </summary>
        bool Training { get; set; }

        /// <summary>
        /// One step of mini-batch gradient descent on several threads: the gradients of all images are computed with
        /// the weights before the step, averaged, and every weight is updated once. Returns the mean loss. The result
        /// does not depend on the number of processors (see <see cref="BatchOptions"/>), and a batch of one image trains
        /// exactly like Propagate and Backpropagate
        /// </summary>
        Quad TrainBatch(IReadOnlyList<Quad[]> inputs, IReadOnlyList<Quad[]> targets, BatchOptions options = null);
    }
}
