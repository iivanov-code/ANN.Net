using System.Collections.Generic;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Convolutional network built by <c>NetworkFactory.BuildConvolutional</c>. <see cref="INetwork.Propagate"/> takes one
    /// image as a flat array of <see cref="InputChannels"/> x <see cref="InputHeight"/> x <see cref="InputWidth"/> values in
    /// channel, row, column order; <see cref="INetwork.Backpropagate"/> trains on that image and returns its loss
    /// </summary>
    public interface IConvolutionalNetwork : INetwork
    {
        /// <summary>
        /// Optimizers of the weights and biases, one per weight or bias array of a layer, to set the learning rate
        /// before training
        /// </summary>
        ICollection<LearningRateOptimizer> Optimizers { get; }

        /// <summary>
        /// Number of trainable values (weights, biases, batch normalization scales and shifts)
        /// </summary>
        long ParameterCount { get; }

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
    }
}
