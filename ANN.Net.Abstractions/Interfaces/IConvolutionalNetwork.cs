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
        /// Optimizers of every weight and bias, to set the learning rate before training
        /// </summary>
        ICollection<LearningRateOptimizer> Optimizers { get; }

        int InputChannels { get; }

        int InputHeight { get; }

        int InputWidth { get; }

        int OutputDimension { get; }

        bool OutputApplySoftMax { get; }
    }
}
