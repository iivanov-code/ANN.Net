using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Serialization
{
    internal sealed class ConvolutionalDocument
    {
        public int InputChannels { get; set; }

        public int InputHeight { get; set; }

        public int InputWidth { get; set; }

        public int OutputDimension { get; set; }

        public bool OutputApplySoftMax { get; set; }

        public ActivationTypes OutputActivationType { get; set; }

        public OptimizerTypes OptimizerType { get; set; }

        /// <summary>
        /// Every layer in order; the last one is the dense output layer
        /// </summary>
        public List<ConvolutionalLayerDocument> Layers { get; set; } = new List<ConvolutionalLayerDocument>();
    }

    internal enum ConvolutionalLayerType
    {
        Convolution = 0,
        Pooling = 1,
        Dense = 2
    }

    internal sealed class ConvolutionalLayerDocument
    {
        public ConvolutionalLayerType Type { get; set; }

        public ActivationTypes? Activation { get; set; }

        // convolution
        public int? Filters { get; set; }

        public int? KernelHeight { get; set; }

        public int? KernelWidth { get; set; }

        public int? Padding { get; set; }

        // convolution and pooling
        public int? Stride { get; set; }

        // pooling
        public PoolingType? Pooling { get; set; }

        public int? Size { get; set; }

        // dense
        public int? NeuronsCount { get; set; }

        public ParametersDocument Weights { get; set; }

        public ParametersDocument Biases { get; set; }
    }

    internal sealed class ParametersDocument
    {
        public float[] Values { get; set; }

        /// <summary>
        /// Learning rate shared by every value, or null when <see cref="LearningRates"/> holds one per value
        /// </summary>
        public float? LearningRate { get; set; }

        public float[] LearningRates { get; set; }
    }
}
