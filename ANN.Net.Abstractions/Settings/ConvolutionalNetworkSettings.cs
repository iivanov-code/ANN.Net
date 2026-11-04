using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    /// <summary>
    /// Convolutional network for images of <see cref="InputChannels"/> x <see cref="InputHeight"/> x <see cref="InputWidth"/>
    /// values. Inputs are flat arrays in channel, row, column order (channel-major, as in NCHW). The layers run in order;
    /// after the first dense layer only dense layers may follow. A dense output layer with
    /// <see cref="OutputDimension"/> neurons always comes last
    /// </summary>
    public class ConvolutionalNetworkSettings
    {
        public int InputChannels { get; set; } = 1;

        public int InputHeight { get; set; }

        public int InputWidth { get; set; }

        public IList<ConvolutionalLayerSettings> Layers { get; set; } = new List<ConvolutionalLayerSettings>();

        public int OutputDimension { get; set; }

        /// <summary>
        /// Activation of the output layer; ignored when <see cref="OutputApplySoftMax"/> is set
        /// </summary>
        public ActivationTypes OutputActivationType { get; set; } = ActivationTypes.Identity;

        /// <summary>
        /// Soft-max over the outputs, trained with cross entropy; otherwise the squared error is used
        /// </summary>
        public bool OutputApplySoftMax { get; set; }

        public OptimizerTypes OptimizerType { get; set; } = OptimizerTypes.Normal;

        public WeightInitType InitializationType { get; set; } = WeightInitType.Xavier;

        /// <summary>
        /// Seed of the initial weights; the same seed builds the same network. Null draws them from a shared random source
        /// </summary>
        public int? Seed { get; set; }
    }

    /// <summary>
    /// One layer of a <see cref="ConvolutionalNetworkSettings"/>
    /// </summary>
    public abstract class ConvolutionalLayerSettings
    {
    }

    /// <summary>
    /// 2-D convolution: <see cref="Filters"/> kernels of <see cref="KernelHeight"/> x <see cref="KernelWidth"/> over all
    /// input channels, moved by <see cref="Stride"/>, over the input surrounded by <see cref="Padding"/> zeros
    /// </summary>
    public class ConvolutionLayerSettings : ConvolutionalLayerSettings
    {
        public int Filters { get; set; } = 1;

        public int KernelHeight { get; set; } = 3;

        public int KernelWidth { get; set; } = 3;

        /// <summary>
        /// Sets <see cref="KernelHeight"/> and <see cref="KernelWidth"/> to the same value
        /// </summary>
        public int KernelSize
        {
            set
            {
                KernelHeight = value;
                KernelWidth = value;
            }
        }

        public int Stride { get; set; } = 1;

        public int Padding { get; set; }

        public ActivationTypes ActivationType { get; set; } = ActivationTypes.RELU;

        /// <summary>
        /// Whether every filter has a bias; convolutions followed by batch normalization usually have none
        /// </summary>
        public bool UseBias { get; set; } = true;
    }

    /// <summary>
    /// Max or average over windows of <see cref="Size"/> x <see cref="Size"/> of every channel
    /// </summary>
    public class PoolingLayerSettings : ConvolutionalLayerSettings
    {
        public PoolingType Type { get; set; } = PoolingType.Max;

        public int Size { get; set; } = 2;

        /// <summary>
        /// Step between windows; 0 means <see cref="Size"/> (windows side by side)
        /// </summary>
        public int Stride { get; set; }

        /// <summary>
        /// Positions added around the input that windows may cover but that never count: max pooling ignores them and
        /// average pooling divides by the real positions only. Must be smaller than <see cref="Size"/>
        /// </summary>
        public int Padding { get; set; }
    }

    /// <summary>
    /// Batch normalization of every channel followed by an activation: values are normalized with the mean and variance
    /// of the channel, then scaled and shifted by a learned gamma and beta. While the network is training the statistics
    /// of the current image are used and running averages are updated; otherwise the running averages are used
    /// </summary>
    public class BatchNormalizationLayerSettings : ConvolutionalLayerSettings
    {
        public ActivationTypes ActivationType { get; set; } = ActivationTypes.Identity;

        /// <summary>
        /// Weight of the newest statistics in the running averages
        /// </summary>
        public float Momentum { get; set; } = 0.1f;

        public float Epsilon { get; set; } = 1e-5f;
    }

    /// <summary>
    /// Average of every channel: channels x height x width become channels x 1 x 1
    /// </summary>
    public class GlobalAveragePoolingLayerSettings : ConvolutionalLayerSettings
    {
    }

    /// <summary>
    /// Residual block of ResNet-18/34 ("basic block"): 3x3 convolution with <see cref="Stride"/>, batch normalization
    /// and activation, 3x3 convolution and batch normalization, added to the input and activated. When the stride or the
    /// number of channels changes, the input goes through a 1x1 convolution with batch normalization first. The
    /// convolutions have no bias
    /// </summary>
    public class ResidualBlockSettings : ConvolutionalLayerSettings
    {
        public int Filters { get; set; }

        public int Stride { get; set; } = 1;

        public ActivationTypes ActivationType { get; set; } = ActivationTypes.RELU;
    }

    /// <summary>
    /// Fully connected layer over all values of the previous layer
    /// </summary>
    public class DenseLayerSettings : ConvolutionalLayerSettings
    {
        public int NeuronsCount { get; set; }

        public ActivationTypes ActivationType { get; set; } = ActivationTypes.RELU;
    }
}
