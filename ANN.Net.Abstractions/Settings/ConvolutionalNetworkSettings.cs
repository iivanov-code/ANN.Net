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
