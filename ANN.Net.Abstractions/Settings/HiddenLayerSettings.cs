using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    public class HiddenLayerSettings : CommonLayerSettings
    {
        public ushort LayersCount { get; set; }
        public LayerConnectionType PrevLayerConnectionType { get; set; }

        /// <summary>
        /// Kernel length of <see cref="LayerConnectionType.Convolutional"/> and <see cref="LayerConnectionType.Deconvolutional"/>
        /// layers, which treat the neurons of the previous layer (without its bias neuron) as a 1-D signal. For images use
        /// <c>NetworkFactory.BuildConvolutional</c>
        /// </summary>
        public int KernelSize { get; set; } = 3;

        /// <summary>
        /// Step of the kernel (convolution) or between the kernels of neighbouring inputs (deconvolution)
        /// </summary>
        public int Stride { get; set; } = 1;

        /// <summary>
        /// Zeros added on both sides of the signal (convolution) or removed from both sides of the output
        /// (deconvolution); must be smaller than <see cref="KernelSize"/>
        /// </summary>
        public int Padding { get; set; }

        /// <summary>
        /// Number of kernels; every kernel produces its own output signal. The neurons count of a convolutional layer
        /// follows from the signal length, kernel, stride, padding and this number, so
        /// <see cref="CommonLayerSettings.NeuronsCount"/> is ignored
        /// </summary>
        public int FiltersCount { get; set; } = 1;
    }
}
