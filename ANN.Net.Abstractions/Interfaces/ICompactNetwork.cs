using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Inference-only copy of a trained network that keeps nothing but its weights in flat arrays, built by
    /// <c>NetworkCompactor.Compact</c>. It computes what the original computes (convolutional networks as with
    /// <see cref="IConvolutionalNetwork.Training"/> set to false), keeps no state between calls and can therefore be
    /// used from several threads at once
    /// </summary>
    public interface ICompactNetwork
    {
        int InputCount { get; }

        int OutputCount { get; }

        /// <summary>
        /// Number of weights, biases and other stored values
        /// </summary>
        long ParameterCount { get; }

        /// <summary>
        /// Number format of the weight matrices
        /// </summary>
        WeightPrecision Precision { get; }

        /// <summary>
        /// Bytes taken by the stored weights and other values, without the few objects around them
        /// </summary>
        long SizeInBytes { get; }

        Quad[] Propagate(params Quad[] values);
    }
}
