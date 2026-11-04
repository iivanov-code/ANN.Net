using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Inference-only copy of a trained sequence network (a transformer), built by <c>NetworkCompactor.CompactSequence</c>.
    /// It computes what the original computes, keeps no state between calls and can be used from several threads at once
    /// </summary>
    public interface ICompactSequenceNetwork
    {
        int InputDimension { get; }

        int OutputDimension { get; }

        int MaxSequenceLength { get; }

        bool OutputApplySoftMax { get; }

        /// <summary>
        /// Number of weights, biases and normalization values
        /// </summary>
        long ParameterCount { get; }

        WeightPrecision Precision { get; }

        /// <summary>
        /// Bytes taken by the stored values, including the positional encodings
        /// </summary>
        long SizeInBytes { get; }

        /// <summary>
        /// One output vector for every input vector
        /// </summary>
        Quad[][] Propagate(Quad[][] sequence);

        /// <summary>
        /// Outputs of many sequences, run on several threads
        /// </summary>
        float[][][] PropagateBatch(IReadOnlyList<float[][]> sequences);
    }
}
