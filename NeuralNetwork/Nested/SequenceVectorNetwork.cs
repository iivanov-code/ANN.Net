using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;
using ANN.Net.Transformers;

namespace ANN.Net.Nested
{
    /// <summary>
    /// A transformer as a vector network: the input vector is a sequence of <see cref="SequenceLength"/> tokens laid
    /// one after the other, and so is the output
    /// </summary>
    internal sealed class SequenceVectorNetwork : IVectorNetwork
    {
        public SequenceVectorNetwork(TransformerNetwork network, int sequenceLength)
        {
            if (sequenceLength < 1 || sequenceLength > network.MaxSequenceLength)
                throw new ArgumentOutOfRangeException(nameof(sequenceLength), sequenceLength, $"Must be between 1 and {network.MaxSequenceLength}");

            Network = network;
            SequenceLength = sequenceLength;
        }

        public TransformerNetwork Network { get; }

        public int SequenceLength { get; }

        public int InputSize => SequenceLength * Network.InputDimension;

        public int OutputSize => SequenceLength * Network.OutputDimension;

        public long ParameterCount => Network.Optimizers.OfType<ArrayOptimizer>().Sum(x => (long)x.Length);

        public ICollection<LearningRateOptimizer> Optimizers => Network.Optimizers;

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.Length != InputSize)
                throw new ArgumentException($"Expected {InputSize} values ({SequenceLength} tokens of {Network.InputDimension}) but found {values.Length}", nameof(values));

            return Flatten(Network.Propagate(Split(values, Network.InputDimension)));
        }

        public Quad[] BackpropagateGradient(Quad[] outputGradients)
        {
            Utils.GradientUtils.CheckLength(outputGradients, OutputSize, nameof(outputGradients));
            return Flatten(Network.BackpropagateGradient(Split(outputGradients, Network.OutputDimension)));
        }

        private Quad[][] Split(Quad[] values, int dimension) =>
            Enumerable.Range(0, SequenceLength).Select(t => values.AsSpan(t * dimension, dimension).ToArray()).ToArray();

        private static Quad[] Flatten(Quad[][] sequence) => sequence.SelectMany(x => x).ToArray();
    }
}
