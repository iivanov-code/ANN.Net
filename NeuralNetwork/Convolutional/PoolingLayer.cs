using System;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Max or average pooling of every channel over windows of size x size, without padding
    /// </summary>
    [Serializable]
    internal sealed class PoolingLayer : IConvolutionalLayer
    {
        private int[] maxIndexes;

        public PoolingLayer(Shape inputShape, PoolingType type, int size, int stride)
        {
            if (size < 1) throw new ArgumentOutOfRangeException(nameof(size), size, "Must be at least 1");
            if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride), stride, "Must be at least 1");
            if (size > inputShape.Height || size > inputShape.Width)
                throw new ArgumentException($"A {size}x{size} pooling window does not fit the {inputShape} input");

            InputShape = inputShape;
            OutputShape = new Shape(inputShape.Channels, ((inputShape.Height - size) / stride) + 1, ((inputShape.Width - size) / stride) + 1);
            Type = type;
            Size = size;
            Stride = stride;
        }

        public Shape InputShape { get; }

        public Shape OutputShape { get; }

        public PoolingType Type { get; }

        public int Size { get; }

        public int Stride { get; }

        public Quad[] Propagate(Quad[] input)
        {
            Shape output = OutputShape;
            var values = new Quad[output.Size];
            maxIndexes = new int[output.Size];

            for (int c = 0; c < output.Channels; c++)
            {
                for (int oy = 0; oy < output.Height; oy++)
                {
                    for (int ox = 0; ox < output.Width; ox++)
                    {
                        int outputIndex = output.Index(c, oy, ox);
                        Quad max = Quad.NegativeInfinity, sum = 0;
                        int maxIndex = -1;

                        for (int ky = 0; ky < Size; ky++)
                        {
                            for (int kx = 0; kx < Size; kx++)
                            {
                                int inputIndex = InputShape.Index(c, (oy * Stride) + ky, (ox * Stride) + kx);
                                Quad value = input[inputIndex];
                                sum += value;
                                if (maxIndex < 0 || value > max)
                                {
                                    max = value;
                                    maxIndex = inputIndex;
                                }
                            }
                        }

                        maxIndexes[outputIndex] = maxIndex;
                        values[outputIndex] = Type == PoolingType.Max ? max : sum / (Size * Size);
                    }
                }
            }

            return values;
        }

        public Quad[] Backpropagate(Quad[] outputGradient)
        {
            if (maxIndexes == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            Shape output = OutputShape;
            var inputGradient = new Quad[InputShape.Size];

            for (int c = 0; c < output.Channels; c++)
            {
                for (int oy = 0; oy < output.Height; oy++)
                {
                    for (int ox = 0; ox < output.Width; ox++)
                    {
                        int outputIndex = output.Index(c, oy, ox);
                        if (Type == PoolingType.Max)
                        {
                            // only the maximum influenced the output
                            inputGradient[maxIndexes[outputIndex]] += outputGradient[outputIndex];
                            continue;
                        }

                        Quad share = outputGradient[outputIndex] / (Size * Size);
                        for (int ky = 0; ky < Size; ky++)
                        {
                            for (int kx = 0; kx < Size; kx++)
                            {
                                inputGradient[InputShape.Index(c, (oy * Stride) + ky, (ox * Stride) + kx)] += share;
                            }
                        }
                    }
                }
            }

            return inputGradient;
        }
    }
}
