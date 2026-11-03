using System;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Max or average pooling of every channel over windows of size x size. Padding positions never count: max pooling
    /// ignores them and average pooling divides by the number of real positions in the window
    /// </summary>
    [Serializable]
    internal sealed class PoolingLayer : IConvolutionalLayer
    {
        private int[] maxIndexes;

        public PoolingLayer(Shape inputShape, PoolingType type, int size, int stride, int padding = 0)
        {
            if (size < 1) throw new ArgumentOutOfRangeException(nameof(size), size, "Must be at least 1");
            if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride), stride, "Must be at least 1");
            if (padding < 0 || padding >= size) throw new ArgumentOutOfRangeException(nameof(padding), padding, "Must be at least 0 and smaller than the window");
            if (size > inputShape.Height + (2 * padding) || size > inputShape.Width + (2 * padding))
                throw new ArgumentException($"A {size}x{size} pooling window does not fit the {inputShape} input");

            InputShape = inputShape;
            OutputShape = new Shape(inputShape.Channels,
                                    ((inputShape.Height + (2 * padding) - size) / stride) + 1,
                                    ((inputShape.Width + (2 * padding) - size) / stride) + 1);
            Type = type;
            Size = size;
            Stride = stride;
            Padding = padding;
        }

        public Shape InputShape { get; }

        public Shape OutputShape { get; }

        public PoolingType Type { get; }

        public int Size { get; }

        public int Stride { get; }

        public int Padding { get; }

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
                        int maxIndex = -1, count = 0;

                        foreach (int inputIndex in Window(c, oy, ox))
                        {
                            Quad value = input[inputIndex];
                            sum += value;
                            count++;
                            if (maxIndex < 0 || value > max)
                            {
                                max = value;
                                maxIndex = inputIndex;
                            }
                        }

                        maxIndexes[outputIndex] = maxIndex;
                        values[outputIndex] = Type == PoolingType.Max ? max : sum / count;
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

                        var window = new System.Collections.Generic.List<int>(Window(c, oy, ox));
                        Quad share = outputGradient[outputIndex] / window.Count;
                        foreach (int inputIndex in window)
                        {
                            inputGradient[inputIndex] += share;
                        }
                    }
                }
            }

            return inputGradient;
        }

        /// <summary>
        /// Input positions under the window at the output position, without the padding
        /// </summary>
        private System.Collections.Generic.IEnumerable<int> Window(int channel, int outputRow, int outputColumn)
        {
            for (int ky = 0; ky < Size; ky++)
            {
                int iy = (outputRow * Stride) - Padding + ky;
                if (iy < 0 || iy >= InputShape.Height) continue;

                for (int kx = 0; kx < Size; kx++)
                {
                    int ix = (outputColumn * Stride) - Padding + kx;
                    if (ix >= 0 && ix < InputShape.Width) yield return InputShape.Index(channel, iy, ix);
                }
            }
        }
    }
}
