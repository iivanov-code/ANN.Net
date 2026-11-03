using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Transformers;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// 2-D convolution with zero padding; every filter has one weight per input channel and kernel position, shared by
    /// all output positions, and one bias
    /// </summary>
    [Serializable]
    internal sealed class ConvolutionLayer : IConvolutionalLayer
    {
        private readonly Activation activation;
        private Quad[] lastInput;

        public ConvolutionLayer(Shape inputShape, int filters, int kernelHeight, int kernelWidth, int stride, int padding,
                                ActivationTypes activationType, OptimizerTypes optimizerType, WeightInitType initializationType,
                                ICollection<LearningRateOptimizer> optimizers, bool useBias = true)
        {
            if (filters < 1) throw new ArgumentOutOfRangeException(nameof(filters), filters, "Must be at least 1");
            if (kernelHeight < 1 || kernelWidth < 1) throw new ArgumentOutOfRangeException(nameof(kernelHeight), "The kernel must be at least 1 x 1");
            if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride), stride, "Must be at least 1");
            if (padding < 0 || padding >= kernelHeight || padding >= kernelWidth)
                throw new ArgumentOutOfRangeException(nameof(padding), padding, "Must be at least 0 and smaller than the kernel");

            int outputHeight = ((inputShape.Height + (2 * padding) - kernelHeight) / stride) + 1;
            int outputWidth = ((inputShape.Width + (2 * padding) - kernelWidth) / stride) + 1;
            if (inputShape.Height + (2 * padding) < kernelHeight || inputShape.Width + (2 * padding) < kernelWidth)
                throw new ArgumentException($"A {kernelHeight}x{kernelWidth} kernel does not fit the {inputShape} input");

            InputShape = inputShape;
            OutputShape = new Shape(filters, outputHeight, outputWidth);
            KernelHeight = kernelHeight;
            KernelWidth = kernelWidth;
            Stride = stride;
            Padding = padding;
            activation = new Activation(activationType);

            int fanIn = inputShape.Channels * kernelHeight * kernelWidth;
            int fanOut = filters * kernelHeight * kernelWidth;
            Weights = new Parameters(filters * fanIn, optimizerType, optimizers, () => TransformerUtils.InitialWeight(initializationType, fanIn, fanOut));
            Biases = useBias ? new Parameters(filters, optimizerType, optimizers, () => 0) : null;
        }

        public Shape InputShape { get; }

        public Shape OutputShape { get; }

        public int KernelHeight { get; }

        public int KernelWidth { get; }

        public int Stride { get; }

        public int Padding { get; }

        public ActivationTypes ActivationType => activation.Type;

        /// <summary>
        /// Weights in filter, channel, kernel row, kernel column order (as in ONNX)
        /// </summary>
        public Parameters Weights { get; }

        /// <summary>
        /// One bias per filter, or null for a convolution without bias
        /// </summary>
        public Parameters Biases { get; }

        private int WeightIndex(int filter, int channel, int row, int column)
        {
            return (((((filter * InputShape.Channels) + channel) * KernelHeight) + row) * KernelWidth) + column;
        }

        public Quad[] Propagate(Quad[] input)
        {
            lastInput = input;
            Shape output = OutputShape;
            var sums = new Quad[output.Size];

            for (int f = 0; f < output.Channels; f++)
            {
                for (int oy = 0; oy < output.Height; oy++)
                {
                    for (int ox = 0; ox < output.Width; ox++)
                    {
                        Quad sum = Biases?.Values[f] ?? 0;
                        ForEachInput(oy, ox, (c, iy, ix, ky, kx) => sum += input[InputShape.Index(c, iy, ix)] * Weights.Values[WeightIndex(f, c, ky, kx)]);
                        sums[output.Index(f, oy, ox)] = sum;
                    }
                }
            }

            return activation.Apply(sums);
        }

        public Quad[] Backpropagate(Quad[] outputGradient)
        {
            if (lastInput == null) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            Quad[] gradient = activation.Backpropagate(outputGradient);
            Shape output = OutputShape;
            var inputGradient = new Quad[InputShape.Size];
            var weightGradients = new Quad[Weights.Values.Length];
            var biasGradients = new Quad[output.Channels];

            for (int f = 0; f < output.Channels; f++)
            {
                for (int oy = 0; oy < output.Height; oy++)
                {
                    for (int ox = 0; ox < output.Width; ox++)
                    {
                        Quad delta = gradient[output.Index(f, oy, ox)];
                        biasGradients[f] += delta;

                        ForEachInput(oy, ox, (c, iy, ix, ky, kx) =>
                        {
                            int inputIndex = InputShape.Index(c, iy, ix);
                            int weightIndex = WeightIndex(f, c, ky, kx);
                            weightGradients[weightIndex] += delta * lastInput[inputIndex];
                            inputGradient[inputIndex] += delta * Weights.Values[weightIndex];
                        });
                    }
                }
            }

            Weights.Update(weightGradients);
            Biases?.Update(biasGradients);
            return inputGradient;
        }

        /// <summary>
        /// Calls <paramref name="action"/> for every input value under the kernel at the output position; positions in
        /// the zero padding are skipped since they add nothing
        /// </summary>
        private void ForEachInput(int outputRow, int outputColumn, Action<int, int, int, int, int> action)
        {
            for (int c = 0; c < InputShape.Channels; c++)
            {
                for (int ky = 0; ky < KernelHeight; ky++)
                {
                    int iy = (outputRow * Stride) - Padding + ky;
                    if (iy < 0 || iy >= InputShape.Height) continue;

                    for (int kx = 0; kx < KernelWidth; kx++)
                    {
                        int ix = (outputColumn * Stride) - Padding + kx;
                        if (ix < 0 || ix >= InputShape.Width) continue;

                        action(c, iy, ix, ky, kx);
                    }
                }
            }
        }
    }
}
