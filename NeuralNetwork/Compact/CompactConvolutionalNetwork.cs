using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Convolutional;
using ANN.Net.Transformers;
using ANN.Net.Utils;

namespace ANN.Net.Compact
{
    /// <summary>
    /// Inference copy of a convolutional network as plain array operations. Batch normalization uses the running
    /// averages, and a batch normalization that directly follows a convolution without activation is folded into the
    /// convolution's weights and bias, so it disappears
    /// </summary>
    internal sealed class CompactConvolutionalNetwork : ICompactNetwork
    {
        private readonly IOperation[] operations;
        private readonly bool softMax;

        public CompactConvolutionalNetwork(ConvolutionalNetwork network)
        {
            operations = Compile(network.Layers).ToArray();
            softMax = network.OutputApplySoftMax;
            InputCount = network.InputChannels * network.InputHeight * network.InputWidth;
            OutputCount = network.OutputDimension;
            ParameterCount = operations.Sum(x => x.ParameterCount);
        }

        public int InputCount { get; }

        public int OutputCount { get; }

        public long ParameterCount { get; }

        /// <summary>
        /// Operations in order; batch normalizations folded into convolutions are gone
        /// </summary>
        internal int OperationCount => operations.Sum(x => x.Count);

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} input values but found {values.Length}", nameof(values));

            float[] current = values.Select(x => (float)x).ToArray();
            foreach (IOperation operation in operations)
            {
                current = operation.Run(current);
            }

            Quad[] outputs = current.Select(x => (Quad)x).ToArray();
            if (softMax) TransformerUtils.SoftMax(outputs);
            return outputs;
        }

        private static List<IOperation> Compile(IReadOnlyList<IConvolutionalLayer> layers)
        {
            var operations = new List<IOperation>();
            for (int i = 0; i < layers.Count; i++)
            {
                switch (layers[i])
                {
                    case ConvolutionLayer convolution when convolution.ActivationType == ActivationTypes.Identity
                                                        && i + 1 < layers.Count && layers[i + 1] is BatchNormalizationLayer normalization:
                        operations.Add(Convolution.Folded(convolution, normalization));
                        i++;
                        break;

                    case ConvolutionLayer convolution:
                        operations.Add(new Convolution(convolution));
                        break;

                    case BatchNormalizationLayer normalization:
                        operations.Add(new ScaleShift(normalization));
                        break;

                    case PoolingLayer pooling:
                        operations.Add(new Pooling(pooling));
                        break;

                    case GlobalAveragePoolingLayer pooling:
                        operations.Add(new GlobalAverage(pooling.InputShape));
                        break;

                    case ResidualBlockLayer residual:
                        operations.Add(new Residual(Compile(residual.Main), Compile(residual.Shortcut), residual.ActivationType));
                        break;

                    case DenseLayer dense:
                        operations.Add(new Dense(dense));
                        break;

                    default:
                        throw new NotSupportedException($"A {layers[i].GetType().Name} cannot be compacted");
                }
            }

            return operations;
        }

        private static float[] Floats(Parameters parameters) => parameters?.Values.Select(x => (float)x).ToArray();

        private static void Activate(IActivationFunction function, float[] values)
        {
            if (function == null) return;

            for (int i = 0; i < values.Length; i++)
            {
                Quad value = values[i];
                values[i] = function.Activation(ref value);
            }
        }

        private static IActivationFunction Function(ActivationTypes type) => type == ActivationTypes.Identity ? null : NetworkUtils.GetActivation(type);

        private interface IOperation
        {
            long ParameterCount { get; }

            /// <summary>
            /// Number of operations, counting those inside a residual block
            /// </summary>
            int Count { get; }

            float[] Run(float[] input);
        }

        private sealed class Convolution : IOperation
        {
            private readonly Shape input, output;
            private readonly int kernelHeight, kernelWidth, stride, padding;
            private readonly float[] weights;
            private readonly float[] bias;
            private readonly IActivationFunction activation;

            public Convolution(ConvolutionLayer layer)
                : this(layer, Floats(layer.Weights), Floats(layer.Biases), layer.ActivationType)
            {
            }

            private Convolution(ConvolutionLayer layer, float[] weights, float[] bias, ActivationTypes activationType)
            {
                input = layer.InputShape;
                output = layer.OutputShape;
                kernelHeight = layer.KernelHeight;
                kernelWidth = layer.KernelWidth;
                stride = layer.Stride;
                padding = layer.Padding;
                this.weights = weights;
                this.bias = bias;
                activation = Function(activationType);
            }

            /// <summary>
            /// y = gamma * (conv(x) - mean) / sqrt(var + eps) + beta becomes one convolution with weights scaled per filter
            /// </summary>
            public static Convolution Folded(ConvolutionLayer convolution, BatchNormalizationLayer normalization)
            {
                float[] weights = Floats(convolution.Weights);
                float[] bias = new float[convolution.OutputShape.Channels];
                int perFilter = weights.Length / bias.Length;

                for (int f = 0; f < bias.Length; f++)
                {
                    double scale = (float)normalization.Gamma.Values[f] / Math.Sqrt((float)normalization.RunningVariance[f] + normalization.Epsilon);
                    for (int i = f * perFilter; i < (f + 1) * perFilter; i++) weights[i] = (float)(weights[i] * scale);

                    double shifted = (convolution.Biases == null ? 0 : (float)convolution.Biases.Values[f]) - (float)normalization.RunningMean[f];
                    bias[f] = (float)((shifted * scale) + (float)normalization.Beta.Values[f]);
                }

                return new Convolution(convolution, weights, bias, normalization.ActivationType);
            }

            public long ParameterCount => weights.Length + (bias?.Length ?? 0);

            public int Count => 1;

            public float[] Run(float[] values)
            {
                var result = new float[output.Size];
                int channels = input.Channels, height = input.Height, width = input.Width;
                int kernelSize = kernelHeight * kernelWidth;

                for (int f = 0; f < output.Channels; f++)
                {
                    float filterBias = bias?[f] ?? 0;
                    int filterStart = f * channels * kernelSize;

                    for (int oy = 0; oy < output.Height; oy++)
                    {
                        int top = (oy * stride) - padding;
                        for (int ox = 0; ox < output.Width; ox++)
                        {
                            int left = (ox * stride) - padding;
                            float sum = filterBias;

                            for (int c = 0; c < channels; c++)
                            {
                                int channelStart = c * height * width;
                                int kernelStart = filterStart + (c * kernelSize);

                                for (int ky = 0; ky < kernelHeight; ky++)
                                {
                                    int iy = top + ky;
                                    if (iy < 0 || iy >= height) continue;

                                    int rowStart = channelStart + (iy * width);
                                    int kernelRow = kernelStart + (ky * kernelWidth);
                                    for (int kx = 0; kx < kernelWidth; kx++)
                                    {
                                        int ix = left + kx;
                                        if (ix >= 0 && ix < width) sum += values[rowStart + ix] * weights[kernelRow + kx];
                                    }
                                }
                            }

                            result[output.Index(f, oy, ox)] = sum;
                        }
                    }
                }

                Activate(activation, result);
                return result;
            }
        }

        private sealed class ScaleShift : IOperation
        {
            private readonly int positions;
            private readonly float[] scale, shift;
            private readonly IActivationFunction activation;

            public ScaleShift(BatchNormalizationLayer layer)
            {
                int channels = layer.InputShape.Channels;
                positions = layer.InputShape.Height * layer.InputShape.Width;
                scale = new float[channels];
                shift = new float[channels];
                for (int c = 0; c < channels; c++)
                {
                    double inverse = 1 / Math.Sqrt((float)layer.RunningVariance[c] + layer.Epsilon);
                    scale[c] = (float)((float)layer.Gamma.Values[c] * inverse);
                    shift[c] = (float)((float)layer.Beta.Values[c] - ((float)layer.RunningMean[c] * scale[c]));
                }

                activation = Function(layer.ActivationType);
            }

            public long ParameterCount => scale.Length + shift.Length;

            public int Count => 1;

            public float[] Run(float[] values)
            {
                var result = new float[values.Length];
                for (int c = 0; c < scale.Length; c++)
                {
                    for (int i = c * positions; i < (c + 1) * positions; i++) result[i] = (values[i] * scale[c]) + shift[c];
                }

                Activate(activation, result);
                return result;
            }
        }

        private sealed class Pooling : IOperation
        {
            private readonly PoolingLayer layer;

            public Pooling(PoolingLayer layer) => this.layer = layer;

            public long ParameterCount => 0;

            public int Count => 1;

            public float[] Run(float[] values)
            {
                Shape input = layer.InputShape, output = layer.OutputShape;
                var result = new float[output.Size];

                for (int c = 0; c < output.Channels; c++)
                {
                    for (int oy = 0; oy < output.Height; oy++)
                    {
                        for (int ox = 0; ox < output.Width; ox++)
                        {
                            float max = float.NegativeInfinity, sum = 0;
                            int count = 0;
                            for (int ky = 0; ky < layer.Size; ky++)
                            {
                                int iy = (oy * layer.Stride) - layer.Padding + ky;
                                if (iy < 0 || iy >= input.Height) continue;

                                for (int kx = 0; kx < layer.Size; kx++)
                                {
                                    int ix = (ox * layer.Stride) - layer.Padding + kx;
                                    if (ix < 0 || ix >= input.Width) continue;

                                    float value = values[input.Index(c, iy, ix)];
                                    max = Math.Max(max, value);
                                    sum += value;
                                    count++;
                                }
                            }

                            result[output.Index(c, oy, ox)] = layer.Type == PoolingType.Max ? max : sum / count;
                        }
                    }
                }

                return result;
            }
        }

        private sealed class GlobalAverage : IOperation
        {
            private readonly Shape input;

            public GlobalAverage(Shape input) => this.input = input;

            public long ParameterCount => 0;

            public int Count => 1;

            public float[] Run(float[] values)
            {
                int positions = input.Height * input.Width;
                var result = new float[input.Channels];
                for (int c = 0; c < result.Length; c++)
                {
                    double sum = 0;
                    for (int i = c * positions; i < (c + 1) * positions; i++) sum += values[i];
                    result[c] = (float)(sum / positions);
                }

                return result;
            }
        }

        private sealed class Residual : IOperation
        {
            private readonly IReadOnlyList<IOperation> main, shortcut;
            private readonly IActivationFunction activation;

            public Residual(IReadOnlyList<IOperation> main, IReadOnlyList<IOperation> shortcut, ActivationTypes activationType)
            {
                this.main = main;
                this.shortcut = shortcut;
                activation = Function(activationType);
            }

            public long ParameterCount => main.Concat(shortcut).Sum(x => x.ParameterCount);

            public int Count => 1 + main.Concat(shortcut).Sum(x => x.Count);

            public float[] Run(float[] values)
            {
                float[] result = main.Aggregate(values, (value, operation) => operation.Run(value));
                float[] added = shortcut.Aggregate(values, (value, operation) => operation.Run(value));

                var sum = new float[result.Length];
                for (int i = 0; i < sum.Length; i++) sum[i] = result[i] + added[i];
                Activate(activation, sum);
                return sum;
            }
        }

        private sealed class Dense : IOperation
        {
            private readonly int inputs, outputs;
            private readonly float[] weights, bias;
            private readonly IActivationFunction activation;

            public Dense(DenseLayer layer)
            {
                inputs = layer.InputShape.Size;
                outputs = layer.OutputShape.Size;
                weights = Floats(layer.Weights);
                bias = Floats(layer.Biases);
                activation = Function(layer.ActivationType);
            }

            public long ParameterCount => weights.Length + bias.Length;

            public int Count => 1;

            public float[] Run(float[] values)
            {
                var result = (float[])bias.Clone();
                for (int i = 0; i < inputs; i++)
                {
                    float value = values[i];
                    if (value == 0) continue;

                    int row = i * outputs;
                    for (int j = 0; j < outputs; j++) result[j] += value * weights[row + j];
                }

                Activate(activation, result);
                return result;
            }
        }
    }
}
