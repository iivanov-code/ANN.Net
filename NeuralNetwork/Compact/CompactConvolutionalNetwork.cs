using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
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

        public CompactConvolutionalNetwork(ConvolutionalNetwork network, CompactOptions options = null)
        {
            options ??= new CompactOptions();
            Precision = options.Precision;
            operations = Compile(network.Layers, options).ToArray();

            // weights are pruned and stored once the threshold over all of them is known
            float threshold = options.Threshold(operations.SelectMany(x => x.Weights));
            foreach (IOperation operation in operations) operation.Seal(threshold, options.Precision);

            softMax = network.OutputApplySoftMax;
            InputCount = network.InputChannels * network.InputHeight * network.InputWidth;
            OutputCount = network.OutputDimension;
            ParameterCount = operations.Sum(x => x.ParameterCount);
            SizeInBytes = operations.Sum(x => x.SizeInBytes);
        }

        private CompactConvolutionalNetwork(IOperation[] operations, bool softMax, int inputCount, int outputCount, WeightPrecision precision)
        {
            this.operations = operations;
            this.softMax = softMax;
            InputCount = inputCount;
            OutputCount = outputCount;
            Precision = precision;
            ParameterCount = operations.Sum(x => x.ParameterCount);
            SizeInBytes = operations.Sum(x => x.SizeInBytes);
        }

        internal void Write(BinaryWriter writer)
        {
            writer.Write(InputCount);
            writer.Write(OutputCount);
            writer.Write(softMax);
            WriteOperations(writer, operations);
        }

        /// <summary>
        /// Reads what <see cref="Write"/> wrote, checking that every operation fits the one before it
        /// </summary>
        internal static CompactConvolutionalNetwork Read(BinaryReader reader, WeightPrecision precision)
        {
            int inputCount = reader.ReadInt32(), outputCount = reader.ReadInt32();
            bool softMax = reader.ReadBoolean();
            CompactIO.Require(inputCount > 0 && outputCount > 0, "Invalid input or output size");

            IOperation[] operations = ReadOperations(reader, precision, inputCount, 0);
            CompactIO.Require(operations.Length > 0 && operations[^1].OutputSize == outputCount, "The operations do not end with the outputs");
            return new CompactConvolutionalNetwork(operations, softMax, inputCount, outputCount, precision);
        }

        private static void WriteOperations(BinaryWriter writer, IReadOnlyList<IOperation> operations)
        {
            writer.Write(operations.Count);
            foreach (IOperation operation in operations) operation.Write(writer);
        }

        private static IOperation[] ReadOperations(BinaryReader reader, WeightPrecision precision, int inputSize, int depth)
        {
            CompactIO.Require(depth < 16, "Residual blocks nested too deeply");
            int count = reader.ReadInt32();
            CompactIO.Require(count >= 0 && count <= 10_000, "Invalid number of operations");

            var operations = new IOperation[count];
            for (int i = 0; i < count; i++)
            {
                byte kind = reader.ReadByte();
                operations[i] = kind switch
                {
                    1 => Convolution.Read(reader, precision),
                    2 => ScaleShift.Read(reader),
                    3 => Pooling.Read(reader),
                    4 => GlobalAverage.Read(reader),
                    5 => Residual.Read(reader, precision, inputSize, depth),
                    6 => Dense.Read(reader, precision),
                    _ => throw new InvalidDataException($"Unknown operation {kind}")
                };

                CompactIO.Require(operations[i].InputSize == inputSize, $"Operation {i} takes {operations[i].InputSize} values but gets {inputSize}");
                inputSize = operations[i].OutputSize;
            }

            return operations;
        }

        private static Shape ReadShape(BinaryReader reader)
        {
            var shape = new Shape(reader.ReadInt32(), reader.ReadInt32(), reader.ReadInt32());
            CompactIO.Require(shape.Channels > 0 && shape.Height > 0 && shape.Width > 0 && (long)shape.Channels * shape.Height * shape.Width <= int.MaxValue,
                              $"Invalid shape {shape}");
            return shape;
        }

        private static void WriteShape(BinaryWriter writer, Shape shape)
        {
            writer.Write(shape.Channels);
            writer.Write(shape.Height);
            writer.Write(shape.Width);
        }

        public int InputCount { get; }

        public int OutputCount { get; }

        public long ParameterCount { get; }

        public WeightPrecision Precision { get; }

        public long SizeInBytes { get; }

        /// <summary>
        /// Operations in order; batch normalizations folded into convolutions are gone
        /// </summary>
        internal int OperationCount => operations.Sum(x => x.Count);

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} input values but found {values.Length}", nameof(values));

            return Array.ConvertAll(Run(Array.ConvertAll(values, x => (float)x), parallel: true), x => (Quad)x);
        }

        public void Propagate(ReadOnlySpan<float> inputs, Span<float> outputs)
        {
            if (inputs.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} input values but found {inputs.Length}", nameof(inputs));
            if (outputs.Length < OutputCount)
                throw new ArgumentException($"Expected room for {OutputCount} outputs but found {outputs.Length}", nameof(outputs));

            Run(inputs.ToArray(), parallel: true).CopyTo(outputs);
        }

        /// <summary>
        /// Samples run on several threads, each convolution on one
        /// </summary>
        public float[][] PropagateBatch(IReadOnlyList<float[]> inputs)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            for (int i = 0; i < inputs.Count; i++)
            {
                if (inputs[i]?.Length != InputCount)
                    throw new ArgumentException($"Sample {i} has {inputs[i]?.Length ?? 0} values instead of {InputCount}", nameof(inputs));
            }

            var results = new float[inputs.Count][];
            if (inputs.Count == 1) results[0] = Run((float[])inputs[0].Clone(), parallel: true);
            else Parallel.For(0, inputs.Count, i => results[i] = Run((float[])inputs[i].Clone(), parallel: false));
            return results;
        }

        private float[] Run(float[] values, bool parallel)
        {
            float[] current = values;
            foreach (IOperation operation in operations)
            {
                current = operation.Run(current, parallel);
            }

            if (!softMax) return current;

            Quad[] outputs = Array.ConvertAll(current, x => (Quad)x);
            TransformerUtils.SoftMax(outputs);
            return Array.ConvertAll(outputs, x => (float)x);
        }

        private static List<IOperation> Compile(IReadOnlyList<IConvolutionalLayer> layers, CompactOptions options)
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
                        operations.Add(new Residual(Compile(residual.Main, options), Compile(residual.Shortcut, options), residual.ActivationType));
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

        private static float[] Pruned(float[] weights, float threshold)
        {
            if (threshold <= 0) return weights;

            for (int i = 0; i < weights.Length; i++)
            {
                if (Math.Abs(weights[i]) < threshold) weights[i] = 0;
            }

            return weights;
        }

        private static void Activate(CompactActivation activation, float[] values) => activation.Apply(values);

        private static CompactActivation Function(ActivationTypes type) => CompactActivation.Of(type);

        private interface IOperation
        {
            int InputSize { get; }

            int OutputSize { get; }

            /// <summary>
            /// Writes the kind of the operation and its values
            /// </summary>
            void Write(BinaryWriter writer);

            /// <summary>
            /// Weights that are not stored yet
            /// </summary>
            IEnumerable<float[]> Weights => Array.Empty<float[]>();

            /// <summary>
            /// Removes the weights below the threshold and stores the others
            /// </summary>
            void Seal(float threshold, WeightPrecision precision)
            {
            }

            long ParameterCount { get; }

            long SizeInBytes { get; }

            /// <summary>
            /// Number of operations, counting those inside a residual block
            /// </summary>
            int Count { get; }

            /// <param name="parallel">Whether a large operation may spread over several threads</param>
            float[] Run(float[] input, bool parallel);
        }

        /// <summary>
        /// Convolution as one matrix product per output position: the input patch under the kernel (channels x kernel
        /// height x kernel width values, 0 outside the image) times the weights of all filters
        /// </summary>
        private sealed class Convolution : IOperation
        {
            [ThreadStatic]
            private static float[] buffer;

            private readonly Shape input, output;
            private readonly int kernelHeight, kernelWidth, stride, padding;
            private readonly float[] bias;
            private readonly CompactActivation activation;
            private float[] transposed;
            private WeightMatrix weights;

            public Convolution(ConvolutionLayer layer)
                : this(layer, Floats(layer.Weights), Floats(layer.Biases), layer.ActivationType)
            {
            }

            private Convolution(Shape input, Shape output, int kernelHeight, int kernelWidth, int stride, int padding, float[] bias,
                                ActivationTypes activationType, WeightMatrix weights)
            {
                this.input = input;
                this.output = output;
                this.kernelHeight = kernelHeight;
                this.kernelWidth = kernelWidth;
                this.stride = stride;
                this.padding = padding;
                this.bias = bias;
                activation = Function(activationType);
                this.weights = weights;
            }

            public int InputSize => input.Size;

            public int OutputSize => output.Size;

            public void Write(BinaryWriter writer)
            {
                writer.Write((byte)1);
                WriteShape(writer, input);
                WriteShape(writer, output);
                writer.Write(kernelHeight);
                writer.Write(kernelWidth);
                writer.Write(stride);
                writer.Write(padding);
                writer.WriteArray(bias);
                writer.Write((int)activation.Type);
                weights.Write(writer);
            }

            public static Convolution Read(BinaryReader reader, WeightPrecision precision)
            {
                Shape input = ReadShape(reader), output = ReadShape(reader);
                int kernelHeight = reader.ReadInt32(), kernelWidth = reader.ReadInt32(), stride = reader.ReadInt32(), padding = reader.ReadInt32();
                float[] bias = reader.ReadArray<float>(true);
                ActivationTypes activation = reader.ReadActivation();
                WeightMatrix weights = WeightMatrix.Read(reader);

                CompactIO.Require(kernelHeight > 0 && kernelWidth > 0 && stride > 0 && padding >= 0, "Invalid convolution");
                CompactIO.Require(output.Height == ((input.Height + (2 * padding) - kernelHeight) / stride) + 1
                                  && output.Width == ((input.Width + (2 * padding) - kernelWidth) / stride) + 1, "The convolution output does not fit its input");
                CompactIO.Require(weights.Rows == input.Channels * kernelHeight * kernelWidth && weights.Columns == output.Channels && weights.Precision == precision,
                                  "The convolution weights do not fit its kernel");
                CompactIO.Require(bias == null || bias.Length == output.Channels, "The convolution needs one bias per filter");
                return new Convolution(input, output, kernelHeight, kernelWidth, stride, padding, bias, activation, weights);
            }

            /// <param name="filterWeights">Filter by filter, as the layer keeps them: [filter, channel, ky, kx]</param>
            private Convolution(ConvolutionLayer layer, float[] filterWeights, float[] bias, ActivationTypes activationType)
            {
                input = layer.InputShape;
                output = layer.OutputShape;
                kernelHeight = layer.KernelHeight;
                kernelWidth = layer.KernelWidth;
                stride = layer.Stride;
                padding = layer.Padding;
                this.bias = bias;
                activation = Function(activationType);

                int filters = output.Channels, patch = filterWeights.Length / filters;
                transposed = new float[filterWeights.Length];
                for (int f = 0; f < filters; f++)
                {
                    for (int k = 0; k < patch; k++) transposed[(k * filters) + f] = filterWeights[(f * patch) + k];
                }
            }

            public IEnumerable<float[]> Weights => new[] { transposed };

            public void Seal(float threshold, WeightPrecision precision)
            {
                weights = WeightMatrix.Create(Pruned(transposed, threshold), transposed.Length / output.Channels, output.Channels, precision);
                transposed = null;
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

            public long ParameterCount => weights.StoredValues + (bias?.Length ?? 0);

            public long SizeInBytes => weights.SizeInBytes + (4L * (bias?.Length ?? 0));

            public int Count => 1;

            /// <summary>
            /// Multiply-adds of one image; above this the output rows are spread over the processors
            /// </summary>
            private const long ParallelWork = 1 << 21;

            public float[] Run(float[] values, bool parallel)
            {
                var result = new float[output.Size];
                long work = (long)output.Height * output.Width * weights.Rows * weights.Columns;

                if (parallel && Environment.ProcessorCount > 1 && output.Height > 1 && work >= ParallelWork)
                {
                    Parallel.For(0, output.Height, oy => Rows(values, result, oy, oy + 1));
                }
                else
                {
                    Rows(values, result, 0, output.Height);
                }

                Activate(activation, result);
                return result;
            }

            /// <summary>
            /// Output positions whose patches are multiplied together, so that 16-bit and 8-bit weight rows are converted
            /// once per tile instead of once per position
            /// </summary>
            private const int Tile = 16;

            private void Rows(float[] values, float[] result, int firstRow, int endRow)
            {
                int filters = output.Channels, positions = output.Height * output.Width;
                int patchSize = weights.Rows;

                int needed = Tile * (patchSize + filters);
                if (buffer == null || buffer.Length < needed) buffer = new float[needed];
                Span<float> patches = buffer.AsSpan(0, Tile * patchSize);
                Span<float> sums = buffer.AsSpan(Tile * patchSize, Tile * filters);

                int end = endRow * output.Width;
                for (int first = firstRow * output.Width; first < end; first += Tile)
                {
                    int count = Math.Min(Tile, end - first);
                    for (int t = 0; t < count; t++)
                    {
                        int position = first + t;
                        Patch(values, position / output.Width, position % output.Width, patches.Slice(t * patchSize, patchSize));

                        Span<float> sum = sums.Slice(t * filters, filters);
                        if (bias == null) sum.Clear();
                        else bias.CopyTo(sum);
                    }

                    weights.MultiplyAddBatch(patches, patchSize, sums, filters, count);

                    for (int t = 0; t < count; t++)
                    {
                        ReadOnlySpan<float> sum = sums.Slice(t * filters, filters);
                        for (int f = 0; f < filters; f++) result[(f * positions) + first + t] = sum[f];
                    }
                }
            }

            /// <summary>
            /// Input values under the kernel at output position (oy, ox), channel by channel, 0 outside the image
            /// </summary>
            private void Patch(float[] values, int oy, int ox, Span<float> patch)
            {
                int channels = input.Channels, height = input.Height, width = input.Width;
                int top = (oy * stride) - padding, left = (ox * stride) - padding;
                int k = 0;
                for (int c = 0; c < channels; c++)
                {
                    int channelStart = c * height * width;
                    for (int ky = 0; ky < kernelHeight; ky++)
                    {
                        int iy = top + ky;
                        if (iy < 0 || iy >= height)
                        {
                            patch.Slice(k, kernelWidth).Clear();
                            k += kernelWidth;
                            continue;
                        }

                        int rowStart = channelStart + (iy * width);
                        if (left >= 0 && left + kernelWidth <= width)
                        {
                            values.AsSpan(rowStart + left, kernelWidth).CopyTo(patch.Slice(k, kernelWidth));
                            k += kernelWidth;
                            continue;
                        }

                        for (int kx = 0; kx < kernelWidth; kx++)
                        {
                            int ix = left + kx;
                            patch[k++] = ix >= 0 && ix < width ? values[rowStart + ix] : 0;
                        }
                    }
                }
            }
        }

        private sealed class ScaleShift : IOperation
        {
            private readonly int positions;
            private readonly float[] scale, shift;
            private readonly CompactActivation activation;

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

            private ScaleShift(int positions, float[] scale, float[] shift, ActivationTypes activationType)
            {
                this.positions = positions;
                this.scale = scale;
                this.shift = shift;
                activation = Function(activationType);
            }

            public int InputSize => positions * scale.Length;

            public int OutputSize => InputSize;

            public void Write(BinaryWriter writer)
            {
                writer.Write((byte)2);
                writer.Write(positions);
                writer.WriteArray(scale);
                writer.WriteArray(shift);
                writer.Write((int)activation.Type);
            }

            public static ScaleShift Read(BinaryReader reader)
            {
                int positions = reader.ReadInt32();
                float[] scale = reader.ReadArray<float>(), shift = reader.ReadArray<float>();
                ActivationTypes activation = reader.ReadActivation();
                CompactIO.Require(positions > 0 && scale.Length > 0 && shift.Length == scale.Length && (long)positions * scale.Length <= int.MaxValue,
                                  "Invalid batch normalization");
                return new ScaleShift(positions, scale, shift, activation);
            }

            public long ParameterCount => scale.Length + shift.Length;

            public long SizeInBytes => 4L * ParameterCount;

            public int Count => 1;

            public float[] Run(float[] values, bool parallel)
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

            public int InputSize => layer.InputShape.Size;

            public int OutputSize => layer.OutputShape.Size;

            public void Write(BinaryWriter writer)
            {
                writer.Write((byte)3);
                WriteShape(writer, layer.InputShape);
                writer.Write((int)layer.Type);
                writer.Write(layer.Size);
                writer.Write(layer.Stride);
                writer.Write(layer.Padding);
            }

            public static Pooling Read(BinaryReader reader)
            {
                Shape input = ReadShape(reader);
                var type = (PoolingType)reader.ReadInt32();
                int size = reader.ReadInt32(), stride = reader.ReadInt32(), padding = reader.ReadInt32();
                CompactIO.Require(Enum.IsDefined(type) && size > 0 && stride > 0 && padding >= 0 && padding < size
                                  && input.Height + (2 * padding) >= size && input.Width + (2 * padding) >= size, "Invalid pooling");
                return new Pooling(new PoolingLayer(input, type, size, stride, padding));
            }

            public long ParameterCount => 0;

            public long SizeInBytes => 0;

            public int Count => 1;

            public float[] Run(float[] values, bool parallel)
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

            public int InputSize => input.Size;

            public int OutputSize => input.Channels;

            public void Write(BinaryWriter writer)
            {
                writer.Write((byte)4);
                WriteShape(writer, input);
            }

            public static GlobalAverage Read(BinaryReader reader) => new GlobalAverage(ReadShape(reader));

            public long ParameterCount => 0;

            public long SizeInBytes => 0;

            public int Count => 1;

            public float[] Run(float[] values, bool parallel)
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
            private readonly CompactActivation activation;

            public Residual(IReadOnlyList<IOperation> main, IReadOnlyList<IOperation> shortcut, ActivationTypes activationType)
            {
                this.main = main;
                this.shortcut = shortcut;
                activation = Function(activationType);
            }

            public int InputSize => main.Count > 0 ? main[0].InputSize : shortcut[0].InputSize;

            public int OutputSize => main.Count > 0 ? main[^1].OutputSize : InputSize;

            public void Write(BinaryWriter writer)
            {
                writer.Write((byte)5);
                WriteOperations(writer, main);
                WriteOperations(writer, shortcut);
                writer.Write((int)activation.Type);
            }

            public static Residual Read(BinaryReader reader, WeightPrecision precision, int inputSize, int depth)
            {
                IOperation[] main = ReadOperations(reader, precision, inputSize, depth + 1);
                IOperation[] shortcut = ReadOperations(reader, precision, inputSize, depth + 1);
                ActivationTypes activation = reader.ReadActivation();

                CompactIO.Require(main.Length > 0, "A residual block without operations");
                int mainOutput = main[^1].OutputSize, shortcutOutput = shortcut.Length > 0 ? shortcut[^1].OutputSize : inputSize;
                CompactIO.Require(mainOutput == shortcutOutput, "The two paths of a residual block end with different sizes");
                return new Residual(main, shortcut, activation);
            }

            public IEnumerable<float[]> Weights => main.Concat(shortcut).SelectMany(x => x.Weights);

            public void Seal(float threshold, WeightPrecision precision)
            {
                foreach (IOperation operation in main.Concat(shortcut)) operation.Seal(threshold, precision);
            }

            public long ParameterCount => main.Concat(shortcut).Sum(x => x.ParameterCount);

            public long SizeInBytes => main.Concat(shortcut).Sum(x => x.SizeInBytes);

            public int Count => 1 + main.Concat(shortcut).Sum(x => x.Count);

            public float[] Run(float[] values, bool parallel)
            {
                float[] result = main.Aggregate(values, (value, operation) => operation.Run(value, parallel));
                float[] added = shortcut.Aggregate(values, (value, operation) => operation.Run(value, parallel));

                var sum = new float[result.Length];
                for (int i = 0; i < sum.Length; i++) sum[i] = result[i] + added[i];
                Activate(activation, sum);
                return sum;
            }
        }

        private sealed class Dense : IOperation
        {
            private readonly int inputs, outputs;
            private readonly float[] bias;
            private readonly CompactActivation activation;
            private float[] pending;
            private WeightMatrix weights;

            public Dense(DenseLayer layer)
            {
                inputs = layer.InputShape.Size;
                outputs = layer.OutputShape.Size;
                pending = Floats(layer.Weights);
                bias = Floats(layer.Biases);
                activation = Function(layer.ActivationType);
            }

            private Dense(float[] bias, ActivationTypes activationType, WeightMatrix weights)
            {
                inputs = weights.Rows;
                outputs = weights.Columns;
                this.bias = bias;
                activation = Function(activationType);
                this.weights = weights;
            }

            public int InputSize => inputs;

            public int OutputSize => outputs;

            public void Write(BinaryWriter writer)
            {
                writer.Write((byte)6);
                writer.WriteArray(bias);
                writer.Write((int)activation.Type);
                weights.Write(writer);
            }

            public static Dense Read(BinaryReader reader, WeightPrecision precision)
            {
                float[] bias = reader.ReadArray<float>();
                ActivationTypes activation = reader.ReadActivation();
                WeightMatrix weights = WeightMatrix.Read(reader);
                CompactIO.Require(weights.Rows > 0 && weights.Columns == bias.Length && weights.Precision == precision, "The dense weights do not fit its biases");
                return new Dense(bias, activation, weights);
            }

            public IEnumerable<float[]> Weights => new[] { pending };

            public void Seal(float threshold, WeightPrecision precision)
            {
                weights = WeightMatrix.Create(Pruned(pending, threshold), inputs, outputs, precision);
                pending = null;
            }

            public long ParameterCount => weights.StoredValues + bias.Length;

            public long SizeInBytes => weights.SizeInBytes + (4L * bias.Length);

            public int Count => 1;

            public float[] Run(float[] values, bool parallel)
            {
                var result = (float[])bias.Clone();
                weights.MultiplyAdd(values, result);
                Activate(activation, result);
                return result;
            }
        }
    }
}
