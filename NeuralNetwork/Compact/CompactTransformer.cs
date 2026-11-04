using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Serialization;
using ANN.Net.Transformers;

namespace ANN.Net.Compact
{
    /// <summary>
    /// Inference copy of a <see cref="TransformerNetwork"/>: every projection is a <see cref="WeightMatrix"/> applied
    /// to all tokens of a sequence at once, and layer normalization, attention, residual sums and soft-max repeat the
    /// operations of the network in the same order
    /// </summary>
    internal sealed class CompactTransformer : ICompactSequenceNetwork
    {
        private readonly Projection inputProjection;
        private readonly float[] positional;
        private readonly Block[] blocks;
        private readonly Norm outputNormalization;
        private readonly Projection outputProjection;

        public CompactTransformer(TransformerNetwork network, CompactOptions options = null)
        {
            options ??= new CompactOptions();
            Precision = options.Precision;
            InputDimension = network.Settings.InputDimension;
            ModelDimension = network.Settings.ModelDimension;
            OutputDimension = network.Settings.OutputDimension;
            MaxSequenceLength = network.Settings.MaxSequenceLength;
            OutputApplySoftMax = network.Settings.OutputApplySoftMax;
            positional = Positional(MaxSequenceLength, ModelDimension);

            inputProjection = new Projection(network.InputProjection);
            blocks = network.Blocks.Select(x => new Block(x)).ToArray();
            outputNormalization = new Norm(network.OutputNormalization);
            outputProjection = new Projection(network.OutputProjection);

            // weights are pruned and stored once the threshold over all of them is known
            float threshold = options.Threshold(Projections().Select(x => x.Pending).Concat(blocks.SelectMany(x => x.GraphWeights)));
            foreach (Projection projection in Projections()) projection.Seal(threshold, Precision);
            foreach (Block block in blocks) block.Seal(threshold, Precision);
            (ParameterCount, SizeInBytes) = Sizes();
        }

        private CompactTransformer(WeightPrecision precision, int inputDimension, int modelDimension, int outputDimension, int maxSequenceLength, bool softMax,
                                   Projection inputProjection, Block[] blocks, Norm outputNormalization, Projection outputProjection)
        {
            Precision = precision;
            InputDimension = inputDimension;
            ModelDimension = modelDimension;
            OutputDimension = outputDimension;
            MaxSequenceLength = maxSequenceLength;
            OutputApplySoftMax = softMax;
            positional = Positional(maxSequenceLength, modelDimension);
            this.inputProjection = inputProjection;
            this.blocks = blocks;
            this.outputNormalization = outputNormalization;
            this.outputProjection = outputProjection;
            (ParameterCount, SizeInBytes) = Sizes();
        }

        public int InputDimension { get; }

        public int ModelDimension { get; }

        public int OutputDimension { get; }

        public int MaxSequenceLength { get; }

        public bool OutputApplySoftMax { get; }

        public long ParameterCount { get; }

        public WeightPrecision Precision { get; }

        public long SizeInBytes { get; }

        private static float[] Positional(int length, int dimension) =>
            PositionalEncoding.Build(length, dimension).SelectMany(x => x).Select(x => (float)x).ToArray();

        private IEnumerable<Projection> Projections() =>
            new[] { inputProjection }.Concat(blocks.SelectMany(x => x.Projections)).Append(outputProjection);

        private (long Parameters, long Bytes) Sizes()
        {
            long parameters = Projections().Sum(x => x.ParameterCount) + outputNormalization.ParameterCount + blocks.Sum(x => x.ExtraParameters);
            long bytes = Projections().Sum(x => x.SizeInBytes) + (4L * (outputNormalization.ParameterCount + positional.Length)) + blocks.Sum(x => x.ExtraBytes);
            return (parameters, bytes);
        }

        public Quad[][] Propagate(Quad[][] sequence)
        {
            float[] flat = Flatten(sequence);
            float[] outputs = Run(flat, sequence.Length);
            return Enumerable.Range(0, sequence.Length).Select(t => outputs.AsSpan(t * OutputDimension, OutputDimension).ToArray().Select(x => (Quad)x).ToArray()).ToArray();
        }

        public float[][][] PropagateBatch(IReadOnlyList<float[][]> sequences)
        {
            ArgumentNullException.ThrowIfNull(sequences);
            float[][] flat = sequences.Select(x => Flatten(x?.Select(v => v?.Select(f => (Quad)f).ToArray()).ToArray())).ToArray();

            var results = new float[sequences.Count][][];
            Parallel.For(0, sequences.Count, i =>
            {
                int length = sequences[i].Length;
                float[] outputs = Run(flat[i], length);
                results[i] = Enumerable.Range(0, length).Select(t => outputs.AsSpan(t * OutputDimension, OutputDimension).ToArray()).ToArray();
            });
            return results;
        }

        private float[] Flatten(Quad[][] sequence)
        {
            if (sequence == null) throw new ArgumentNullException(nameof(sequence));
            if (sequence.Length < 1) throw new ArgumentException("Sequence must not be empty", nameof(sequence));
            if (sequence.Length > MaxSequenceLength)
                throw new ArgumentException($"Sequence is longer than the maximum of {MaxSequenceLength}", nameof(sequence));

            var flat = new float[sequence.Length * InputDimension];
            for (int t = 0; t < sequence.Length; t++)
            {
                if (sequence[t]?.Length != InputDimension)
                    throw new ArgumentException($"Expected vectors of size {InputDimension} but found {sequence[t]?.Length ?? 0}", nameof(sequence));
                for (int i = 0; i < InputDimension; i++) flat[(t * InputDimension) + i] = sequence[t][i];
            }

            return flat;
        }

        private float[] Run(float[] inputs, int length)
        {
            int d = ModelDimension;
            var hidden = new float[length * d];
            inputProjection.Run(inputs, hidden, length);
            for (int i = 0; i < hidden.Length; i++) hidden[i] = (Quad)hidden[i] + (Quad)positional[i];

            foreach (Block block in blocks) hidden = block.Run(hidden, length);

            var normalized = new float[hidden.Length];
            outputNormalization.Run(hidden, normalized, length);
            var outputs = new float[length * OutputDimension];
            outputProjection.Run(normalized, outputs, length);

            if (OutputApplySoftMax)
            {
                for (int t = 0; t < length; t++) SoftMax(outputs.AsSpan(t * OutputDimension, OutputDimension));
            }

            return outputs;
        }

        /// <summary>
        /// <see cref="TransformerUtils.SoftMax"/> on floats
        /// </summary>
        private static void SoftMax(Span<float> values)
        {
            Quad[] quads = new Quad[values.Length];
            for (int i = 0; i < quads.Length; i++) quads[i] = values[i];
            TransformerUtils.SoftMax(quads);
            for (int i = 0; i < quads.Length; i++) values[i] = quads[i];
        }

        private static void Add(float[] a, float[] b, float[] sum)
        {
            for (int i = 0; i < sum.Length; i++) sum[i] = (Quad)a[i] + (Quad)b[i];
        }

        internal void Write(BinaryWriter writer)
        {
            writer.Write(InputDimension);
            writer.Write(ModelDimension);
            writer.Write(OutputDimension);
            writer.Write(MaxSequenceLength);
            writer.Write(OutputApplySoftMax);
            inputProjection.Write(writer);
            writer.Write(blocks.Length);
            foreach (Block block in blocks) block.Write(writer);
            outputNormalization.Write(writer);
            outputProjection.Write(writer);
        }

        internal static CompactTransformer Read(BinaryReader reader, WeightPrecision precision)
        {
            int inputDimension = reader.ReadInt32(), modelDimension = reader.ReadInt32(), outputDimension = reader.ReadInt32(), maxLength = reader.ReadInt32();
            bool softMax = reader.ReadBoolean();
            CompactIO.Require(inputDimension > 0 && modelDimension > 0 && outputDimension > 0 && maxLength > 0 && (long)maxLength * modelDimension <= 1 << 28,
                              "Invalid transformer dimensions");

            Projection input = Projection.Read(reader, precision, inputDimension, modelDimension);
            int count = reader.ReadInt32();
            CompactIO.Require(count >= 0 && count <= 1000, "Invalid number of blocks");
            var blocks = new Block[count];
            for (int i = 0; i < count; i++) blocks[i] = Block.Read(reader, precision, modelDimension);

            Norm normalization = Norm.Read(reader, modelDimension);
            Projection output = Projection.Read(reader, precision, modelDimension, outputDimension);
            return new CompactTransformer(precision, inputDimension, modelDimension, outputDimension, maxLength, softMax, input, blocks, normalization, output);
        }

        /// <summary>
        /// x W + b for every token of a sequence
        /// </summary>
        private sealed class Projection
        {
            private WeightMatrix weights;

            public Projection(LinearProjection projection)
            {
                Inputs = projection.InputsCount;
                Outputs = projection.OutputsCount;
                Pending = new float[Inputs * Outputs];
                for (int i = 0; i < Inputs; i++)
                {
                    for (int j = 0; j < Outputs; j++) Pending[(i * Outputs) + j] = projection.Weights[i, j];
                }

                Bias = projection.Biases.Select(x => (float)x).ToArray();
            }

            public Projection(WeightMatrix weights, float[] bias)
            {
                this.weights = weights;
                Inputs = weights.Rows;
                Outputs = weights.Columns;
                Bias = bias;
            }

            public int Inputs { get; }

            public int Outputs { get; }

            public float[] Bias { get; }

            public float[] Pending { get; private set; }

            public long ParameterCount => weights.StoredValues + Bias.Length;

            public long SizeInBytes => weights.SizeInBytes + (4L * Bias.Length);

            public void Seal(float threshold, WeightPrecision precision)
            {
                if (threshold > 0)
                {
                    for (int i = 0; i < Pending.Length; i++)
                    {
                        if (Math.Abs(Pending[i]) < threshold) Pending[i] = 0;
                    }
                }

                weights = WeightMatrix.Create(Pending, Inputs, Outputs, precision);
                Pending = null;
            }

            public void Run(ReadOnlySpan<float> inputs, Span<float> outputs, int length)
            {
                for (int t = 0; t < length; t++) Bias.CopyTo(outputs.Slice(t * Outputs, Outputs));
                weights.MultiplyAddBatch(inputs, Inputs, outputs, Outputs, length);
            }

            public void Write(BinaryWriter writer)
            {
                writer.WriteArray(Bias);
                weights.Write(writer);
            }

            public static Projection Read(BinaryReader reader, WeightPrecision precision, int inputs, int outputs)
            {
                float[] bias = reader.ReadArray<float>();
                WeightMatrix weights = WeightMatrix.Read(reader);
                CompactIO.Require(weights.Rows == inputs && weights.Columns == outputs && bias.Length == outputs && weights.Precision == precision,
                                  "A projection does not fit the transformer");
                return new Projection(weights, bias);
            }
        }

        /// <summary>
        /// Layer normalization of every token, computed as <see cref="LayerNormalization"/> computes it
        /// </summary>
        private sealed class Norm
        {
            private readonly float[] gains, offsets;

            public Norm(LayerNormalization normalization)
                : this(normalization.Gains.Select(x => (float)x).ToArray(), normalization.Offsets.Select(x => (float)x).ToArray())
            {
            }

            private Norm(float[] gains, float[] offsets)
            {
                this.gains = gains;
                this.offsets = offsets;
            }

            public long ParameterCount => gains.Length + offsets.Length;

            public void Run(float[] inputs, float[] outputs, int length)
            {
                int dimension = gains.Length;
                for (int t = 0; t < length; t++)
                {
                    int start = t * dimension;
                    Quad mean = 0;
                    for (int i = 0; i < dimension; i++) mean += inputs[start + i];
                    mean /= dimension;

                    Quad variance = 0;
                    for (int i = 0; i < dimension; i++)
                    {
                        Quad centered = inputs[start + i] - mean;
                        variance += centered * centered;
                    }

                    variance /= dimension;
                    Quad inverseDeviation = (Quad)(1 / Math.Sqrt(variance + LayerNormalization.EPSILON));

                    for (int i = 0; i < dimension; i++)
                    {
                        Quad normalized = (inputs[start + i] - mean) * inverseDeviation;
                        outputs[start + i] = (normalized * gains[i]) + offsets[i];
                    }
                }
            }

            public void Write(BinaryWriter writer)
            {
                writer.WriteArray(gains);
                writer.WriteArray(offsets);
            }

            public static Norm Read(BinaryReader reader, int dimension)
            {
                float[] gains = reader.ReadArray<float>(), offsets = reader.ReadArray<float>();
                CompactIO.Require(gains.Length == dimension && offsets.Length == dimension, "A layer normalization does not fit the transformer");
                return new Norm(gains, offsets);
            }
        }

        private sealed class Block
        {
            private readonly Norm attentionNormalization, feedForwardNormalization;
            private readonly Projection query, key, value, output;
            private readonly int heads;
            private readonly bool causal;
            private readonly float scale;

            // the feed-forward network: two projections around an activation, or a neuron network
            private readonly Projection expansion, contraction;
            private readonly CompactActivation activation;
            private readonly GraphPlan graphPlan;
            private CompactGraphNetwork graph;

            public Block(TransformerEncoderBlock block)
            {
                attentionNormalization = new Norm(block.AttentionNormalization);
                feedForwardNormalization = new Norm(block.FeedForwardNormalization);
                query = new Projection(block.Attention.QueryProjection);
                key = new Projection(block.Attention.KeyProjection);
                value = new Projection(block.Attention.ValueProjection);
                output = new Projection(block.Attention.OutputProjection);
                heads = block.Attention.HeadsCount;
                causal = block.Attention.UseCausalMask;
                scale = block.Attention.Scale;

                switch (block.FeedForward)
                {
                    case FeedForwardNetwork matrix:
                        expansion = new Projection(matrix.Expansion);
                        contraction = new Projection(matrix.Contraction);
                        activation = CompactActivation.Of(matrix.ActivationType);
                        break;

                    case NeuronFeedForwardNetwork neurons:
                        graphPlan = GraphPlan.Create(NetworkSerializer.ToDocument(neurons.Network));
                        break;

                    default:
                        throw new NotSupportedException($"A {block.FeedForward.GetType().Name} cannot be compacted");
                }
            }

            private Block(Norm attentionNormalization, Projection query, Projection key, Projection value, Projection output, int heads, bool causal, float scale,
                          Norm feedForwardNormalization, Projection expansion, CompactActivation activation, Projection contraction, CompactGraphNetwork graph)
            {
                this.attentionNormalization = attentionNormalization;
                this.query = query;
                this.key = key;
                this.value = value;
                this.output = output;
                this.heads = heads;
                this.causal = causal;
                this.scale = scale;
                this.feedForwardNormalization = feedForwardNormalization;
                this.expansion = expansion;
                this.activation = activation;
                this.contraction = contraction;
                this.graph = graph;
            }

            public IEnumerable<Projection> Projections =>
                new[] { query, key, value, output }.Concat(expansion == null ? Array.Empty<Projection>() : new[] { expansion, contraction });

            public IEnumerable<float[]> GraphWeights => graphPlan == null ? Array.Empty<float[]>() : GraphPruner.Weights(graphPlan);

            public long ExtraParameters => attentionNormalization.ParameterCount + feedForwardNormalization.ParameterCount + (graph?.ParameterCount ?? 0);

            public long ExtraBytes => (4L * (attentionNormalization.ParameterCount + feedForwardNormalization.ParameterCount)) + (graph?.SizeInBytes ?? 0);

            public void Seal(float threshold, WeightPrecision precision)
            {
                if (graphPlan != null) graph = new CompactGraphNetwork(graphPlan, new CompactOptions { Precision = precision, PruneThreshold = threshold });
            }

            public float[] Run(float[] hidden, int length)
            {
                int d = query.Inputs;
                var normalized = new float[hidden.Length];
                attentionNormalization.Run(hidden, normalized, length);

                float[] attended = Attend(normalized, length, d);
                var residual = new float[hidden.Length];
                Add(hidden, attended, residual);

                feedForwardNormalization.Run(residual, normalized, length);
                var transformed = new float[hidden.Length];
                if (graph != null)
                {
                    for (int t = 0; t < length; t++) graph.Propagate(normalized.AsSpan(t * d, d), transformed.AsSpan(t * d, d));
                }
                else
                {
                    var expanded = new float[length * expansion.Outputs];
                    expansion.Run(normalized, expanded, length);
                    activation.Apply(expanded);
                    contraction.Run(expanded, transformed, length);
                }

                var result = new float[hidden.Length];
                Add(residual, transformed, result);
                return result;
            }

            private float[] Attend(float[] inputs, int length, int d)
            {
                float[] queries = new float[length * d], keys = new float[length * d], values = new float[length * d];
                query.Run(inputs, queries, length);
                key.Run(inputs, keys, length);
                value.Run(inputs, values, length);

                int headDimension = d / heads;
                var concatenated = new float[length * d];
                var scores = new Quad[length];
                for (int head = 0; head < heads; head++)
                {
                    int offset = head * headDimension;
                    for (int i = 0; i < length; i++)
                    {
                        for (int j = 0; j < length; j++)
                        {
                            if (causal && j > i)
                            {
                                scores[j] = Quad.NegativeInfinity;
                                continue;
                            }

                            Quad dot = 0;
                            for (int c = 0; c < headDimension; c++) dot += (Quad)queries[(i * d) + offset + c] * keys[(j * d) + offset + c];
                            scores[j] = dot * scale;
                        }

                        TransformerUtils.SoftMax(scores);

                        for (int c = 0; c < headDimension; c++)
                        {
                            Quad sum = 0;
                            for (int j = 0; j < length; j++) sum += scores[j] * values[(j * d) + offset + c];
                            concatenated[(i * d) + offset + c] = sum;
                        }
                    }
                }

                var result = new float[length * d];
                output.Run(concatenated, result, length);
                return result;
            }

            public void Write(BinaryWriter writer)
            {
                attentionNormalization.Write(writer);
                query.Write(writer);
                key.Write(writer);
                value.Write(writer);
                output.Write(writer);
                writer.Write(heads);
                writer.Write(causal);
                writer.Write(scale);
                feedForwardNormalization.Write(writer);
                writer.Write(graph == null ? (byte)1 : (byte)2);
                if (graph == null)
                {
                    expansion.Write(writer);
                    writer.Write((int)activation.Type);
                    writer.Write(expansion.Outputs);
                    contraction.Write(writer);
                }
                else
                {
                    graph.Write(writer);
                }
            }

            public static Block Read(BinaryReader reader, WeightPrecision precision, int d)
            {
                Norm attentionNormalization = Norm.Read(reader, d);
                Projection query = Projection.Read(reader, precision, d, d), key = Projection.Read(reader, precision, d, d);
                Projection value = Projection.Read(reader, precision, d, d), output = Projection.Read(reader, precision, d, d);
                int heads = reader.ReadInt32();
                bool causal = reader.ReadBoolean();
                float scale = reader.ReadSingle();
                CompactIO.Require(heads > 0 && d % heads == 0, "Invalid number of attention heads");
                Norm feedForwardNormalization = Norm.Read(reader, d);

                byte kind = reader.ReadByte();
                if (kind == 1)
                {
                    float[] bias = reader.ReadArray<float>();
                    WeightMatrix weights = WeightMatrix.Read(reader);
                    CompactIO.Require(weights.Rows == d && weights.Columns == bias.Length && weights.Precision == precision, "Invalid feed-forward expansion");
                    var expansion = new Projection(weights, bias);
                    CompactActivation activation = CompactActivation.Of(reader.ReadActivation());
                    CompactIO.Require(reader.ReadInt32() == expansion.Outputs, "Invalid feed-forward size");
                    Projection contraction = Projection.Read(reader, precision, expansion.Outputs, d);
                    return new Block(attentionNormalization, query, key, value, output, heads, causal, scale, feedForwardNormalization, expansion, activation, contraction, null);
                }

                CompactIO.Require(kind == 2, $"Unknown feed-forward kind {kind}");
                CompactGraphNetwork graph = CompactGraphNetwork.Read(reader, precision);
                CompactIO.Require(graph.InputCount == d && graph.OutputCount == d, "The feed-forward network does not fit the transformer");
                return new Block(attentionNormalization, query, key, value, output, heads, causal, scale, feedForwardNormalization, null, null, null, graph);
            }
        }
    }
}
