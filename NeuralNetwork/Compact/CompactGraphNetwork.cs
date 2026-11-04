using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.Compact
{
    /// <summary>
    /// Runs a <see cref="GraphPlan"/>: one row of columns per sample, filled level by level
    /// </summary>
    internal sealed class CompactGraphNetwork : ICompactNetwork
    {
        /// <summary>
        /// Samples computed together; their columns stay in the cache while every weight row is applied to all of them
        /// </summary>
        private const int BlockSize = 32;

        [ThreadStatic]
        private static float[] buffer;

        private readonly int inputCount;
        private readonly float[] inputMinimum, inputRange;
        private readonly InputGroup[] inputGroups;
        private readonly Group[] groups;
        private readonly int width;
        private readonly int[] outputColumns;
        private readonly bool divideBySum;
        private readonly int gatherLength;

        public CompactGraphNetwork(GraphPlan plan, CompactOptions options = null)
        {
            options ??= new CompactOptions();
            Precision = options.Precision;

            float threshold = options.Threshold(GraphPruner.Weights(plan));
            if (threshold > 0) plan = GraphPruner.Prune(plan, threshold);

            inputCount = plan.InputCount;
            inputMinimum = plan.InputMinimum;
            inputRange = plan.InputRange;
            inputGroups = plan.InputGroups.Select(g => new InputGroup(g.Positions, CompactActivation.Of(g.Activation))).ToArray();
            groups = plan.Levels.SelectMany(level => level).Select(g => new Group(g, options.Precision)).ToArray();
            width = plan.Width;
            outputColumns = plan.OutputColumns;
            divideBySum = plan.DivideBySum;
            gatherLength = Math.Max(inputCount, groups.Select(g => g.Sources.Length).DefaultIfEmpty(0).Max());

            ParameterCount = groups.Sum(g => (g.Weights?.StoredValues ?? 0) + g.Bias.Length) + ((inputMinimum?.Length ?? 0) * 2);
            SizeInBytes = groups.Sum(g => (g.Weights?.SizeInBytes ?? 0) + (4L * (g.Bias.Length + g.Sources.Length)))
                          + ((inputMinimum?.Length ?? 0) * 8L) + (4L * outputColumns.Length);
        }

        public int InputCount => inputCount;

        public int OutputCount => outputColumns.Length;

        public long ParameterCount { get; }

        public WeightPrecision Precision { get; }

        public long SizeInBytes { get; }

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} input values but found {values.Length}", nameof(values));

            var inputs = new float[values.Length];
            for (int i = 0; i < inputs.Length; i++) inputs[i] = values[i];

            var outputs = new float[OutputCount];
            Propagate(inputs, outputs);
            return Array.ConvertAll(outputs, x => (Quad)x);
        }

        public void Propagate(ReadOnlySpan<float> inputs, Span<float> outputs)
        {
            if (inputs.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} input values but found {inputs.Length}", nameof(inputs));
            if (outputs.Length < OutputCount)
                throw new ArgumentException($"Expected room for {OutputCount} outputs but found {outputs.Length}", nameof(outputs));

            Run(inputs, inputCount, outputs, OutputCount, 1);
        }

        public float[][] PropagateBatch(IReadOnlyList<float[]> inputs)
        {
            ArgumentNullException.ThrowIfNull(inputs);
            int count = inputs.Count;
            var results = new float[count][];
            int blocks = (count + BlockSize - 1) / BlockSize;

            void RunBlock(int block)
            {
                int first = block * BlockSize, size = Math.Min(BlockSize, count - first);
                var packed = new float[size * inputCount];
                for (int b = 0; b < size; b++)
                {
                    float[] input = inputs[first + b] ?? throw new ArgumentException($"Sample {first + b} is null", nameof(inputs));
                    if (input.Length != inputCount)
                        throw new ArgumentException($"Sample {first + b} has {input.Length} values instead of {inputCount}", nameof(inputs));
                    input.CopyTo(packed, b * inputCount);
                }

                var outputs = new float[size * OutputCount];
                Run(packed, inputCount, outputs, OutputCount, size);
                for (int b = 0; b < size; b++) results[first + b] = outputs.AsSpan(b * OutputCount, OutputCount).ToArray();
            }

            if (blocks > 1) Parallel.For(0, blocks, RunBlock);
            else if (blocks == 1) RunBlock(0);

            return results;
        }

        /// <summary>
        /// <paramref name="count"/> samples, sample b reading its inputs from b * <paramref name="inputStride"/> and
        /// writing its outputs from b * <paramref name="outputStride"/>
        /// </summary>
        private void Run(ReadOnlySpan<float> inputs, int inputStride, Span<float> outputs, int outputStride, int count)
        {
            int needed = (width + gatherLength) * count;
            if (buffer == null || buffer.Length < needed) buffer = new float[needed];
            Span<float> columns = buffer.AsSpan(0, width * count);
            Span<float> work = buffer.AsSpan(width * count, gatherLength * count);

            for (int b = 0; b < count; b++)
            {
                ReadOnlySpan<float> input = inputs.Slice(b * inputStride, inputCount);
                Span<float> row = columns.Slice(b * width, width);
                Span<float> normalized = work.Slice(0, inputCount);
                for (int i = 0; i < inputCount; i++) normalized[i] = inputMinimum == null ? input[i] : (input[i] - inputMinimum[i]) / inputRange[i];

                if (inputGroups.Length == 0)
                {
                    normalized.CopyTo(row);
                    continue;
                }

                int column = 0;
                foreach (InputGroup group in inputGroups)
                {
                    foreach (int position in group.Positions) row[column++] = group.Activation.Apply(normalized[position]);
                }
            }

            foreach (Group group in groups) group.Run(columns, width, work, count);

            for (int b = 0; b < count; b++)
            {
                Span<float> row = columns.Slice(b * width, width);
                Span<float> output = outputs.Slice(b * outputStride, outputColumns.Length);
                for (int i = 0; i < output.Length; i++) output[i] = outputColumns[i] < 0 ? 0 : row[outputColumns[i]];

                if (!divideBySum) continue;

                Quad total = 0;
                foreach (float value in output) total += value;
                for (int i = 0; i < output.Length; i++) output[i] = (Quad)output[i] / total;
            }
        }

        private sealed record InputGroup(int[] Positions, CompactActivation Activation);

        /// <summary>
        /// Neurons of one level with the same activation: they read <see cref="Sources"/> and write the columns from
        /// <see cref="FirstColumn"/> on
        /// </summary>
        private sealed class Group
        {
            private readonly CompactActivation activation;
            private readonly bool contiguous;

            public Group(GraphPlan.NeuronGroup group, WeightPrecision precision)
            {
                Sources = group.Sources;
                Bias = group.Bias;
                FirstColumn = group.FirstColumn;
                activation = CompactActivation.Of(group.Activation);
                Weights = Sources.Length == 0 ? null : WeightMatrix.Create(group.Weights, Sources.Length, group.Count, precision);
                contiguous = Sources.Select((c, i) => c - i).Distinct().Count() <= 1;
            }

            public int[] Sources { get; }

            public float[] Bias { get; }

            public int FirstColumn { get; }

            public WeightMatrix Weights { get; }

            public void Run(Span<float> columns, int width, Span<float> work, int count)
            {
                for (int b = 0; b < count; b++) Bias.CopyTo(columns.Slice((b * width) + FirstColumn, Bias.Length));

                if (Weights != null)
                {
                    Span<float> sums = columns.Slice(FirstColumn);
                    if (contiguous)
                    {
                        Weights.MultiplyAddBatch(columns.Slice(Sources[0]), width, sums, width, count);
                    }
                    else
                    {
                        int length = Sources.Length;
                        for (int b = 0; b < count; b++)
                        {
                            Span<float> row = columns.Slice(b * width, width);
                            Span<float> gathered = work.Slice(b * length, length);
                            for (int s = 0; s < length; s++) gathered[s] = row[Sources[s]];
                        }

                        Weights.MultiplyAddBatch(work, length, sums, width, count);
                    }
                }

                for (int b = 0; b < count; b++) activation.Apply(columns.Slice((b * width) + FirstColumn, Bias.Length));
            }
        }
    }
}
