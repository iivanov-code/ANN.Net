using System;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Utils;

namespace ANN.Net.Compact
{
    /// <summary>
    /// Runs a <see cref="GraphPlan"/>: one array of columns per call, filled level by level
    /// </summary>
    internal sealed class CompactGraphNetwork : ICompactNetwork
    {
        [ThreadStatic]
        private static float[] buffer;

        private readonly int inputCount;
        private readonly float[] inputMinimum, inputRange;
        private readonly InputGroup[] inputGroups;
        private readonly Group[][] levels;
        private readonly int width;
        private readonly int[] outputColumns;
        private readonly bool divideBySum;
        private readonly int bufferLength;

        public CompactGraphNetwork(GraphPlan plan, CompactOptions options = null)
        {
            options ??= new CompactOptions();
            Precision = options.Precision;

            inputCount = plan.InputCount;
            inputMinimum = plan.InputMinimum;
            inputRange = plan.InputRange;
            inputGroups = plan.InputGroups.Select(g => new InputGroup(g.Positions, Function(g.Activation))).ToArray();
            levels = plan.Levels.Select(level => level.Select(g => new Group(g, options.Precision)).ToArray()).ToArray();
            width = plan.Width;
            outputColumns = plan.OutputColumns;
            divideBySum = plan.DivideBySum;

            Group[] groups = levels.SelectMany(x => x).ToArray();
            bufferLength = width + Math.Max(inputCount, groups.Select(g => g.Sources.Length).DefaultIfEmpty(0).Max());
            ParameterCount = groups.Sum(g => (g.Weights?.StoredValues ?? 0) + g.Bias.Length) + ((inputMinimum?.Length ?? 0) * 2);
            SizeInBytes = groups.Sum(g => (g.Weights?.SizeInBytes ?? 0) + (4L * (g.Bias.Length + g.Sources.Length)))
                          + ((inputMinimum?.Length ?? 0) * 8L) + (4L * outputColumns.Length);
        }

        public int InputCount => inputCount;

        public int OutputCount => outputColumns.Length;

        public long ParameterCount { get; }

        public WeightPrecision Precision { get; }

        public long SizeInBytes { get; }

        private static IActivationFunction Function(ActivationTypes type) => type == ActivationTypes.Identity ? null : NetworkUtils.GetActivation(type);

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} input values but found {values.Length}", nameof(values));

            if (buffer == null || buffer.Length < bufferLength) buffer = new float[bufferLength];
            Span<float> columns = buffer.AsSpan(0, width);
            Span<float> work = buffer.AsSpan(width);

            Span<float> inputs = work.Slice(0, inputCount);
            for (int i = 0; i < inputCount; i++)
            {
                inputs[i] = inputMinimum == null ? values[i] : (values[i] - inputMinimum[i]) / inputRange[i];
            }

            if (inputGroups.Length == 0)
            {
                inputs.CopyTo(columns);
            }
            else
            {
                int column = 0;
                foreach (InputGroup group in inputGroups)
                {
                    foreach (int position in group.Positions)
                    {
                        columns[column++] = Activate(group.Activation, inputs[position]);
                    }
                }
            }

            foreach (Group[] level in levels)
            {
                foreach (Group group in level)
                {
                    group.Run(columns, work);
                }
            }

            var outputs = new Quad[outputColumns.Length];
            for (int i = 0; i < outputs.Length; i++) outputs[i] = outputColumns[i] < 0 ? 0 : columns[outputColumns[i]];

            if (divideBySum)
            {
                Quad total = 0;
                foreach (Quad output in outputs) total += output;
                for (int i = 0; i < outputs.Length; i++) outputs[i] /= total;
            }

            return outputs;
        }

        private static float Activate(IActivationFunction function, float value)
        {
            Quad quad = value;
            return function == null ? value : function.Activation(ref quad);
        }

        private sealed record InputGroup(int[] Positions, IActivationFunction Activation);

        /// <summary>
        /// Neurons of one level with the same activation: they read <see cref="Sources"/> and write the columns from
        /// <see cref="FirstColumn"/> on
        /// </summary>
        private sealed class Group
        {
            private readonly IActivationFunction activation;
            private readonly bool contiguous;

            public Group(GraphPlan.NeuronGroup group, WeightPrecision precision)
            {
                Sources = group.Sources;
                Bias = group.Bias;
                FirstColumn = group.FirstColumn;
                activation = Function(group.Activation);
                Weights = Sources.Length == 0 ? null : WeightMatrix.Create(group.Weights, Sources.Length, group.Count, precision);
                contiguous = Sources.Select((c, i) => c - i).Distinct().Count() <= 1;
            }

            public int[] Sources { get; }

            public float[] Bias { get; }

            public int FirstColumn { get; }

            public WeightMatrix Weights { get; }

            public void Run(Span<float> columns, Span<float> work)
            {
                Span<float> sums = columns.Slice(FirstColumn, Bias.Length);
                Bias.CopyTo(sums);

                if (Weights != null)
                {
                    ReadOnlySpan<float> input;
                    if (contiguous)
                    {
                        input = columns.Slice(Sources[0], Sources.Length);
                    }
                    else
                    {
                        Span<float> gathered = work.Slice(0, Sources.Length);
                        for (int s = 0; s < Sources.Length; s++) gathered[s] = columns[Sources[s]];
                        input = gathered;
                    }

                    Weights.MultiplyAdd(input, sums);
                }

                if (activation == null) return;

                for (int n = 0; n < sums.Length; n++)
                {
                    Quad value = sums[n];
                    sums[n] = activation.Activation(ref value);
                }
            }
        }
    }
}
