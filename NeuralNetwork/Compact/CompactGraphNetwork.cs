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
        private readonly GraphPlan plan;
        private readonly IActivationFunction[] inputActivations;
        private readonly IActivationFunction[][] groupActivations;

        public CompactGraphNetwork(GraphPlan plan)
        {
            this.plan = plan;
            inputActivations = plan.InputGroups.Select(g => Function(g.Activation)).ToArray();
            groupActivations = plan.Levels.Select(level => level.Select(g => Function(g.Activation)).ToArray()).ToArray();
            ParameterCount = plan.Levels.SelectMany(x => x).Sum(g => (long)g.Weights.Length + g.Bias.Length)
                             + ((plan.InputMinimum?.Length ?? 0) * 2);
        }

        public int InputCount => plan.InputCount;

        public int OutputCount => plan.OutputColumns.Length;

        public long ParameterCount { get; }

        private static IActivationFunction Function(ActivationTypes type) => type == ActivationTypes.Identity ? null : NetworkUtils.GetActivation(type);

        public Quad[] Propagate(params Quad[] values)
        {
            ArgumentNullException.ThrowIfNull(values);
            if (values.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} input values but found {values.Length}", nameof(values));

            var columns = new Quad[plan.Width];
            var inputs = new Quad[InputCount];
            for (int i = 0; i < inputs.Length; i++)
            {
                inputs[i] = plan.InputMinimum == null ? values[i] : (values[i] - plan.InputMinimum[i]) / plan.InputRange[i];
            }

            if (plan.InputGroups.Count == 0)
            {
                Array.Copy(inputs, columns, inputs.Length);
            }
            else
            {
                int column = 0;
                for (int g = 0; g < plan.InputGroups.Count; g++)
                {
                    foreach (int position in plan.InputGroups[g].Positions)
                    {
                        columns[column++] = Activate(inputActivations[g], inputs[position]);
                    }
                }
            }

            for (int l = 0; l < plan.Levels.Count; l++)
            {
                for (int g = 0; g < plan.Levels[l].Count; g++)
                {
                    GraphPlan.NeuronGroup group = plan.Levels[l][g];
                    int count = group.Count;
                    for (int n = 0; n < count; n++)
                    {
                        float sum = group.Bias[n];
                        for (int s = 0; s < group.Sources.Length; s++)
                        {
                            sum += columns[group.Sources[s]] * group.Weights[(s * count) + n];
                        }

                        columns[group.FirstColumn + n] = Activate(groupActivations[l][g], sum);
                    }
                }
            }

            var outputs = plan.OutputColumns.Select(c => c < 0 ? (Quad)0 : columns[c]).ToArray();
            if (plan.DivideBySum)
            {
                Quad total = 0;
                foreach (Quad output in outputs) total += output;
                for (int i = 0; i < outputs.Length; i++) outputs[i] /= total;
            }

            return outputs;
        }

        private static Quad Activate(IActivationFunction function, Quad value)
        {
            return function == null ? value : function.Activation(ref value);
        }
    }
}
