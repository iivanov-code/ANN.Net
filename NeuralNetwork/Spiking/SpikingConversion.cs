using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.Compact;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Rate-coded conversion of a trained network (Diehl et al. 2015, Rueckauer et al. 2017). Every neuron of the plan
    /// becomes a spiking neuron whose firing rate r encodes its activation as a * r + c: a = lambda, c = 0 for ReLU,
    /// leaky ReLU and the identity, lambda being a high percentile of the activation in the calibration inputs;
    /// a = 1, c = 0 for the sigmoid; a = 2, c = -1 for tanh. A synapse therefore carries w * a(source) / d(target),
    /// where d is what the target's input is divided by (lambda, or 1 for the sigmoid neurons, which take the plain
    /// input), and the offsets c go into the biases. The outputs are decoded from the average input current of the
    /// output neurons, which estimates their input in the original network
    /// </summary>
    internal static class SpikingConversion
    {
        /// <summary>
        /// The groups of a plan in column order, its input transform, and every column of every calibration input
        /// </summary>
        internal sealed record Calibration(List<GraphPlan.NeuronGroup> Groups, int Width, InputTransform Transform,
                                           CompactActivation[] Analog, List<float[]> Columns);

        internal static Calibration Calibrate(GraphPlan plan, SpikingConversionOptions options)
        {
            ArgumentNullException.ThrowIfNull(options);
            if (options.CalibrationInputs == null || options.CalibrationInputs.Count == 0)
                throw new ArgumentException("Conversion needs calibration inputs to normalize the inputs and scale the layers", nameof(options));
            if (options.CalibrationInputs.Any(x => x == null || x.Length != plan.InputCount))
                throw new ArgumentException($"Every calibration input needs {plan.InputCount} values", nameof(options));
            ArgumentOutOfRangeException.ThrowIfLessThan(options.TimeSteps, 1);
            if (!(options.Percentile > 0 && options.Percentile <= 100))
                throw new ArgumentException("The percentile must be greater than 0 and at most 100", nameof(options));

            List<GraphPlan.NeuronGroup> groups = plan.Levels.SelectMany(x => x).ToList();
            int width = plan.InputCount;
            foreach (GraphPlan.NeuronGroup group in groups)
            {
                if (group.FirstColumn != width) throw new NotSupportedException("The columns of the network are not in layer order");
                width += group.Count;
            }

            if (plan.OutputColumns.Any(x => x >= 0 && x < plan.InputCount)) throw new NotSupportedException("An output of the network is an input");

            InputTransform transform = Transform(plan);
            CompactActivation[] analog = groups.Select(g => CompactActivation.Of(g.Activation)).ToArray();
            List<float[]> columns = options.CalibrationInputs.Select(x => Columns(plan, groups, analog, transform, x, width)).ToList();
            return new Calibration(groups, width, transform, analog, columns);
        }

        public static SpikingNetwork Convert(GraphPlan plan, SpikingConversionOptions options)
        {
            var (groups, width, transform, analog, columns) = Calibrate(plan, options);

            // what every column's firing rate is multiplied by and shifted by to give its value, and what its input is divided by
            float[] scale = new float[width], offset = new float[width], divisor = new float[width];
            var layerSizes = new List<int> { plan.InputCount };
            var activations = new List<ISpikingActivation>();

            List<float[]> inputValues = columns.Select(c => c.AsSpan(0, plan.InputCount).ToArray()).ToList();
            InputScaling scaling = options.SignedInputs ? InputScaling.MaxAbs(inputValues) : InputScaling.MinMax(inputValues);
            activations.Add(options.SignedInputs ? new SignedIntegrateAndFireActivation(negativeSlope: 1) : new IntegrateAndFireActivation());
            for (int i = 0; i < plan.InputCount; i++)
            {
                scale[i] = scaling.Scale[i] > 0 ? 1 / scaling.Scale[i] : 0;
                offset[i] = scaling.Offset[i];
            }

            float[] bias = new float[width];
            var synapses = new List<SpikingSynapse>();

            foreach (GraphPlan.NeuronGroup group in groups)
            {
                var (activation, a, c, d) = Spiking(group, columns, options.Percentile);
                layerSizes.Add(group.Count);
                activations.Add(activation);

                for (int n = 0; n < group.Count; n++)
                {
                    int column = group.FirstColumn + n;
                    scale[column] = a;
                    offset[column] = c;
                    divisor[column] = d;
                }

                double[] biases = group.Bias.Select(x => (double)x).ToArray();
                for (int row = 0; row < group.Sources.Length; row++)
                {
                    int source = group.Sources[row];
                    for (int n = 0; n < group.Count; n++)
                    {
                        float weight = group.Weights[(row * group.Count) + n];
                        if (weight == 0) continue;

                        biases[n] += (double)weight * offset[source];
                        float scaled = weight * scale[source] / d;
                        if (scaled != 0) synapses.Add(new SpikingSynapse(source, group.FirstColumn + n, scaled));
                    }
                }

                for (int n = 0; n < group.Count; n++) bias[group.FirstColumn + n] = (float)(biases[n] / d);
            }

            int[] outputs = plan.OutputColumns.ToArray();
            CompactActivation identity = CompactActivation.Of(ActivationTypes.Identity);
            CompactActivation[] outputActivations = outputs.Select(column => column < 0 ? identity : analog[GroupOf(groups, column)]).ToArray();
            float[] outputDivisors = outputs.Select(column => column < 0 ? 1 : divisor[column]).ToArray();
            var readout = SpikingReadout.Potentials(outputs, outputDivisors, outputActivations, plan.DivideBySum);

            return new SpikingNetwork(layerSizes, activations, layerSizes.Select(_ => false).ToArray(), bias, synapses, scaling, transform,
                                      readout, options.TimeSteps, null);
        }

        /// <summary>
        /// Spiking neuron of a group and the factors of its encoding: value = a * rate + c, input divided by d
        /// </summary>
        private static (ISpikingActivation Activation, float A, float C, float D) Spiking(GraphPlan.NeuronGroup group, List<float[]> columns, double percentile)
        {
            switch (group.Activation)
            {
                case ActivationTypes.RELU:
                    {
                        float lambda = Scale(group, columns, percentile, magnitude: false);
                        return (new IntegrateAndFireActivation(), lambda, 0, lambda);
                    }
                case ActivationTypes.LeakyRELU:
                    {
                        float lambda = Scale(group, columns, percentile, magnitude: true);
                        return (new SignedIntegrateAndFireActivation(negativeSlope: 0.01f), lambda, 0, lambda);
                    }
                case ActivationTypes.Identity:
                    {
                        float lambda = Scale(group, columns, percentile, magnitude: true);
                        return (new SignedIntegrateAndFireActivation(negativeSlope: 1), lambda, 0, lambda);
                    }
                case ActivationTypes.Sigmoid:
                    return (new SigmoidRateActivation(), 1, 0, 1);
                case ActivationTypes.HyperbolicTangens:
                    // tanh(x) = 2 * sigmoid(2x) - 1
                    return (new SigmoidRateActivation(gain: 2), 2, -1, 1);
                default:
                    throw new NotSupportedException($"{group.Activation} neurons cannot be converted to spiking neurons; " +
                                                    "supported are RELU, LeakyRELU, Identity, Sigmoid and HyperbolicTangens");
            }
        }

        /// <summary>
        /// The percentile of the positive activations (or of the magnitudes) of a group in the calibration inputs; 1 when
        /// there are none
        /// </summary>
        internal static float Scale(GraphPlan.NeuronGroup group, List<float[]> columns, double percentile, bool magnitude)
        {
            var values = new List<float>();
            foreach (float[] row in columns)
            {
                for (int n = 0; n < group.Count; n++)
                {
                    float value = row[group.FirstColumn + n];
                    if (magnitude) value = Math.Abs(value);
                    if (value > 0 && float.IsFinite(value)) values.Add(value);
                }
            }

            if (values.Count == 0) return 1;

            values.Sort();
            int index = (int)Math.Ceiling(percentile / 100 * values.Count) - 1;
            return values[Math.Clamp(index, 0, values.Count - 1)];
        }

        internal static int GroupOf(List<GraphPlan.NeuronGroup> groups, int column)
        {
            return groups.FindIndex(g => column >= g.FirstColumn && column < g.FirstColumn + g.Count);
        }

        /// <summary>
        /// The input normalization and input activations of the plan, as CompactGraphNetwork applies them; null when the
        /// inputs are used as they are
        /// </summary>
        private static InputTransform Transform(GraphPlan plan)
        {
            if (plan.InputMinimum == null && plan.InputGroups.Count == 0) return null;

            float[] minimum = plan.InputMinimum, range = plan.InputRange;
            var groups = plan.InputGroups.Select(g => (g.Positions, Activation: CompactActivation.Of(g.Activation))).ToArray();
            int count = plan.InputCount;
            float[] buffer = new float[count];

            // like the network, the transform must not be used by several threads at once
            return (inputs, values) =>
            {
                for (int i = 0; i < count; i++) buffer[i] = minimum == null ? inputs[i] : (inputs[i] - minimum[i]) / range[i];

                if (groups.Length == 0)
                {
                    buffer.CopyTo(values);
                    return;
                }

                int column = 0;
                foreach (var (positions, activation) in groups)
                {
                    foreach (int position in positions) values[column++] = activation.Apply(buffer[position]);
                }
            };
        }

        /// <summary>
        /// Every column of the plan for one input, computed with floats
        /// </summary>
        private static float[] Columns(GraphPlan plan, List<GraphPlan.NeuronGroup> groups, CompactActivation[] activations,
                                       InputTransform transform, float[] input, int width)
        {
            float[] row = new float[width];
            if (transform != null) transform(input, row.AsSpan(0, plan.InputCount));
            else input.CopyTo(row, 0);

            for (int g = 0; g < groups.Count; g++)
            {
                GraphPlan.NeuronGroup group = groups[g];
                for (int n = 0; n < group.Count; n++)
                {
                    float sum = group.Bias[n];
                    for (int r = 0; r < group.Sources.Length; r++) sum += row[group.Sources[r]] * group.Weights[(r * group.Count) + n];
                    row[group.FirstColumn + n] = activations[g].Apply(sum);
                }
            }

            return row;
        }
    }
}
