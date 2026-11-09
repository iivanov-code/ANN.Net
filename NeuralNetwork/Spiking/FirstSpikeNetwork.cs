using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using ANN.Net.Abstractions.Spiking;
using ANN.Net.Compact;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Time-to-first-spike network converted from a trained network (as in Rueckauer and Liu 2018, Stanojevic et al.
    /// 2023). Layer d fires in window d of W steps. A neuron that fires in step k of its window stands for the value
    /// s * (W - k) / W of its layer's scale s: in a neuromorphic chip its spike switches on a constant current w / W
    /// into every target, which flows until the end of the window and so delivers w * (W - k) / W. When its own window
    /// starts, a target has collected its whole input z; with a ReLU it then fires as soon as z plus a rising ramp
    /// reaches the threshold, which is in step round(W * (1 - z / s)), earlier for larger inputs. Every neuron
    /// therefore fires at most once, and a neuron with the value 0 not at all.
    /// The simulation computes these spike steps directly instead of going through every step: a spike adds its
    /// value times the weights to its targets, one multiply-add per synapse, with SIMD for consecutive targets and four
    /// spikes at a time
    /// </summary>
    internal sealed class FirstSpikeNetwork : IFirstSpikeNetwork
    {
        private readonly int inputCount;
        private readonly int[] layerStart;
        private readonly int[] layerLevel;
        private readonly CompactActivation[] activations;
        private readonly float[] scales;
        private readonly float[] bias;
        private readonly InputScaling scaling;
        private readonly InputTransform transform;
        private readonly int[] outputs;
        private readonly CompactActivation[] outputActivations;
        private readonly bool divideBySum;
        private readonly int levels;

        private readonly int[] start, target, run;
        private readonly float[] weight;

        private readonly float[] values;
        private readonly float[] potentials;
        private readonly sbyte[] signs;
        private readonly float[] delivered;
        private readonly float[] scratch;
        private readonly int[] steps;
        private int window;

        private FirstSpikeNetwork(int inputCount, int[] layerStart, int[] layerLevel, CompactActivation[] activations, float[] scales, float[] bias,
                                  List<SpikingSynapse> synapses, InputScaling scaling, InputTransform transform, int[] outputs,
                                  CompactActivation[] outputActivations, bool divideBySum, int window)
        {
            this.inputCount = inputCount;
            this.layerStart = layerStart;
            this.layerLevel = layerLevel;
            this.activations = activations;
            this.scales = scales;
            this.bias = bias;
            this.scaling = scaling;
            this.transform = transform;
            this.outputs = outputs;
            this.outputActivations = outputActivations;
            this.divideBySum = divideBySum;
            levels = layerLevel.Length == 0 ? 0 : layerLevel.Max();
            Window = window;

            int count = bias.Length;
            start = new int[count + 1];
            foreach (SpikingSynapse synapse in synapses) start[synapse.Source + 1]++;
            for (int i = 0; i < count; i++) start[i + 1] += start[i];

            target = new int[synapses.Count];
            weight = new float[synapses.Count];
            int[] fill = start.Take(count).ToArray();
            foreach (SpikingSynapse synapse in synapses)
            {
                int k = fill[synapse.Source]++;
                target[k] = synapse.Target;
                weight[k] = synapse.Weight;
            }

            run = new int[count];
            for (int source = 0; source < count; source++)
            {
                run[source] = -1;
                if (start[source] == start[source + 1]) continue;

                bool consecutive = true;
                for (int k = start[source] + 1; k < start[source + 1] && consecutive; k++) consecutive = target[k] == target[k - 1] + 1;
                if (consecutive) run[source] = target[start[source]];
            }

            values = new float[inputCount];
            potentials = new float[count];
            signs = new sbyte[count];
            steps = new int[count];
            delivered = new float[count];
            scratch = new float[Math.Max(1, activations.Length == 0 ? 0 : Enumerable.Range(0, activations.Length).Max(i => layerStart[i + 1] - layerStart[i]))];
        }

        public int InputCount => inputCount;

        public int OutputCount => outputs.Length;

        public int NeuronCount => bias.Length;

        public int SynapseCount => target.Length;

        public IReadOnlyList<int> LayerSizes => new[] { inputCount }.Concat(Enumerable.Range(0, activations.Length).Select(i => layerStart[i + 1] - layerStart[i])).ToArray();

        public int Window
        {
            get => window;
            set
            {
                ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
                window = value;
            }
        }

        /// <summary>
        /// Converts a plan: the inputs are normalized onto [0, 1] (or [-1, 1]) and every neuron's value is divided by
        /// its layer's scale, a high percentile of the layer's activation magnitudes in the calibration inputs. Any
        /// activation works, since a neuron fires according to its activated value; with ReLU the firing is the
        /// plain threshold crossing described above
        /// </summary>
        public static FirstSpikeNetwork Convert(GraphPlan plan, SpikingConversionOptions options)
        {
            var (groups, width, transform, analog, columns) = SpikingConversion.Calibrate(plan, options);

            List<float[]> inputValues = columns.Select(c => c.AsSpan(0, plan.InputCount).ToArray()).ToList();
            InputScaling scaling = options.SignedInputs ? InputScaling.MaxAbs(inputValues) : InputScaling.MinMax(inputValues);

            // value of a column = magnitude * (decoded spike value) + offset
            float[] magnitude = new float[width], offset = new float[width];
            for (int i = 0; i < plan.InputCount; i++)
            {
                magnitude[i] = scaling.Scale[i] > 0 ? 1 / scaling.Scale[i] : 0;
                offset[i] = scaling.Offset[i];
            }

            int[] level = new int[groups.Count];
            for (int l = 0, g = 0; l < plan.Levels.Count; l++)
            {
                for (int i = 0; i < plan.Levels[l].Count; i++) level[g++] = l + 1;
            }

            float[] scales = new float[groups.Count];
            float[] bias = new float[width];
            var synapses = new List<SpikingSynapse>();

            for (int g = 0; g < groups.Count; g++)
            {
                GraphPlan.NeuronGroup group = groups[g];
                scales[g] = SpikingConversion.Scale(group, columns, options.Percentile, magnitude: true);
                for (int n = 0; n < group.Count; n++) magnitude[group.FirstColumn + n] = scales[g];

                double[] biases = group.Bias.Select(x => (double)x).ToArray();
                for (int row = 0; row < group.Sources.Length; row++)
                {
                    int source = group.Sources[row];
                    for (int n = 0; n < group.Count; n++)
                    {
                        float w = group.Weights[(row * group.Count) + n];
                        if (w == 0) continue;

                        biases[n] += (double)w * offset[source];
                        float scaled = w * magnitude[source];
                        if (scaled != 0) synapses.Add(new SpikingSynapse(source, group.FirstColumn + n, scaled));
                    }
                }

                for (int n = 0; n < group.Count; n++) bias[group.FirstColumn + n] = (float)biases[n];
            }

            int[] layerStart = groups.Select(g => g.FirstColumn).Append(width).ToArray();
            int[] outputs = plan.OutputColumns.ToArray();
            CompactActivation identity = CompactActivation.Of(Abstractions.Enums.ActivationTypes.Identity);
            CompactActivation[] outputActivations = outputs.Select(c => c < 0 ? identity : analog[SpikingConversion.GroupOf(groups, c)]).ToArray();

            return new FirstSpikeNetwork(plan.InputCount, layerStart, level, analog, scales, bias, synapses, scaling, transform, outputs,
                                         outputActivations, plan.DivideBySum, options.TimeSteps);
        }

        public float[] Propagate(ReadOnlySpan<float> inputs) => (float[])Run(inputs, Window).Outputs;

        public SpikingRun Run(ReadOnlySpan<float> inputs, int window)
        {
            if (inputs.Length != inputCount) throw new ArgumentException($"Expected {inputCount} inputs, got {inputs.Length}", nameof(inputs));
            ArgumentOutOfRangeException.ThrowIfLessThan(window, 1);

            if (transform != null) transform(inputs, values);
            else inputs.CopyTo(values);
            scaling.Apply(values, values);

            bias.CopyTo(potentials, 0);
            Array.Clear(signs);
            long events = 0;

            for (int i = 0; i < inputCount; i++) delivered[i] = Encode(i, values[i], 0, window);
            events += Deliver(0, inputCount);

            for (int layer = 0; layer < activations.Length; layer++)
            {
                CompactActivation activation = activations[layer];
                float inverse = 1 / scales[layer];
                int stepOffset = layerLevel[layer] * window;
                int first = layerStart[layer], end = layerStart[layer + 1];

                // every input of the layer has arrived: its neurons fire, then their spikes reach the later layers
                Span<float> activated = scratch.AsSpan(0, end - first);
                potentials.AsSpan(first, end - first).CopyTo(activated);
                activation.Apply(activated);
                for (int neuron = first; neuron < end; neuron++)
                {
                    delivered[neuron] = Encode(neuron, activated[neuron - first] * inverse, stepOffset, window);
                }

                events += Deliver(first, end);
            }

            float[] result = new float[outputs.Length];
            for (int i = 0; i < outputs.Length; i++) result[i] = outputs[i] < 0 ? 0 : outputActivations[i].Apply(potentials[outputs[i]]);

            if (divideBySum)
            {
                Quad total = 0;
                foreach (float value in result) total += value;
                for (int i = 0; i < result.Length; i++) result[i] = (Quad)result[i] / total;
            }

            int[] outputSpikes = outputs.Select(n => n < 0 ? 0 : signs[n]).Select(x => (int)x).ToArray();
            int fired = 0;
            foreach (sbyte sign in signs) if (sign != 0) fired++;
            return new SpikingRun(result, outputSpikes, Math.Max(1, levels) * window, fired, events);
        }

        public IReadOnlyList<FirstSpike> GetLastSpikes()
        {
            return Enumerable.Range(0, NeuronCount).Where(n => signs[n] != 0)
                             .Select(n => new FirstSpike(n, steps[n], signs[n]))
                             .OrderBy(s => s.Step).ToList();
        }

        public IReadOnlyList<SpikingSynapse> GetSynapses()
        {
            var result = new List<SpikingSynapse>(target.Length);
            for (int source = 0; source < NeuronCount; source++)
            {
                for (int k = start[source]; k < start[source + 1]; k++) result.Add(new SpikingSynapse(source, target[k], weight[k]));
            }

            return result;
        }

        /// <summary>
        /// Fires a neuron whose value, relative to its layer's scale, is <paramref name="value"/>: in step
        /// W - round(W * |value|) of its window, not at all when that rounds to 0. Returns what its synapses deliver
        /// per unit of weight: the current switched on in step k flows for the remaining W - k steps
        /// </summary>
        private float Encode(int neuron, float value, int stepOffset, int window)
        {
            if (value == 0) return 0;

            // the conversion saturates, NaN giving 0
            int level = (int)((MathF.Min(MathF.Abs(value), 1) * window) + 0.5f);
            if (level == 0) return 0;

            int sign = value < 0 ? -1 : 1;
            signs[neuron] = (sbyte)sign;
            steps[neuron] = stepOffset + window - level;
            return sign * (float)level / window;
        }

        /// <summary>
        /// Adds the spikes of neurons [first, end) to their targets, four sources with the same consecutive targets in
        /// one pass (as the compact networks do), so the targets are loaded and stored once for four spikes
        /// </summary>
        private long Deliver(int first, int end)
        {
            long events = 0;
            int pending = 0;
            Span<int> group = stackalloc int[4];

            for (int neuron = first; neuron < end; neuron++)
            {
                float x = delivered[neuron];
                int count = start[neuron + 1] - start[neuron];
                if (x == 0 || count == 0) continue;

                events += count;
                if (run[neuron] < 0)
                {
                    ReadOnlySpan<int> targets = target.AsSpan(start[neuron], count);
                    ReadOnlySpan<float> weights = weight.AsSpan(start[neuron], count);
                    for (int k = 0; k < count; k++) potentials[targets[k]] += x * weights[k];
                    continue;
                }

                if (pending > 0 && (run[group[0]] != run[neuron] || start[group[0] + 1] - start[group[0]] != count))
                {
                    Flush(group.Slice(0, pending));
                    pending = 0;
                }

                group[pending++] = neuron;
                if (pending == 4)
                {
                    Flush(group);
                    pending = 0;
                }
            }

            Flush(group.Slice(0, pending));
            return events;
        }

        private void Flush(ReadOnlySpan<int> group)
        {
            if (group.Length == 0) return;

            int first = group[0], count = start[first + 1] - start[first];
            Span<float> sums = potentials.AsSpan(run[first], count);

            if (group.Length == 4)
            {
                Axpy4(delivered[group[0]], Row(group[0]), delivered[group[1]], Row(group[1]),
                      delivered[group[2]], Row(group[2]), delivered[group[3]], Row(group[3]), sums);
                return;
            }

            foreach (int neuron in group) Axpy(delivered[neuron], Row(neuron), sums);
        }

        private ReadOnlySpan<float> Row(int neuron) => weight.AsSpan(start[neuron], start[neuron + 1] - start[neuron]);

        private static void Axpy4(float x0, ReadOnlySpan<float> row0, float x1, ReadOnlySpan<float> row1,
                                  float x2, ReadOnlySpan<float> row2, float x3, ReadOnlySpan<float> row3, Span<float> sums)
        {
            int length = sums.Length, i = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
            {
                Vector<float> f0 = new Vector<float>(x0), f1 = new Vector<float>(x1), f2 = new Vector<float>(x2), f3 = new Vector<float>(x3);
                ref float s0 = ref MemoryMarshal.GetReference(row0);
                ref float s1 = ref MemoryMarshal.GetReference(row1);
                ref float s2 = ref MemoryMarshal.GetReference(row2);
                ref float s3 = ref MemoryMarshal.GetReference(row3);
                ref float target = ref MemoryMarshal.GetReference(sums);
                for (; i <= length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    nuint at = (nuint)i;
                    Vector<float> sum = Vector.LoadUnsafe(ref target, at) + (f0 * Vector.LoadUnsafe(ref s0, at));
                    sum += f1 * Vector.LoadUnsafe(ref s1, at);
                    sum += f2 * Vector.LoadUnsafe(ref s2, at);
                    sum += f3 * Vector.LoadUnsafe(ref s3, at);
                    sum.StoreUnsafe(ref target, at);
                }
            }

            for (; i < length; i++)
            {
                float sum = sums[i] + (x0 * row0[i]);
                sum += x1 * row1[i];
                sum += x2 * row2[i];
                sum += x3 * row3[i];
                sums[i] = sum;
            }
        }

        private static void Axpy(float x, ReadOnlySpan<float> row, Span<float> sums)
        {
            int length = sums.Length, i = 0;
            if (Vector.IsHardwareAccelerated && length >= Vector<float>.Count)
            {
                var factor = new Vector<float>(x);
                ref float source = ref MemoryMarshal.GetReference(row);
                ref float target = ref MemoryMarshal.GetReference(sums);
                for (; i <= length - Vector<float>.Count; i += Vector<float>.Count)
                {
                    (Vector.LoadUnsafe(ref target, (nuint)i) + (factor * Vector.LoadUnsafe(ref source, (nuint)i))).StoreUnsafe(ref target, (nuint)i);
                }
            }

            for (; i < length; i++) sums[i] += x * row[i];
        }
    }
}
