using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using ANN.Net.Abstractions.Spiking;

namespace ANN.Net.Spiking
{
    /// <summary>
    /// Turns the raw inputs of a network into the values its input neurons encode
    /// </summary>
    internal delegate void InputTransform(ReadOnlySpan<float> inputs, Span<float> values);

    /// <summary>
    /// Event-driven spiking network. Neurons are numbered across the network, the inputs first, then layer after
    /// layer. The synapses are kept in flat arrays grouped by source (8 bytes each), so a spike reads the weights of
    /// its source one after another; when those go to consecutive neurons, as in a fully connected layer, they are
    /// added with SIMD. A synapse into a later layer delivers in the same step, any other in the next one
    /// </summary>
    internal sealed class SpikingNetwork : ISpikingNetwork
    {
        private readonly int[] layerStart;
        private readonly int[] layerOf;
        private readonly ISpikingActivation[] activations;
        private readonly bool[] learns;
        private readonly float[] bias;
        private readonly InputScaling scaling;
        private readonly InputTransform transform;
        private readonly SpikingReadout readout;

        private readonly int[] forwardStart, forwardTarget, forwardRun;
        private readonly float[] forwardWeight;
        private readonly int[] recurrentStart, recurrentTarget;
        private readonly float[] recurrentWeight;

        /// <summary>
        /// Outputs read by every layer: (output, neuron)
        /// </summary>
        private readonly (int Output, int Neuron)[][] layerOutputs;

        private readonly SpikingNeuronState[] states;
        private readonly sbyte[] spikes;
        private readonly int[] spikeCounts;
        private readonly float[] values;
        private readonly float[] outputCurrents;
        private float[] current, next;

        // STDP: incoming learning synapses of every neuron as positions in the forward arrays, and two traces per neuron
        private int[] incomingStart, incomingSynapse, incomingSource;
        private float[] preTrace, postTrace, preDecay, postDecay;
        private int[] preTime, postTime;
        private StdpOptions plasticity;
        private int timeSteps;
        private int time;
        private long spikeTotal, events;

        /// <param name="layerSizes">Neurons of every layer, layer 0 being the inputs</param>
        /// <param name="activations">Spike function of every layer</param>
        /// <param name="learns">Whether the synapses into a layer learn</param>
        /// <param name="bias">Current every neuron receives every step, null for none</param>
        /// <param name="scaling">Normalization of the input values, null to use them as they are</param>
        /// <param name="transform">Turns raw inputs into input values before the scaling, null to use them as they are</param>
        public SpikingNetwork(IReadOnlyList<int> layerSizes, IReadOnlyList<ISpikingActivation> activations, IReadOnlyList<bool> learns,
                              float[] bias, IEnumerable<SpikingSynapse> synapses, InputScaling scaling, InputTransform transform,
                              SpikingReadout readout, int timeSteps, StdpOptions plasticity)
        {
            ArgumentNullException.ThrowIfNull(layerSizes);
            ArgumentNullException.ThrowIfNull(synapses);
            ArgumentNullException.ThrowIfNull(readout);
            if (layerSizes.Count < 2) throw new ArgumentException("A spiking network needs an input layer and at least one more", nameof(layerSizes));
            if (layerSizes.Any(x => x <= 0)) throw new ArgumentException("Every layer needs at least one neuron", nameof(layerSizes));
            if (activations.Count != layerSizes.Count || learns.Count != layerSizes.Count)
                throw new ArgumentException("Every layer needs an activation and a learning flag", nameof(activations));
            if (activations.Any(x => x == null)) throw new ArgumentException("Every layer needs an activation", nameof(activations));

            layerStart = new int[layerSizes.Count + 1];
            for (int i = 0; i < layerSizes.Count; i++) layerStart[i + 1] = layerStart[i] + layerSizes[i];

            int count = layerStart[^1];
            if (bias != null && bias.Length != count) throw new ArgumentException("The bias needs one value per neuron", nameof(bias));
            if (scaling != null && scaling.Count != layerSizes[0]) throw new ArgumentException("The scaling needs one value per input", nameof(scaling));

            layerOf = new int[count];
            for (int layer = 0; layer < layerSizes.Count; layer++) layerOf.AsSpan(layerStart[layer], layerSizes[layer]).Fill(layer);

            this.activations = activations.ToArray();
            this.learns = learns.ToArray();
            this.bias = bias;
            this.scaling = scaling;
            this.transform = transform;
            this.readout = readout;
            TimeSteps = timeSteps;
            Plasticity = plasticity ?? new StdpOptions();

            var forward = new List<SpikingSynapse>();
            var recurrent = new List<SpikingSynapse>();
            foreach (SpikingSynapse synapse in synapses)
            {
                if ((uint)synapse.Source >= (uint)count || (uint)synapse.Target >= (uint)count)
                    throw new ArgumentException($"Synapse {synapse} connects a neuron that does not exist", nameof(synapses));
                if (layerOf[synapse.Target] == 0) throw new ArgumentException($"Synapse {synapse} ends at an input neuron", nameof(synapses));
                if (!float.IsFinite(synapse.Weight)) throw new ArgumentException($"Synapse {synapse} has no finite weight", nameof(synapses));

                (layerOf[synapse.Target] > layerOf[synapse.Source] ? forward : recurrent).Add(synapse);
            }

            (forwardStart, forwardTarget, forwardWeight) = Group(forward, count);
            (recurrentStart, recurrentTarget, recurrentWeight) = Group(recurrent, count);
            forwardRun = Runs(forwardStart, forwardTarget);

            if (readout.Neurons.Any(x => x >= count)) throw new ArgumentException("An output neuron does not exist", nameof(readout));
            layerOutputs = Enumerable.Range(0, layerSizes.Count)
                                     .Select(layer => readout.Neurons.Select((neuron, output) => (output, neuron))
                                                                     .Where(x => x.neuron >= 0 && layerOf[x.neuron] == layer)
                                                                     .ToArray())
                                     .ToArray();

            states = new SpikingNeuronState[count];
            spikes = new sbyte[count];
            spikeCounts = new int[count];
            values = new float[layerSizes[0]];
            outputCurrents = new float[readout.Neurons.Length];
            current = new float[count];
            next = new float[count];
        }

        public int InputCount => layerStart[1];

        public int OutputCount => readout.Neurons.Length;

        public int NeuronCount => layerStart[^1];

        public int SynapseCount => forwardTarget.Length + recurrentTarget.Length;

        public IReadOnlyList<int> LayerSizes => Enumerable.Range(0, activations.Length).Select(i => layerStart[i + 1] - layerStart[i]).ToArray();

        public int TimeSteps
        {
            get => timeSteps;
            set
            {
                ArgumentOutOfRangeException.ThrowIfLessThan(value, 1);
                timeSteps = value;
            }
        }

        public StdpOptions Plasticity
        {
            get => plasticity;
            set
            {
                ArgumentNullException.ThrowIfNull(value);
                plasticity = value;
            }
        }

        public ISpikingActivation GetActivation(int layer) => activations[layer];

        public void SetActivation(int layer, ISpikingActivation activation)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(layer);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(layer, activations.Length);
            activations[layer] = activation ?? throw new ArgumentNullException(nameof(activation));
        }

        public float[] Propagate(ReadOnlySpan<float> inputs) => (float[])Run(inputs, TimeSteps).Outputs;

        public SpikingRun Learn(ReadOnlySpan<float> inputs) => Run(inputs, TimeSteps, learn: true);

        public SpikingRun Run(ReadOnlySpan<float> inputs, int timeSteps, bool learn = false)
        {
            if (inputs.Length != InputCount)
                throw new ArgumentException($"Expected {InputCount} inputs, got {inputs.Length}", nameof(inputs));
            ArgumentOutOfRangeException.ThrowIfLessThan(timeSteps, 1);

            Restart();
            if (transform != null) transform(inputs, values);
            else inputs.CopyTo(values);
            if (scaling != null) scaling.Apply(values, values);

            if (learn) PrepareLearning();

            for (int step = 0; step < timeSteps; step++)
            {
                Step(learn);
            }

            if (learn && Plasticity.IncomingWeightSum > 0) NormalizeIncoming(Plasticity.IncomingWeightSum);

            int[] outputSpikes = readout.Neurons.Select(n => n < 0 ? 0 : spikeCounts[n]).ToArray();
            float[] outputs = new float[OutputCount];
            readout.Decode(outputCurrents, outputSpikes, timeSteps, outputs);

            return new SpikingRun(outputs, outputSpikes, timeSteps, spikeTotal, events);
        }

        public IReadOnlyList<SpikingSynapse> GetSynapses()
        {
            var result = new List<SpikingSynapse>(SynapseCount);
            for (int source = 0; source < NeuronCount; source++)
            {
                for (int k = forwardStart[source]; k < forwardStart[source + 1]; k++) result.Add(new SpikingSynapse(source, forwardTarget[k], forwardWeight[k]));
                for (int k = recurrentStart[source]; k < recurrentStart[source + 1]; k++) result.Add(new SpikingSynapse(source, recurrentTarget[k], recurrentWeight[k]));
            }

            return result;
        }

        /// <summary>
        /// Neurons start from rest; the adaptation of their thresholds stays
        /// </summary>
        private void Restart()
        {
            for (int i = 0; i < states.Length; i++) states[i].Restart();
            Array.Clear(current);
            Array.Clear(next);
            Array.Clear(spikeCounts);
            Array.Clear(outputCurrents);
            time = 0;
            spikeTotal = 0;
            events = 0;
        }

        private void Step(bool learn)
        {
            time++;

            for (int layer = 0; layer < activations.Length; layer++)
            {
                int start = layerStart[layer], count = layerStart[layer + 1] - start;
                Span<float> currents = current.AsSpan(start, count);

                if (layer == 0) values.CopyTo(currents);
                else if (bias != null) Add(currents, bias.AsSpan(start, count));

                foreach (var (output, neuron) in layerOutputs[layer]) outputCurrents[output] += current[neuron];

                int fired = activations[layer].Step(states.AsSpan(start, count), currents, spikes.AsSpan(start, count));
                currents.Clear();
                if (fired == 0) continue;

                for (int neuron = start; neuron < start + count; neuron++)
                {
                    if (spikes[neuron] != 0) Fire(neuron, spikes[neuron], learn);
                }
            }

            // what the recurrent synapses delivered becomes the current of the next step; every current is 0 again
            (current, next) = (next, current);
        }

        private void Fire(int neuron, int spike, bool learn)
        {
            spikeCounts[neuron] += spike;
            spikeTotal++;

            int start = forwardStart[neuron], count = forwardStart[neuron + 1] - start;
            if (count > 0)
            {
                ReadOnlySpan<float> weights = forwardWeight.AsSpan(start, count);
                int run = forwardRun[neuron];

                if (run >= 0) Add(current.AsSpan(run, count), weights, spike);
                else Scatter(current, forwardTarget.AsSpan(start, count), weights, spike);

                events += count;
            }

            start = recurrentStart[neuron];
            count = recurrentStart[neuron + 1] - start;
            if (count > 0)
            {
                Scatter(next, recurrentTarget.AsSpan(start, count), recurrentWeight.AsSpan(start, count), spike);
                events += count;
            }

            if (learn) Plasticize(neuron);
        }

        private static void Scatter(float[] currents, ReadOnlySpan<int> targets, ReadOnlySpan<float> weights, int spike)
        {
            if (spike > 0)
            {
                for (int i = 0; i < targets.Length; i++) currents[targets[i]] += weights[i];
            }
            else
            {
                for (int i = 0; i < targets.Length; i++) currents[targets[i]] -= weights[i];
            }
        }

        private static void Add(Span<float> currents, ReadOnlySpan<float> weights, int spike = 1)
        {
            int i = 0;
            if (Vector.IsHardwareAccelerated && currents.Length >= Vector<float>.Count)
            {
                Span<Vector<float>> target = MemoryMarshal.Cast<float, Vector<float>>(currents);
                ReadOnlySpan<Vector<float>> source = MemoryMarshal.Cast<float, Vector<float>>(weights);

                if (spike > 0)
                {
                    for (int v = 0; v < target.Length; v++) target[v] += source[v];
                }
                else
                {
                    for (int v = 0; v < target.Length; v++) target[v] -= source[v];
                }

                i = target.Length * Vector<float>.Count;
            }

            if (spike > 0)
            {
                for (; i < currents.Length; i++) currents[i] += weights[i];
            }
            else
            {
                for (; i < currents.Length; i++) currents[i] -= weights[i];
            }
        }

        private void PrepareLearning()
        {
            Plasticity.Validate();
            int count = NeuronCount;

            if (incomingStart == null)
            {
                incomingStart = new int[count + 1];
                for (int k = 0; k < forwardTarget.Length; k++)
                {
                    if (learns[layerOf[forwardTarget[k]]]) incomingStart[forwardTarget[k] + 1]++;
                }

                for (int i = 0; i < count; i++) incomingStart[i + 1] += incomingStart[i];

                incomingSynapse = new int[incomingStart[^1]];
                incomingSource = new int[incomingStart[^1]];
                int[] fill = incomingStart.Take(count).ToArray();
                for (int source = 0; source < count; source++)
                {
                    for (int k = forwardStart[source]; k < forwardStart[source + 1]; k++)
                    {
                        int target = forwardTarget[k];
                        if (!learns[layerOf[target]]) continue;

                        incomingSynapse[fill[target]] = k;
                        incomingSource[fill[target]++] = source;
                    }
                }

                preTrace = new float[count];
                postTrace = new float[count];
                preTime = new int[count];
                postTime = new int[count];
            }

            preDecay = DecayTable(Plasticity.PotentiationTimeConstant);
            postDecay = DecayTable(Plasticity.DepressionTimeConstant);
            Array.Clear(preTrace);
            Array.Clear(postTrace);
            Array.Clear(preTime);
            Array.Clear(postTime);
        }

        /// <summary>
        /// exp(-steps / tau) for every number of steps until it is negligible
        /// </summary>
        private static float[] DecayTable(float tau)
        {
            int length = (int)Math.Min(100_000, Math.Ceiling(tau * 16)) + 1;
            float[] table = new float[length];
            for (int i = 0; i < length; i++) table[i] = MathF.Exp(-i / tau);
            return table;
        }

        private static float Trace(float[] trace, int[] times, float[] decay, int neuron, int now)
        {
            int elapsed = now - times[neuron];
            return elapsed < decay.Length ? trace[neuron] * decay[elapsed] : 0;
        }

        /// <summary>
        /// Pair-based STDP with all-to-all traces: a firing neuron strengthens its incoming learning synapses by the
        /// traces of their sources and weakens its outgoing ones by the traces of their targets
        /// </summary>
        private void Plasticize(int neuron)
        {
            StdpOptions options = Plasticity;
            float minimum = options.MinimumWeight, maximum = options.MaximumWeight;

            if (options.PotentiationRate > 0)
            {
                for (int i = incomingStart[neuron]; i < incomingStart[neuron + 1]; i++)
                {
                    float trace = Trace(preTrace, preTime, preDecay, incomingSource[i], time);
                    if (trace == 0) continue;

                    int k = incomingSynapse[i];
                    float weight = forwardWeight[k];
                    float change = options.PotentiationRate * trace * (options.SoftBounds ? maximum - weight : 1);
                    forwardWeight[k] = Math.Clamp(weight + change, minimum, maximum);
                }
            }

            if (options.DepressionRate > 0)
            {
                for (int k = forwardStart[neuron]; k < forwardStart[neuron + 1]; k++)
                {
                    int target = forwardTarget[k];
                    if (!learns[layerOf[target]]) continue;

                    float trace = Trace(postTrace, postTime, postDecay, target, time);
                    if (trace == 0) continue;

                    float weight = forwardWeight[k];
                    float change = options.DepressionRate * trace * (options.SoftBounds ? weight - minimum : 1);
                    forwardWeight[k] = Math.Clamp(weight - change, minimum, maximum);
                }
            }

            preTrace[neuron] = Trace(preTrace, preTime, preDecay, neuron, time) + 1;
            preTime[neuron] = time;
            postTrace[neuron] = Trace(postTrace, postTime, postDecay, neuron, time) + 1;
            postTime[neuron] = time;
        }

        private void NormalizeIncoming(float sum)
        {
            for (int neuron = 0; neuron < NeuronCount; neuron++)
            {
                int start = incomingStart[neuron], end = incomingStart[neuron + 1];
                if (start == end) continue;

                float total = 0;
                for (int i = start; i < end; i++) total += forwardWeight[incomingSynapse[i]];
                if (!(total > 0)) continue;

                float factor = sum / total;
                for (int i = start; i < end; i++)
                {
                    int k = incomingSynapse[i];
                    forwardWeight[k] = Math.Clamp(forwardWeight[k] * factor, Plasticity.MinimumWeight, Plasticity.MaximumWeight);
                }
            }
        }

        /// <summary>
        /// The synapses grouped by source, in their given order within a source
        /// </summary>
        private static (int[] Start, int[] Target, float[] Weight) Group(List<SpikingSynapse> synapses, int count)
        {
            int[] start = new int[count + 1];
            foreach (SpikingSynapse synapse in synapses) start[synapse.Source + 1]++;
            for (int i = 0; i < count; i++) start[i + 1] += start[i];

            int[] target = new int[synapses.Count];
            float[] weight = new float[synapses.Count];
            int[] fill = start.Take(count).ToArray();
            foreach (SpikingSynapse synapse in synapses)
            {
                int k = fill[synapse.Source]++;
                target[k] = synapse.Target;
                weight[k] = synapse.Weight;
            }

            return (start, target, weight);
        }

        /// <summary>
        /// First target of every source whose targets are consecutive neurons, otherwise -1
        /// </summary>
        private static int[] Runs(int[] start, int[] target)
        {
            int[] runs = new int[start.Length - 1];
            for (int source = 0; source < runs.Length; source++)
            {
                runs[source] = -1;
                int first = start[source], end = start[source + 1];
                if (first == end) continue;

                bool consecutive = true;
                for (int k = first + 1; k < end && consecutive; k++) consecutive = target[k] == target[k - 1] + 1;
                if (consecutive) runs[source] = target[first];
            }

            return runs;
        }
    }
}
