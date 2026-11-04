using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Compact
{
    /// <summary>
    /// Removes small weights from a <see cref="GraphPlan"/>, then the neurons that no longer matter: a neuron without
    /// inputs computes the constant activation(bias), which is added to the biases of the neurons it feeds, and a neuron
    /// that is not an output and feeds nothing is dropped. Levels and columns are planned again for what is left
    /// </summary>
    internal static class GraphPruner
    {
        private sealed class Neuron
        {
            public int Column { get; init; }

            public ActivationTypes Activation { get; init; }

            public float Bias { get; set; }

            /// <summary>
            /// Column of the source and weight
            /// </summary>
            public Dictionary<int, float> Inputs { get; } = new Dictionary<int, float>();

            public HashSet<int> Consumers { get; } = new HashSet<int>();

            public bool Output { get; set; }

            public bool Removed { get; set; }
        }

        public static IEnumerable<float[]> Weights(GraphPlan plan) => plan.Levels.SelectMany(x => x).Select(g => g.Weights);

        public static GraphPlan Prune(GraphPlan plan, float threshold)
        {
            var neurons = new Dictionary<int, Neuron>();
            foreach (GraphPlan.NeuronGroup group in plan.Levels.SelectMany(x => x))
            {
                for (int n = 0; n < group.Count; n++)
                {
                    var neuron = new Neuron { Column = group.FirstColumn + n, Activation = group.Activation, Bias = group.Bias[n] };
                    for (int s = 0; s < group.Sources.Length; s++)
                    {
                        float weight = group.Weights[(s * group.Count) + n];
                        if (weight != 0 && Math.Abs(weight) >= threshold) neuron.Inputs.Add(group.Sources[s], weight);
                    }

                    neurons.Add(neuron.Column, neuron);
                }
            }

            foreach (int column in plan.OutputColumns.Where(x => x >= 0)) neurons[column].Output = true;
            foreach (Neuron neuron in neurons.Values)
            {
                foreach (int source in neuron.Inputs.Keys)
                {
                    if (neurons.TryGetValue(source, out Neuron from)) from.Consumers.Add(neuron.Column);
                }
            }

            // columns are in dependency order, so one pass from the first column folds chains of constants
            var pending = new Queue<int>(neurons.Keys.OrderBy(x => x));
            while (pending.Count > 0)
            {
                Neuron neuron = neurons[pending.Dequeue()];
                if (neuron.Removed) continue;

                if (neuron.Inputs.Count == 0 && neuron.Consumers.Count > 0)
                {
                    float value = Activate(neuron.Activation, neuron.Bias);
                    foreach (int consumerColumn in neuron.Consumers)
                    {
                        Neuron consumer = neurons[consumerColumn];
                        consumer.Bias += consumer.Inputs[neuron.Column] * value;
                        consumer.Inputs.Remove(neuron.Column);
                        pending.Enqueue(consumerColumn);
                    }

                    neuron.Consumers.Clear();
                }

                if (!neuron.Output && neuron.Consumers.Count == 0)
                {
                    neuron.Removed = true;
                    foreach (int source in neuron.Inputs.Keys)
                    {
                        if (!neurons.TryGetValue(source, out Neuron from)) continue;

                        from.Consumers.Remove(neuron.Column);
                        pending.Enqueue(source);
                    }
                }
            }

            return Replan(plan, neurons.Values.Where(x => !x.Removed).ToList());
        }

        private static float Activate(ActivationTypes activation, float value)
        {
            if (activation == ActivationTypes.Identity) return value;

            Quad quad = value;
            return Utils.NetworkUtils.GetActivation(activation).Activation(ref quad);
        }

        private static GraphPlan Replan(GraphPlan plan, List<Neuron> neurons)
        {
            var byColumn = neurons.ToDictionary(x => x.Column);
            var depth = new Dictionary<int, int>();
            int Depth(int column)
            {
                if (!byColumn.TryGetValue(column, out Neuron neuron)) return 0; // an input
                if (depth.TryGetValue(column, out int known)) return known;

                int result = 1 + neuron.Inputs.Keys.Select(Depth).DefaultIfEmpty(0).Max();
                depth[column] = result;
                return result;
            }

            var newColumn = new Dictionary<int, int>();
            for (int i = 0; i < plan.InputCount; i++) newColumn[i] = i;

            int width = plan.InputCount;
            var levels = new List<IReadOnlyList<GraphPlan.NeuronGroup>>();
            foreach (var level in neurons.GroupBy(x => Depth(x.Column)).OrderBy(x => x.Key))
            {
                var groups = new List<GraphPlan.NeuronGroup>();
                foreach (var group in level.GroupBy(x => x.Activation))
                {
                    List<Neuron> members = group.OrderBy(x => x.Column).ToList();
                    int[] sources = members.SelectMany(x => x.Inputs.Keys).Distinct().Select(x => newColumn[x]).OrderBy(x => x).ToArray();
                    var row = sources.Select((c, i) => (c, i)).ToDictionary(x => x.c, x => x.i);

                    var weights = new float[sources.Length * members.Count];
                    var bias = new float[members.Count];
                    for (int n = 0; n < members.Count; n++)
                    {
                        bias[n] = members[n].Bias;
                        foreach (var (source, weight) in members[n].Inputs)
                        {
                            weights[(row[newColumn[source]] * members.Count) + n] = weight;
                        }
                    }

                    groups.Add(new GraphPlan.NeuronGroup(sources, weights, bias, group.Key, width));
                    foreach (Neuron member in members) newColumn[member.Column] = width++;
                }

                levels.Add(groups);
            }

            return plan.With(levels, width, plan.OutputColumns.Select(x => x < 0 ? -1 : newColumn[x]).ToArray());
        }
    }
}
