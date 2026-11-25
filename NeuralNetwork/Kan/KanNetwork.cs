using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Connections;
using ANN.Net.Utils;

namespace ANN.Net.Kan
{
    /// <summary>
    /// Kolmogorov-Arnold network as a graph network: every input of a layer feeds one hidden node per basis function
    /// (and one for the base function), and the connections from these nodes to the nodes of the next layer hold the
    /// learned coefficients. The graph trains, saves, compacts and exports like any other graph network
    /// </summary>
    [Serializable]
    internal sealed class KanNetwork : GraphNetwork, IKanNetwork
    {
        private KanNetwork(GraphNetwork graph, KanLayout layout)
            : base(graph.Network, new Dictionary<uint, ActiveSynapse>(graph.Synapses), graph.HiddenNodeIds.ToList())
        {
            Layout = layout;
            foreach (uint id in layout.Features.SelectMany(x => x).SelectMany(x => x)
                                      .SelectMany(x => x.Outputs.Append(x.Input).Concat(x.Bias is uint bias ? new[] { bias } : Array.Empty<uint>())))
            {
                if (!graph.Synapses.ContainsKey(id))
                    throw new ArgumentException($"Connection {id} of the Kolmogorov-Arnold network is missing", nameof(graph));
            }
        }

        internal KanLayout Layout { get; }

        public IReadOnlyList<int> LayerSizes => Layout.LayerSizes;

        internal static KanNetwork Create(GraphNetwork graph, KanLayout layout) => new KanNetwork(graph, layout);

        public Quad EvaluateEdge(int layer, int from, int to, Quad x)
        {
            if (layer < 0 || layer >= Layout.Features.Length) throw new ArgumentOutOfRangeException(nameof(layer));
            if (from < 0 || from >= Layout.LayerSizes[layer]) throw new ArgumentOutOfRangeException(nameof(from));
            if (to < 0 || to >= Layout.LayerSizes[layer + 1]) throw new ArgumentOutOfRangeException(nameof(to));

            IReadOnlyDictionary<uint, ActiveSynapse> synapses = Synapses;
            double sum = 0;
            foreach (KanLayout.Feature feature in Layout.Features[layer][from])
            {
                Quad value = (synapses[feature.Input].Weight * x) + (feature.Bias is uint bias ? synapses[bias].Weight : 0);
                sum += (double)synapses[feature.Outputs[to]].Weight * NetworkUtils.GetActivation(feature.Activation).Activation(ref value);
            }

            return (Quad)sum;
        }
    }
}
