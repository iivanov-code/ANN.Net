using System.Collections.Generic;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Kolmogorov-Arnold network built by <c>NetworkFactory.BuildKan</c>: a graph network whose edges between two
    /// layers each learn a function of one variable
    /// </summary>
    public interface IKanNetwork : IGraphNetwork
    {
        /// <summary>
        /// Number of nodes of every layer, inputs first and outputs last
        /// </summary>
        IReadOnlyList<int> LayerSizes { get; }

        /// <summary>
        /// The function the edge from node <paramref name="from"/> of layer <paramref name="layer"/> to node
        /// <paramref name="to"/> of layer <paramref name="layer"/> + 1 has learned, at <paramref name="x"/>. Every node
        /// of a layer is the sum of the edge functions of its inputs, so these functions show what the network computes,
        /// for example to plot them or to fit formulas to them
        /// </summary>
        Quad EvaluateEdge(int layer, int from, int to, Quad x);
    }
}
