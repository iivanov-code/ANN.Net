using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    /// <summary>
    /// A Kolmogorov-Arnold network (Liu et al. 2024): every edge between two layers learns a function of one variable
    /// and every node adds up the functions of its incoming edges. The function of an edge is
    /// w · base(x) + Σₖ cₖ Bₖ(x) with fixed basis functions Bₖ on a grid and learned w and cₖ. It is built as a graph
    /// network: every input of a layer feeds one node per basis function (and one for the base function), whose
    /// connections to the node of the next layer hold w and cₖ
    /// </summary>
    public class KanSettings
    {
        /// <summary>
        /// Number of nodes of every layer, inputs first and outputs last, for example [2, 5, 1]
        /// </summary>
        public IList<int> LayerSizes { get; set; } = new List<int>();

        /// <summary>
        /// Number of grid intervals G between <see cref="GridMinimum"/> and <see cref="GridMaximum"/>: more intervals
        /// let every edge learn a more detailed function
        /// </summary>
        public int GridSize { get; set; } = 5;

        /// <summary>
        /// Start of the grid. Inputs of a layer outside the grid reach fewer basis functions (none beyond two intervals),
        /// so scale the inputs to the grid, widen it, or set <see cref="TrainableGrid"/>
        /// </summary>
        public float GridMinimum { get; set; } = -1;

        public float GridMaximum { get; set; } = 1;

        public KanBasisTypes Basis { get; set; } = KanBasisTypes.BSpline;

        /// <summary>
        /// Base function added to the spline of every edge with its own learned weight, SiLU (<see cref="ActivationTypes.SWISH"/>)
        /// as in the original KAN; null for edges made of the basis functions alone
        /// </summary>
        public ActivationTypes? BaseActivation { get; set; } = ActivationTypes.SWISH;

        /// <summary>
        /// When set, the position and width of every basis function are trained as well, so the grid can follow the
        /// values that reach it. By default they stay where the grid puts them, as in the original KAN
        /// </summary>
        public bool TrainableGrid { get; set; }

        /// <summary>
        /// Standard deviation of the random starting coefficients of the basis functions
        /// </summary>
        public float SplineInitScale { get; set; } = 0.1f;

        /// <summary>
        /// Applied to the sums of the output nodes; the identity as in the original KAN
        /// </summary>
        public ActivationTypes OutputActivationType { get; set; } = ActivationTypes.Identity;

        public bool OutputApplySoftMax { get; set; }

        public OptimizerTypes OptimizerType { get; set; } = OptimizerTypes.Adam;

        /// <summary>
        /// Loss the network trains on and <see cref="Interfaces.INetwork.Backpropagate"/> returns
        /// </summary>
        public LossFunctionTypes LossType { get; set; } = LossFunctionTypes.Squared;

        /// <summary>
        /// Parameters of the loss, such as δ of the Huber loss; null for the defaults
        /// </summary>
        public LossOptions LossOptions { get; set; }

        /// <summary>
        /// Seed of the random starting weights, for networks that are built the same every time; random when null
        /// </summary>
        public int? Seed { get; set; }
    }
}
