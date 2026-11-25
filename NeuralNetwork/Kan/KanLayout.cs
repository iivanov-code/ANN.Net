using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Kan
{
    /// <summary>
    /// Where the nodes and connections of a Kolmogorov-Arnold network lie in its graph. Node and connection ids follow
    /// from the shape alone, so a saved network only keeps the shape: input nodes 0 to n₀ - 1, the bias node, and then
    /// for every layer its feature nodes followed by the nodes of the next layer
    /// </summary>
    internal sealed class KanLayout
    {
        private KanLayout(int[] layerSizes, KanBasisTypes basis, int gridSize, ActivationTypes? baseActivation)
        {
            LayerSizes = layerSizes;
            Basis = basis;
            GridSize = gridSize;
            BaseActivation = baseActivation;
        }

        public int[] LayerSizes { get; }

        public KanBasisTypes Basis { get; }

        public int GridSize { get; }

        public ActivationTypes? BaseActivation { get; }

        public uint BiasNode { get; private set; }

        /// <summary>
        /// Node id of every node of every layer
        /// </summary>
        public uint[][] LayerNodes { get; private set; }

        /// <summary>
        /// Features of node i of layer l: [l][i]
        /// </summary>
        public Feature[][][] Features { get; private set; }

        public int BasisCount => Basis == KanBasisTypes.BSpline ? GridSize + 3 : GridSize + 1;

        /// <summary>
        /// One function of a layer's input node, computed by a hidden node: the base function, or one basis function
        /// whose input connection scales and whose bias connection shifts the input onto its place on the grid
        /// </summary>
        public sealed class Feature
        {
            public uint Node { get; init; }

            public ActivationTypes Activation { get; init; }

            /// <summary>
            /// Connection from the input node of the layer to the feature node
            /// </summary>
            public uint Input { get; init; }

            /// <summary>
            /// Connection from the bias node; null for the base function
            /// </summary>
            public uint? Bias { get; init; }

            /// <summary>
            /// Index of the basis function on the grid; -1 for the base function
            /// </summary>
            public int Index { get; init; }

            /// <summary>
            /// Connection to every node of the next layer, in their order
            /// </summary>
            public uint[] Outputs { get; init; }
        }

        public static KanLayout Create(IList<int> layerSizes, KanBasisTypes basis, int gridSize, ActivationTypes? baseActivation)
        {
            if (layerSizes == null || layerSizes.Count < 2)
                throw new ArgumentException("A Kolmogorov-Arnold network needs at least an input and an output layer", nameof(layerSizes));
            if (layerSizes.Any(x => x < 1)) throw new ArgumentException("Every layer needs at least one node", nameof(layerSizes));
            if (layerSizes[0] > ushort.MaxValue || layerSizes[^1] > ushort.MaxValue)
                throw new ArgumentException($"At most {ushort.MaxValue} inputs and outputs are supported", nameof(layerSizes));
            if (gridSize < 1) throw new ArgumentException("The grid needs at least one interval", nameof(gridSize));
            if (!Enum.IsDefined(basis)) throw new ArgumentException($"Unknown basis {basis}", nameof(basis));

            var layout = new KanLayout(layerSizes.ToArray(), basis, gridSize, baseActivation);
            layout.Assign();
            return layout;
        }

        private void Assign()
        {
            uint node = 0, connection = 0;
            int layers = LayerSizes.Length;
            LayerNodes = new uint[layers][];
            Features = new Feature[layers - 1][][];
            ActivationTypes basisActivation = Basis == KanBasisTypes.BSpline ? ActivationTypes.CubicBSpline : ActivationTypes.Gaussian;

            LayerNodes[0] = Enumerable.Range(0, LayerSizes[0]).Select(_ => node++).ToArray();
            BiasNode = node++;

            for (int layer = 0; layer < layers - 1; layer++)
            {
                int outputs = LayerSizes[layer + 1];
                var features = new Feature[LayerSizes[layer]][];
                for (int i = 0; i < features.Length; i++)
                {
                    var list = new List<Feature>();
                    if (BaseActivation is ActivationTypes baseActivation)
                        list.Add(new Feature { Node = node++, Activation = baseActivation, Input = connection++, Index = -1, Outputs = new uint[outputs] });

                    for (int k = 0; k < BasisCount; k++)
                        list.Add(new Feature { Node = node++, Activation = basisActivation, Input = connection++, Bias = connection++, Index = k, Outputs = new uint[outputs] });

                    features[i] = list.ToArray();
                }

                LayerNodes[layer + 1] = Enumerable.Range(0, outputs).Select(_ => node++).ToArray();
                foreach (Feature feature in features.SelectMany(x => x))
                {
                    for (int j = 0; j < outputs; j++) feature.Outputs[j] = connection++;
                }

                Features[layer] = features;
            }
        }

        /// <summary>
        /// Centre of basis function <paramref name="index"/> on the grid and the distance h between two centres
        /// </summary>
        public (double Centre, double Width) Place(int index, double minimum, double maximum)
        {
            double width = (maximum - minimum) / GridSize;
            // B-spline bumps reach two intervals to each side, so one more centre outside each end covers the grid
            int first = Basis == KanBasisTypes.BSpline ? -1 : 0;
            return (minimum + ((index + first) * width), width);
        }

        public KanDocument ToDocument()
        {
            return new KanDocument { LayerSizes = LayerSizes.ToList(), Basis = Basis, GridSize = GridSize, BaseActivation = BaseActivation };
        }

        public static KanLayout FromDocument(KanDocument document)
        {
            return Create(document.LayerSizes, document.Basis, document.GridSize, document.BaseActivation);
        }

        /// <summary>
        /// The graph of the network with its starting weights
        /// </summary>
        public static (GraphNetworkSettings Graph, KanLayout Layout) Build(KanSettings settings)
        {
            ArgumentNullException.ThrowIfNull(settings);
            if (!float.IsFinite(settings.GridMinimum) || !float.IsFinite(settings.GridMaximum) || settings.GridMaximum <= settings.GridMinimum)
                throw new ArgumentException("The grid maximum must be finite and above the minimum", nameof(settings));
            if (!float.IsFinite(settings.SplineInitScale) || settings.SplineInitScale < 0)
                throw new ArgumentException("The spline starting scale must be finite and not negative", nameof(settings));

            KanLayout layout = Create(settings.LayerSizes, settings.Basis, settings.GridSize, settings.BaseActivation);
            var graph = new GraphNetworkSettings
            {
                InputActivationType = ActivationTypes.Identity,
                HiddenActivationType = ActivationTypes.Identity,
                OutputActivationType = settings.OutputActivationType,
                OutputApplySoftMax = settings.OutputApplySoftMax,
                OptimizerType = settings.OptimizerType,
                LossType = settings.LossType,
                LossOptions = settings.LossOptions,
            };

            foreach (uint input in layout.LayerNodes[0]) graph.Nodes.Add(new GraphNode(input, NodeType.INPUT));
            graph.Nodes.Add(new GraphNode(layout.BiasNode, NodeType.BIAS));

            using IDisposable seed = Utils.NetworkUtils.SeedRandom(settings.Seed);
            Random random = Utils.NetworkUtils.Rand;
            int last = layout.LayerSizes.Length - 1;
            for (int layer = 0; layer < last; layer++)
            {
                uint[] inputs = layout.LayerNodes[layer];
                uint[] outputs = layout.LayerNodes[layer + 1];
                double baseScale = 1 / Math.Sqrt(inputs.Length);

                for (int i = 0; i < inputs.Length; i++)
                {
                    foreach (Feature feature in layout.Features[layer][i])
                    {
                        graph.Nodes.Add(new GraphNode(feature.Node, NodeType.HIDDEN, feature.Activation));
                        if (feature.Bias is uint bias)
                        {
                            (double centre, double width) = layout.Place(feature.Index, settings.GridMinimum, settings.GridMaximum);
                            graph.Connections.Add(new GraphConnection(feature.Input, inputs[i], feature.Node, (Quad)(1 / width), trainable: settings.TrainableGrid));
                            graph.Connections.Add(new GraphConnection(bias, layout.BiasNode, feature.Node, (Quad)(-centre / width), trainable: settings.TrainableGrid));
                        }
                        else
                        {
                            graph.Connections.Add(new GraphConnection(feature.Input, inputs[i], feature.Node, 1, trainable: false));
                        }
                    }
                }

                NodeType type = layer + 1 == last ? NodeType.OUTPUT : NodeType.HIDDEN;
                foreach (uint output in outputs) graph.Nodes.Add(new GraphNode(output, type));

                for (int i = 0; i < inputs.Length; i++)
                {
                    foreach (Feature feature in layout.Features[layer][i])
                    {
                        for (int j = 0; j < outputs.Length; j++)
                        {
                            double weight = feature.Bias == null
                                ? baseScale * ((2 * random.NextDouble()) - 1)
                                : settings.SplineInitScale * Normal(random);
                            graph.Connections.Add(new GraphConnection(feature.Outputs[j], feature.Node, outputs[j], (Quad)weight));
                        }
                    }
                }
            }

            return (graph, layout);
        }

        private static double Normal(Random random)
        {
            // Box-Muller
            double u = 1 - random.NextDouble();
            return Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(2 * Math.PI * random.NextDouble());
        }
    }

    /// <summary>
    /// Shape of a saved Kolmogorov-Arnold network, from which <see cref="KanLayout"/> finds its nodes again
    /// </summary>
    internal sealed class KanDocument
    {
        public List<int> LayerSizes { get; set; }

        public KanBasisTypes Basis { get; set; }

        public int GridSize { get; set; }

        public ActivationTypes? BaseActivation { get; set; }
    }
}
