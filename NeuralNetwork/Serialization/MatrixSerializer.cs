using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Matrix;

namespace ANN.Net.Serialization
{
    /// <summary>
    /// Writes a <see cref="MatrixNetwork"/> as the layered network document the equivalent neuron-graph network would
    /// give, without building that network, so ONNX export and compacting work unchanged
    /// </summary>
    internal static class MatrixSerializer
    {
        public static NetworkDocument ToDocument(MatrixNetwork network, string format, int version)
        {
            var document = new NetworkDocument
            {
                Format = format,
                Version = version,
                Kind = NetworkKind.Layered,
                Matrix = true,
                LossType = network.LossType,
                LossOptions = network.LossOptions,
                ShouldNormalize = network.ShouldNormalize,
                OutputApplySoftMax = network.OutputApplySoftMax,
                MinMaxValues = network.MinMaxValues?.Select(x => new[] { (float)x.Item1, (float)x.Item2 }).ToList()
            };

            List<MatrixLayer> layers = network.Layers;

            // neuron groups: the inputs, every hidden layer and the outputs, each followed by the bias neuron that feeds the next layer
            var starts = new int[layers.Count + 1];
            List<int> Group(int count, NodeType type, ActivationTypes activation, bool? applyActivation, bool bias)
            {
                var indexes = new List<int>();
                for (int i = 0; i < count; i++)
                {
                    indexes.Add(document.Neurons.Count);
                    document.Neurons.Add(new NeuronDocument { Id = (uint)document.Neurons.Count, Type = type, Activation = activation, ApplyActivation = applyActivation });
                }

                if (bias)
                {
                    indexes.Add(document.Neurons.Count);
                    document.Neurons.Add(new NeuronDocument { Id = (uint)document.Neurons.Count, Type = NodeType.BIAS, Activation = ActivationTypes.Identity });
                }

                return indexes;
            }

            document.InputLayer = Group(network.InputCount, NodeType.INPUT, network.InputActivation, null, network.InputBias);
            for (int l = 0; l < layers.Count - 1; l++)
            {
                starts[l + 1] = document.Neurons.Count;
                document.HiddenLayers.Add(Group(layers[l].Outputs, NodeType.HIDDEN, layers[l].Activation, null, layers[l + 1].Bias != null));
            }

            MatrixLayer last = layers[^1];
            starts[layers.Count] = document.Neurons.Count;
            document.OutputLayer = Group(last.Outputs, NodeType.OUTPUT, last.Activation, last.Activate, false);

            // synapses in the order of the output synapses of the neurons: row by row of every weight matrix, the bias row last
            for (int l = 0; l < layers.Count; l++)
            {
                MatrixLayer layer = layers[l];
                int first = document.Synapses.Count;
                int rows = layer.Inputs + (layer.Bias != null ? 1 : 0);

                for (int i = 0; i < rows; i++)
                {
                    bool bias = i == layer.Inputs;
                    var optimizer = bias ? layer.BiasOptimizer : layer.WeightOptimizer;
                    for (int j = 0; j < layer.Outputs; j++)
                    {
                        document.Synapses.Add(new SynapseDocument
                        {
                            From = starts[l] + i,
                            To = starts[l + 1] + j,
                            Weight = bias ? layer.Bias[j] : layer.Weights[(i * layer.Outputs) + j],
                            Optimizer = optimizer.Type,
                            LearningRate = optimizer.LearningRate
                        });
                    }
                }

                for (int j = 0; j < layer.Outputs; j++)
                {
                    document.Neurons[starts[l + 1] + j].Inputs = Enumerable.Range(0, rows).Select(i => first + (i * layer.Outputs) + j).ToList();
                }
            }

            return document;
        }
    }
}
