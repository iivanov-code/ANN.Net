using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Interfaces.Neurons;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Abstractions.Settings;
using ANN.Net.Connections;
using ANN.Net.Neurons;

namespace ANN.Net.Remote
{
    /// <summary>
    /// One part's slice of a breadth-split section, as a neuron graph. Layer by layer it holds its own
    /// neurons, a ghost for every neuron the other parts own in the layer before, and its own bias neuron.
    /// Its own neurons in a layer have one <see cref="RemoteSynapse"/> per other part, which hands the
    /// layer's slice to <see cref="ActivationsReady"/>; the slice of the last layer goes to <see cref="OutputsReady"/>.
    /// Gradients travel the opposite way: ghosts report through <see cref="GradientsReady"/>, the section inputs
    /// through <see cref="InputGradientsReady"/>. The graph is free of threads; the callbacks fire on whichever
    /// thread delivered the last piece
    /// </summary>
    public sealed class PartNetwork
    {
        private readonly NetworkPartition partition;
        private readonly int part;
        private readonly GhostNeuron[] inputs;
        private readonly List<BiasNeuron> biases;
        private readonly List<Dictionary<int, GhostNeuron[]>> ghosts;
        private readonly List<Dictionary<int, CallbackSynapseConnection>> exchanges;
        private readonly List<INeuron[]> slices;
        private readonly CallbackSynapseConnection outputCollector;
        private readonly List<PartSynapse> synapses;
        private readonly int[][] ghostCompleted;
        private int inputsCompleted;

        public PartNetwork(NetworkPartition partition, int part)
        {
            this.partition = partition ?? throw new ArgumentNullException(nameof(partition));
            if (part < 0 || part >= partition.Parts) throw new ArgumentOutOfRangeException(nameof(part));

            this.part = part;
            this.Optimizers = new List<LearningRateOptimizer>();
            this.biases = new List<BiasNeuron>();
            this.ghosts = new List<Dictionary<int, GhostNeuron[]>>();
            this.exchanges = new List<Dictionary<int, CallbackSynapseConnection>>();
            this.slices = new List<INeuron[]>();
            this.synapses = new List<PartSynapse>();
            this.ghostCompleted = new int[partition.LayerCount][];

            NetworkBuilder builder = NetworkBuilder.Create();
            IUniqueIdentityGenerator ids = partition.IdentityGenerator;

            this.inputs = new GhostNeuron[partition.InputWidth];
            for (int i = 0; i < this.inputs.Length; i++)
            {
                this.inputs[i] = new GhostNeuron(i, this.InputGradientCompleted);
            }

            List<(INeuron Neuron, int Index)> previous = this.inputs.Select(n => ((INeuron)n, n.Index)).ToList();
            if (partition.InputHasBias)
            {
                BiasNeuron bias = new BiasNeuron(ActivationTypes.Identity);
                this.biases.Add(bias);
                previous.Add((bias, partition.InputWidth));
            }

            int last = partition.LayerCount - 1;
            for (int layer = 0; layer <= last; layer++)
            {
                RecurrentLayerSettings settings = partition.Layers[layer];
                int start = partition.SliceStart(layer, part);
                int count = partition.SliceCount(layer, part);

                INeuron[] slice = new INeuron[count];
                for (int j = 0; j < count; j++)
                {
                    slice[j] = layer == last
                        ? new OutputNeuron(settings.ActivationType, true)
                        : (INeuron)new HiddenNeuron(settings.ActivationType);

                    foreach ((INeuron Neuron, int Index) source in previous)
                    {
                        builder.ConnectAxon(source.Neuron, slice[j], ids, settings, this.Optimizers);
                        this.synapses.Add(new PartSynapse(layer, source.Index, start + j, (BaseSynapse)source.Neuron.Outputs.Last()));
                    }
                }

                this.slices.Add(slice);
                this.ghostCompleted[layer] = new int[partition.Parts];

                if (layer == last)
                {
                    this.outputCollector = new CallbackSynapseConnection(values => this.OutputsReady?.Invoke(values));
                    foreach (INeuron neuron in slice)
                    {
                        neuron.Outputs.Add(new RemoteSynapse(neuron, this.outputCollector, ids));
                    }

                    this.ghosts.Add(new Dictionary<int, GhostNeuron[]>());
                    this.exchanges.Add(new Dictionary<int, CallbackSynapseConnection>());
                    break;
                }

                Dictionary<int, GhostNeuron[]> layerGhosts = new Dictionary<int, GhostNeuron[]>();
                Dictionary<int, CallbackSynapseConnection> layerExchanges = new Dictionary<int, CallbackSynapseConnection>();
                previous = slice.Select((n, j) => (n, start + j)).ToList();

                for (int other = 0; other < partition.Parts; other++)
                {
                    if (other == part) continue;

                    int otherStart = partition.SliceStart(layer, other);
                    int otherCount = partition.SliceCount(layer, other);
                    int capturedLayer = layer;
                    int capturedOther = other;

                    GhostNeuron[] group = new GhostNeuron[otherCount];
                    for (int g = 0; g < otherCount; g++)
                    {
                        group[g] = new GhostNeuron(otherStart + g, ghost => this.GhostGradientCompleted(capturedLayer, capturedOther));
                        previous.Add((group[g], group[g].Index));
                    }

                    layerGhosts[other] = group;

                    CallbackSynapseConnection exchange = new CallbackSynapseConnection(values => this.ActivationsReady?.Invoke(capturedOther, capturedLayer, values));
                    foreach (INeuron neuron in slice)
                    {
                        neuron.Outputs.Add(new RemoteSynapse(neuron, exchange, ids));
                    }

                    layerExchanges[other] = exchange;
                }

                if (settings.HasBiasNeuron)
                {
                    BiasNeuron bias = new BiasNeuron(settings.ActivationType);
                    this.biases.Add(bias);
                    previous.Add((bias, partition.Width(layer)));
                }

                this.ghosts.Add(layerGhosts);
                this.exchanges.Add(layerExchanges);
            }
        }

        /// <summary>
        /// (to part, layer, values): this part's slice of a layer is complete and the other part needs it
        /// </summary>
        public Action<int, int, Quad[]> ActivationsReady { get; set; }

        /// <summary>
        /// This part's slice of the last layer is complete
        /// </summary>
        public Action<Quad[]> OutputsReady { get; set; }

        /// <summary>
        /// (to part, layer, values): the gradient with respect to the other part's slice of a layer, summed over this part's neurons
        /// </summary>
        public Action<int, int, Quad[]> GradientsReady { get; set; }

        /// <summary>
        /// The gradient with respect to the section inputs, summed over this part's neurons of the first layer
        /// </summary>
        public Action<Quad[]> InputGradientsReady { get; set; }

        public NetworkPartition Partition => this.partition;

        public int Part => this.part;

        public ICollection<LearningRateOptimizer> Optimizers { get; }

        /// <summary>
        /// Every weighted synapse with the global position of its ends, in creation order
        /// </summary>
        public IReadOnlyList<PartSynapse> Synapses => this.synapses;

        /// <summary>
        /// This part's neurons of a layer, in slice order
        /// </summary>
        public IReadOnlyList<INeuron> Slice(int layer)
        {
            return this.slices[layer];
        }

        public void DeliverInputs(Quad[] values)
        {
            if (values == null) throw new ArgumentNullException(nameof(values));
            if (values.Length != this.inputs.Length)
                throw new ArgumentException($"Expected {this.inputs.Length} input values but received {values.Length}", nameof(values));

            for (int i = 0; i < values.Length; i++)
            {
                this.inputs[i].Propagate(new NeuronPropagateEventArgs(values[i]));
            }

            // bias neurons have no inputs, so the part fires them itself on every propagation
            foreach (BiasNeuron bias in this.biases)
            {
                bias.Propagate(new NeuronPropagateEventArgs(1));
            }
        }

        /// <summary>
        /// Another part's slice of <paramref name="layer"/> arrived; its ghosts fire into this part's next layer
        /// </summary>
        public void DeliverActivations(int fromPart, int layer, Quad[] values)
        {
            GhostNeuron[] group = this.Ghosts(fromPart, layer);
            if (values == null || values.Length != group.Length)
                throw new ArgumentException($"Expected {group.Length} activations from part {fromPart} for layer {layer} but received {values?.Length ?? 0}", nameof(values));

            for (int i = 0; i < values.Length; i++)
            {
                group[i].Propagate(new NeuronPropagateEventArgs(values[i]));
            }
        }

        /// <summary>
        /// Gradient with respect to this part's slice of the last layer
        /// </summary>
        public void DeliverOutputGradients(Quad[] slice)
        {
            this.outputCollector.DeliverGradients(slice);
        }

        /// <summary>
        /// Another part's partial gradient with respect to this part's slice of <paramref name="layer"/>
        /// </summary>
        public void DeliverGradients(int fromPart, int layer, Quad[] values)
        {
            if (layer < 0 || layer >= this.exchanges.Count || !this.exchanges[layer].TryGetValue(fromPart, out CallbackSynapseConnection exchange))
                throw new ArgumentException($"Part {fromPart} does not consume layer {layer} of part {this.part}", nameof(fromPart));

            exchange.DeliverGradients(values);
        }

        public void SetLearningRate(float learningRate)
        {
            RemoteNetworkPart.SetRate(this.Optimizers, learningRate);
        }

        private GhostNeuron[] Ghosts(int fromPart, int layer)
        {
            if (layer < 0 || layer >= this.ghosts.Count || !this.ghosts[layer].TryGetValue(fromPart, out GhostNeuron[] group))
                throw new ArgumentException($"Part {this.part} holds no ghosts of part {fromPart} for layer {layer}", nameof(fromPart));

            return group;
        }

        private void GhostGradientCompleted(int layer, int fromPart)
        {
            GhostNeuron[] group = this.ghosts[layer][fromPart];
            if (Interlocked.Increment(ref this.ghostCompleted[layer][fromPart]) < group.Length) return;

            Interlocked.Exchange(ref this.ghostCompleted[layer][fromPart], 0);

            Quad[] gradients = new Quad[group.Length];
            for (int i = 0; i < group.Length; i++)
            {
                gradients[i] = group[i].Gradient;
            }

            this.GradientsReady?.Invoke(fromPart, layer, gradients);
        }

        private void InputGradientCompleted(GhostNeuron input)
        {
            if (Interlocked.Increment(ref this.inputsCompleted) < this.inputs.Length) return;

            Interlocked.Exchange(ref this.inputsCompleted, 0);

            Quad[] gradients = new Quad[this.inputs.Length];
            for (int i = 0; i < gradients.Length; i++)
            {
                gradients[i] = this.inputs[i].Gradient;
            }

            this.InputGradientsReady?.Invoke(gradients);
        }
    }
}
