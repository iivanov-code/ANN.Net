using System;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.Remote
{
    /// <summary>
    /// A run of hidden layers cut in breadth: every part holds a contiguous slice of each layer. The section
    /// takes the full activation vector of the layer before it and produces the full activation vector of
    /// its last layer, so it fits between a <see cref="SplitNetwork"/> head and a tail served by
    /// <see cref="RemoteNetworkPart"/>
    /// </summary>
    public sealed class NetworkPartition
    {
        internal NetworkPartition(IUniqueIdentityGenerator identityGenerator,
                                  RecurrentLayerSettings[] layers,
                                  int firstLayer,
                                  int inputWidth,
                                  bool inputHasBias,
                                  int parts)
        {
            this.IdentityGenerator = identityGenerator;
            this.Layers = layers;
            this.FirstLayer = firstLayer;
            this.InputWidth = inputWidth;
            this.InputHasBias = inputHasBias;
            this.Parts = parts;
        }

        public IUniqueIdentityGenerator IdentityGenerator { get; }

        /// <summary>
        /// The partitioned layers, one entry per actual layer
        /// </summary>
        public RecurrentLayerSettings[] Layers { get; }

        /// <summary>
        /// Index of the first partitioned layer in the whole description's hidden layers
        /// </summary>
        public int FirstLayer { get; }

        public int LastLayer => this.FirstLayer + this.Layers.Length - 1;

        public int LayerCount => this.Layers.Length;

        /// <summary>
        /// Neurons of the layer feeding the section, bias not counted
        /// </summary>
        public int InputWidth { get; }

        /// <summary>
        /// Whether the layer feeding the section has a bias neuron; every part then gets its own
        /// </summary>
        public bool InputHasBias { get; }

        public int Parts { get; }

        public int OutputWidth => this.Width(this.Layers.Length - 1);

        public int Width(int layer)
        {
            return this.Layers[layer].NeuronsCount;
        }

        /// <summary>
        /// First neuron of <paramref name="part"/>'s slice of <paramref name="layer"/>
        /// </summary>
        public int SliceStart(int layer, int part)
        {
            if (part < 0 || part > this.Parts) throw new ArgumentOutOfRangeException(nameof(part));

            return (int)((long)this.Width(layer) * part / this.Parts);
        }

        public int SliceCount(int layer, int part)
        {
            if (part < 0 || part >= this.Parts) throw new ArgumentOutOfRangeException(nameof(part));

            return this.SliceStart(layer, part + 1) - this.SliceStart(layer, part);
        }

        /// <summary>
        /// Part that owns neuron <paramref name="index"/> of <paramref name="layer"/>
        /// </summary>
        public int PartOf(int layer, int index)
        {
            for (int part = 0; part < this.Parts; part++)
            {
                if (index < this.SliceStart(layer, part) + this.SliceCount(layer, part)) return part;
            }

            throw new ArgumentOutOfRangeException(nameof(index));
        }
    }
}
