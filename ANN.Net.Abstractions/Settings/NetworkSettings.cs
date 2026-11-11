using System;
using System.Collections.Generic;
using System.ComponentModel;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.Abstractions.Settings
{
    public class NetworkSettings
    {
        /// <summary>
        /// Loss the network trains on and <see cref="INetwork.Backpropagate"/> returns
        /// </summary>
        [DefaultValue(LossFunctionTypes.Squared)]
        public LossFunctionTypes LossType { get; set; } = LossFunctionTypes.Squared;

        /// <summary>
        /// Parameters of the loss, such as δ of the Huber loss; null for the defaults
        /// </summary>
        public LossOptions LossOptions { get; set; }

        public bool ShouldNormalize { get; set; }
        public List<Tuple<Quad, Quad>> MinMaxValues { get; set; }
        public InputLayerSettings InputSettings { get; set; }
        public OutputLayerSettings OutputSettings { get; set; }

        public bool OutputApplySoftMax { get; set; }
        public IUniqueIdentityGenerator IdentityGenerator { get; set; }

        /// <summary>
        /// Seed of the initial weights; the same seed builds the same network. Null draws them from a shared random source
        /// </summary>
        public int? Seed { get; set; }
    }
}
