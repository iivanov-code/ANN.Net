using System;
using System.Collections.Generic;
using System.ComponentModel;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    public class NetworkSettings
    {
        [DefaultValue(LossFunctionTypes.Difference)]
        public LossFunctionTypes LossType { get; set; }

        public bool ShouldNormalize { get; set; }
        public List<Tuple<Quad, Quad>> MinMaxValues { get; set; }
        public InputLayerSettings InputSettings { get; set; }
        public OutputLayerSettings OutputSettings { get; set; }

        public bool OutputApplySoftMax { get; set; }
    }
}
