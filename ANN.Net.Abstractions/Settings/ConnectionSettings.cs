using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;

namespace ANN.Net.Abstractions.Settings
{
    public class ConnectionSettings
    {
        public OptimizerTypes OptimizerType { get; set; }
        public WeightInitType WeightInit { get; set; }
        public int NeuronsCount { get; set; }
        public ICollection<LearningRateOptimizer> Optimizers { get; set; }
    }
}
