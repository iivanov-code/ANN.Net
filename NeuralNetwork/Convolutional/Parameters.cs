using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Transformers;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Trainable values of a layer, each with its own optimizer
    /// </summary>
    [Serializable]
    internal sealed class Parameters
    {
        private readonly IOptimizer[] optimizers;

        public Parameters(int count, OptimizerTypes optimizerType, ICollection<LearningRateOptimizer> optimizers, Func<Quad> initialValue)
        {
            Values = new Quad[count];
            this.optimizers = new IOptimizer[count];
            OptimizerType = optimizerType;

            for (int i = 0; i < count; i++)
            {
                Values[i] = initialValue();
                this.optimizers[i] = TransformerUtils.CreateOptimizer(optimizerType, optimizers);
            }
        }

        public Quad[] Values { get; }

        public OptimizerTypes OptimizerType { get; }

        public IReadOnlyList<IOptimizer> Optimizers => optimizers;

        /// <summary>
        /// Learning rates of the optimizers that have one
        /// </summary>
        public IEnumerable<float> LearningRates => optimizers.OfType<LearningRateOptimizer>().Select(x => (float)x.LearningRate);

        /// <summary>
        /// Applies one optimizer step to every value
        /// </summary>
        public void Update(Quad[] gradients)
        {
            for (int i = 0; i < Values.Length; i++)
            {
                Values[i] = optimizers[i].Calculate(gradients[i], Values[i]);
            }
        }
    }
}
