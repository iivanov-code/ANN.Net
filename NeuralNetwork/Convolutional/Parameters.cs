using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;

namespace ANN.Net.Convolutional
{
    /// <summary>
    /// Trainable values of a layer with one optimizer for all of them, whose state is kept in arrays
    /// </summary>
    [Serializable]
    internal sealed class Parameters
    {
        public Parameters(int count, OptimizerTypes optimizerType, ICollection<LearningRateOptimizer> optimizers, Func<Quad> initialValue)
        {
            Values = new Quad[count];
            for (int i = 0; i < count; i++)
            {
                Values[i] = initialValue();
            }

            Optimizer = ArrayOptimizer.Create(optimizerType, count, optimizers);
        }

        public Quad[] Values { get; }

        public OptimizerTypes OptimizerType => Optimizer.Type;

        public ArrayOptimizer Optimizer { get; }

        public float LearningRate
        {
            get => Optimizer.LearningRate;
            set => Optimizer.LearningRate = value;
        }

        /// <summary>
        /// Applies one optimizer step to every value
        /// </summary>
        public void Update(Quad[] gradients)
        {
            Optimizer.Update(Values, gradients);
        }
    }
}
