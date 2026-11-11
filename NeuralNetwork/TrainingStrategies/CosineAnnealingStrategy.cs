using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.TrainingStrategies
{
    /// <summary>
    /// Stochastic gradient descent with restarts
    /// </summary>
    internal sealed class CosineAnnealingStrategy : TrainingStrategy, ITrainingStrategy
    {
        public CosineAnnealingStrategy(INetwork network,
           float minValue = 0f,
           float maxValue = 1f,
           bool shouldMinimize = false,
           float minimizingStep = 0.5f)
             : base(network, minValue, maxValue, shouldMinimize, minimizingStep)
        { }

        public void Train(IReadOnlyCollection<Quad[]> data, ushort totalEpochs, ushort epochStep = 1)
        {
            if (epochStep < 1) throw new ArgumentException("Must be greater than 0", nameof(epochStep));

            MinimizeLearningRate();

            ushort currentEpoch = 0;
            Rate currentRate = MaxValue;
            UpdateRates(currentRate);

            for (int i = 0; i < totalEpochs; i++)
            {
                RunEpoch(data);

                currentEpoch++;
                if (currentEpoch % epochStep == 0)
                {
                    currentRate = CalculateNewRate(currentEpoch, totalEpochs);
                    UpdateRates(currentRate);
                }
            }
        }

        private Rate CalculateNewRate(ushort currentEpoch, ushort totalEpochs)
        {
            return MinValue + 0.5f * (MaxValue - MinValue) * (1 + (float)Math.Cos(currentEpoch / totalEpochs * Math.PI));
        }
    }
}
