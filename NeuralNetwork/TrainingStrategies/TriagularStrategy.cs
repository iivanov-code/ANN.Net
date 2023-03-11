using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.TrainingStrategies
{
    internal sealed class TriagularStrategy : TrainingStrategy, ITrainingStrategy
    {
        private Rate stepSize;

        public TriagularStrategy(INetwork network,
             float minValue = 0f,
             float maxValue = 1f,
             bool shouldMinimize = false,
             float minimizingStep = 0.5f)
             : base(network, minValue, maxValue, shouldMinimize, minimizingStep)
        { }

        public void Train(IReadOnlyCollection<Quad[]> data, ushort totalEpochs, ushort epochStep = 2)
        {
            if (epochStep < 2) throw new ArgumentException("Must be greater than 1", nameof(epochStep));

            MinimizeLearningRate();

            Rate currentRate = MaxValue;
            UpdateRates(currentRate);

            stepSize = (MaxValue - MinValue) / (totalEpochs / epochStep);

            HalfLoop(data, currentRate, Decrease, totalEpochs);
            HalfLoop(data, currentRate, Increase, totalEpochs);
        }

        private void HalfLoop(IReadOnlyCollection<Quad[]> data, Rate currentRate, Func<Rate, Rate> currentFn, ushort totalEpochs, ushort epochStep = 2)
        {
            byte currentEpoch = 0;
            for (int i = 0; i < totalEpochs; i++)
            {
                RunEpoch(data);

                currentEpoch++;
                if (currentEpoch == epochStep)
                {
                    currentEpoch = 0;
                    currentRate = currentFn(currentRate);
                    UpdateRates(currentRate);
                }
            }
        }

        private Rate Decrease(Rate inRate)
        {
            return inRate > stepSize ? inRate - stepSize : inRate;
        }

        private Rate Increase(Rate inRate)
        {
            return inRate < Rate.MaxValue ? inRate + stepSize : inRate; ;
        }
    }
}
