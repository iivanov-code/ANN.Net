using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.TrainingStrategies
{
    internal abstract class TrainingStrategy
    {
        protected TrainingStrategy(INetwork network, Rate minValue, Rate maxValue, bool shouldMinimize, float minimizingStep)
        {
            Network = network;
            MinValue = minValue;
            MaxValue = maxValue;
            ShouldMinimize = shouldMinimize;
            MinimizingStep = minimizingStep;

            this.Network = network;
            this.MinValue = minValue;

            if (shouldMinimize)
            {
                this.MaxValue = maxValue + minimizingStep;
            }
            else
            {
                this.MaxValue = maxValue;
            }

            this.MinimizingStep = minimizingStep;
            this.ShouldMinimize = shouldMinimize;
        }

        public INetwork Network { get; protected set; }
        public Rate MinValue { get; protected set; }
        public Rate MaxValue { get; protected set; }
        public bool ShouldMinimize { get; protected set; }
        public float MinimizingStep { get; protected set; }

        protected void RunEpoch(IReadOnlyCollection<Quad[]> data)
        {
            var dataEnumerator = data.GetEnumerator();
            Quad[] nextData, currentData;

            while (dataEnumerator.MoveNext())
            {
                currentData = dataEnumerator.Current;
            beginLoop:
                Network.Propagate(currentData);

                if (dataEnumerator.MoveNext())
                {
                    nextData = dataEnumerator.Current;
                    Network.Backpropagate(nextData);
                    currentData = nextData;
                    goto beginLoop;
                }
            }
        }

        protected void MinimizeLearningRate()
        {
            if (ShouldMinimize)
            {
                this.MaxValue = this.MaxValue - this.MinimizingStep;
            }
        }

        protected void UpdateRates(Rate rate)
        {
            foreach (var optimizer in ((Network)this.Network).Optimizers)
            {
                optimizer.LearningRate = rate;
            }
        }
    }
}
