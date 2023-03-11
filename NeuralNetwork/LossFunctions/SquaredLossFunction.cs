using System;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.LossFunctions
{
    internal class SquaredLossFunction : BaseLossFunction<SquaredLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(Quad target, Quad predicted)
        {
            return (Quad)Math.Pow(target - predicted, 2);
        }
    }
}
