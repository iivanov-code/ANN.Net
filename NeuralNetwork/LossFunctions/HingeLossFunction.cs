using System;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.LossFunctions
{
    internal class HingeLossFunction : BaseLossFunction<HingeLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(Quad target, Quad predicted)
        {
            return Math.Max(0, 1 - target * predicted);
        }
    }
}
