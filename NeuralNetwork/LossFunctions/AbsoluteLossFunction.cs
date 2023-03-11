using System;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.LossFunctions
{
    internal class AbsoluteLossFunction : BaseLossFunction<AbsoluteLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(Quad target, Quad predicted)
        {
            return Math.Abs(target - predicted);
        }
    }
}
