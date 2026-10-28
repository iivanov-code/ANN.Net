using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions
{
    internal class HingeLossFunction : BaseLossFunction<HingeLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            Quad sum = 0;
            foreach (var item in result)
            {
                sum += Math.Max(0, 1 - item.Target * item.Predicted);
            }

            return sum;
        }
    }
}
