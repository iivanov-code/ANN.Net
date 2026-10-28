using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions
{
    internal class SquaredLossFunction : BaseLossFunction<SquaredLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            Quad sum = 0;
            foreach (var item in result)
            {
                sum += (Quad)Math.Pow(item.Target - item.Predicted, 2);
            }

            return sum;
        }
    }
}
