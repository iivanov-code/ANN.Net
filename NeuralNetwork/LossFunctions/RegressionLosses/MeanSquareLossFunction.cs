using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.RegressionLosses
{
    /// <summary>
    /// Mean Squared Error (MSE) Loss Function
    /// </summary>
    internal class MeanSquareLossFunction : BaseLossFunction<MeanSquareLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            return (1 / result.Count) * (Quad)Enumerable.Range(0, result.Count - 1).Select(i => result[i].Target - result[i].Predicted).Sum(x => Math.Pow(x, 2));
        }
    }
}
