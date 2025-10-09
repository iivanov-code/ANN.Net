using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.RegressionLosses
{
    internal class MeanSquaredLogarithmicLossFunction : BaseLossFunction<MeanSquaredLogarithmicLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            
            return (1 / result.Count) * (Quad)Enumerable.Range(0, result.Count - 1)
                                                        .Select(i => (Quad)Math.Pow(Math.Log(result[i].Target + 1) - Math.Log(result[i].Predicted + 1), 2))
                                                        .Sum(x => x);
        }
    }
}
