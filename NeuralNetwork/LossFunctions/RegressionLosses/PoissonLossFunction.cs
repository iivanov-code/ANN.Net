using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.RegressionLosses
{
    internal class PoissonLossFunction : BaseLossFunction<PoissonLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            return (1 / result.Count) * (Quad)Enumerable.Range(0, result.Count - 1).Select(i => (result[i].Target - result[i].Predicted) * Math.Log(result[i].Target)).Sum(x => x);
        }
    }
}
