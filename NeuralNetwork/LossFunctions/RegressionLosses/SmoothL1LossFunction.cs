using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.RegressionLosses
{
    internal class SmoothL1LossFunction : BaseLossFunction<SmoothL1LossFunction>, ILossFunction
    {
        private Quad alpha;

        public SmoothL1LossFunction(float alpha)
        {
            this.alpha = alpha;
        }


        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            Quad sumLosses = (Quad)Enumerable.Range(0, result.Count - 1)
                                 .Select(i => result[i].Target - result[i].Predicted)
                                 .Select(x =>
                                 {
                                     if (Math.Abs(x) < this.alpha)
                                     {
                                         return (Quad)Math.Pow(x, 2) / (2 * alpha);
                                     }
                                     else
                                     {
                                         return this.alpha * (Math.Abs(x) - (alpha / 2));
                                     }
                                 })
                                 .Sum(x => x);

            return (1 / result.Count) * sumLosses;
        }
    }
}
