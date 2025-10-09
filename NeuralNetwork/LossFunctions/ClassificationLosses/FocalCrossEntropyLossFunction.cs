using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.ClassificationLosses
{
    internal class FocalCrossEntropyLossFunction : BaseLossFunction<FocalCrossEntropyLossFunction>, ILossFunction
    {
        private Quad alpha;
        public FocalCrossEntropyLossFunction(Quad alpha)
        {
            this.alpha = alpha;
        }

        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            return (Quad)Enumerable.Range(0, result.Count - 1)
                                  .Select(i => result[i].Target * Math.Pow(1 - result[i].Predicted, alpha) * Math.Log(result[i].Predicted) + (1 - result[i].Target) * Math.Pow(result[i].Predicted, 2) * Math.Log(1 - result[i].Predicted))
                                  .Sum(x => x);
        }
    }
}
