using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.ClassificationLosses
{
    internal class CrossEntropyLossFunction : BaseLossFunction<CrossEntropyLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            //Cross entropy loss function
            return (-1 / result.Count) * (Quad)Enumerable.Range(0, result.Count - 1)
                                                 .Select(i => result[i].Target * Math.Log(result[i].Predicted) + (1 - result[i].Target) * Math.Log(1 - result[i].Predicted))
                                                 .Sum(x => x);
        }
    }
}
