using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.ClassificationLosses
{
    internal class KullbackLeiblerDivergenceLossFunction : BaseLossFunction<KullbackLeiblerDivergenceLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            //Kullback-Leibler divergence loss function
            return (Quad)Enumerable.Range(0, result.Count - 1)
                                  .Select(i => result[i].Target * Math.Log(result[i].Target / result[i].Predicted))
                                  .Sum(x => x);
        }
    }
}
