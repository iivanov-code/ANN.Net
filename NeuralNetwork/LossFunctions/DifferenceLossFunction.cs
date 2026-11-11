using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Not a loss: reports the sum of predicted - target, so whether the outputs are too high or too low, and trains like
    /// <see cref="SquaredLossFunction"/>
    /// </summary>
    internal sealed class DifferenceLossFunction : ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            ArgumentNullException.ThrowIfNull(result);
            double sum = 0;
            foreach (ErrorResult item in result) sum += (double)item.Predicted - item.Target;
            return (Quad)sum;
        }

        public Quad[] CalculateGradient(IList<ErrorResult> result)
        {
            ArgumentNullException.ThrowIfNull(result);
            var gradient = new Quad[result.Count];
            for (int i = 0; i < gradient.Length; i++) gradient[i] = result[i].Predicted - result[i].Target;
            return gradient;
        }
    }
}
