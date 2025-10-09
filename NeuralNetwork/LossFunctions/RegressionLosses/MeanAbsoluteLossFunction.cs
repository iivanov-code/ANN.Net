using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions.RegressionLosses
{
    /// <summary>
    /// Represents a loss function that calculates the mean absolute error (MAE) between predicted and target values.
    /// </summary>
    /// <remarks>The mean absolute error is computed as the average of the absolute differences between the predicted
    /// and target values. This loss function is commonly used in regression problems to measure the accuracy of
    /// predictions.</remarks>
    internal class MeanAbsoluteLossFunction : BaseLossFunction<MeanAbsoluteLossFunction>, ILossFunction
    {
        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            return (1 / result.Count) * (Quad)Enumerable.Range(0, result.Count - 1).Select(i => result[i].Target - result[i].Predicted).Sum(x => Math.Abs(x));
        }
    }
}
