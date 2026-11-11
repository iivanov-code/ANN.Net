using System.Collections.Generic;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.Abstractions.Interfaces
{
    /// <summary>
    /// Loss of one sample, given the target and the predicted value of every output. Networks train with
    /// <see cref="CalculateGradient"/>, so the loss decides what is learned
    /// </summary>
    public interface ILossFunction
    {
        Quad CalculateLoss(IList<ErrorResult> result);

        /// <summary>
        /// Gradient of <see cref="CalculateLoss"/> with respect to every predicted value, in the order of
        /// <paramref name="result"/>
        /// </summary>
        Quad[] CalculateGradient(IList<ErrorResult> result);
    }
}
