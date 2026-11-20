using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Gaussian negative log likelihood: the first half of the outputs are the means, the second half the variances of
    /// the targets in the first half of the targets. Variances below ε count as ε; as in PyTorch the gradient still
    /// reaches them, computed at ε, so a variance that went negative is pushed back up by any error larger than √ε
    /// </summary>
    internal sealed class GaussianNegativeLogLikelihoodLossFunction : ILossFunction
    {
        private const double Epsilon = 1e-6;

        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            int n = Count(result);
            if (n == 0) return 0;

            double sum = 0;
            for (int k = 0; k < n; k++)
            {
                double variance = Math.Max(result[n + k].Predicted, Epsilon);
                double error = result[k].Target - result[k].Predicted;
                sum += Math.Log(variance) + (error * error / variance);
            }

            return (Quad)(sum / (2 * n));
        }

        public Quad[] CalculateGradient(IList<ErrorResult> result)
        {
            int n = Count(result);
            var gradient = new Quad[result.Count];
            for (int k = 0; k < n; k++)
            {
                double variance = Math.Max(result[n + k].Predicted, Epsilon);
                double error = result[k].Target - result[k].Predicted;
                gradient[k] = (Quad)(-error / (variance * n));
                gradient[n + k] = (Quad)(((1 / variance) - (error * error / (variance * variance))) / (2 * n));
            }

            return gradient;
        }

        private static int Count(IList<ErrorResult> result)
        {
            ArgumentNullException.ThrowIfNull(result);
            if (result.Count % 2 != 0)
                throw new ArgumentException("The Gaussian negative log likelihood needs an even number of outputs: the means, then the variances", nameof(result));
            return result.Count / 2;
        }
    }
}
