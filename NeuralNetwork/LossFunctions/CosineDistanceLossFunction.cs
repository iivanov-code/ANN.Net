using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// 1 - cos(predicted, target): only the direction of the output vector counts, not its length. A zero vector has no
    /// direction; it gets the loss 1 and no gradient
    /// </summary>
    internal sealed class CosineDistanceLossFunction : ILossFunction
    {
        private const double Epsilon = 1e-12;

        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            (double dot, double predictedNorm, double targetNorm) = Measure(result);
            if (predictedNorm < Epsilon || targetNorm < Epsilon) return 1;
            return (Quad)(1 - (dot / (predictedNorm * targetNorm)));
        }

        public Quad[] CalculateGradient(IList<ErrorResult> result)
        {
            (double dot, double predictedNorm, double targetNorm) = Measure(result);
            var gradient = new Quad[result.Count];
            if (predictedNorm < Epsilon || targetNorm < Epsilon) return gradient;

            // d cos / dp = t / (|p| |t|) - cos · p / |p|²
            double cosine = dot / (predictedNorm * targetNorm);
            for (int i = 0; i < gradient.Length; i++)
            {
                double p = result[i].Predicted, t = result[i].Target;
                gradient[i] = (Quad)(-((t / (predictedNorm * targetNorm)) - (cosine * p / (predictedNorm * predictedNorm))));
            }

            return gradient;
        }

        private static (double Dot, double PredictedNorm, double TargetNorm) Measure(IList<ErrorResult> result)
        {
            ArgumentNullException.ThrowIfNull(result);
            double dot = 0, predicted = 0, target = 0;
            foreach (ErrorResult item in result)
            {
                double p = item.Predicted, t = item.Target;
                dot += p * t;
                predicted += p * p;
                target += t * t;
            }

            return (dot, Math.Sqrt(predicted), Math.Sqrt(target));
        }
    }
}
