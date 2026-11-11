using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Models;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Loss that adds up, or averages, one term per output that depends only on that output's target and prediction
    /// </summary>
    internal abstract class PointwiseLossFunction : ILossFunction
    {
        protected const double Epsilon = 1e-7;

        private readonly bool mean;

        /// <param name="mean">Whether the terms are averaged over the outputs rather than added up</param>
        protected PointwiseLossFunction(bool mean)
        {
            this.mean = mean;
        }

        public Quad CalculateLoss(IList<ErrorResult> result)
        {
            ArgumentNullException.ThrowIfNull(result);
            double sum = 0;
            foreach (ErrorResult item in result) sum += Value(item.Target, item.Predicted);
            return (Quad)(mean && result.Count > 0 ? sum / result.Count : sum);
        }

        public Quad[] CalculateGradient(IList<ErrorResult> result)
        {
            ArgumentNullException.ThrowIfNull(result);
            double scale = mean && result.Count > 0 ? 1.0 / result.Count : 1;
            var gradient = new Quad[result.Count];
            for (int i = 0; i < gradient.Length; i++) gradient[i] = (Quad)(Derivative(result[i].Target, result[i].Predicted) * scale);
            return gradient;
        }

        /// <summary>
        /// Term of one output
        /// </summary>
        protected abstract double Value(double target, double predicted);

        /// <summary>
        /// Derivative of <see cref="Value"/> with respect to the prediction
        /// </summary>
        protected abstract double Derivative(double target, double predicted);
    }
}
