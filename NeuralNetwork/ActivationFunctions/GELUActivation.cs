using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Gaussian Error Linear Unit using the tanh approximation:
    /// GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / PI) * (x + alpha * x^3))), alpha is usually 0.044715
    /// </summary>
    [ActivationType(Type = ActivationTypes.GELU)]
    internal class GELUActivation : BaseActivation<GELUActivation>, IActivationFunction
    {
        public Quad MinValue => 0;

        public Quad MaxValue => float.MaxValue;

        private Quad alpha;

        public GELUActivation(Quad alpha)
        {
            this.alpha = alpha;
        }

        public Quad Activation(ref Quad x)
        {
            double value = x;
            return (Quad)(0.5 * value * (1 + Math.Tanh(Inner(value))));
        }

        public Quad Derivative(Quad x)
        {
            double value = x;
            double tanh = Math.Tanh(Inner(value));
            double innerDerivative = Math.Sqrt(2 / Math.PI) * (1 + 3 * (double)alpha * value * value);

            return (Quad)(0.5 * (1 + tanh) + 0.5 * value * (1 - tanh * tanh) * innerDerivative);
        }

        private double Inner(double x)
        {
            return Math.Sqrt(2 / Math.PI) * (x + (double)alpha * Math.Pow(x, 3));
        }
    }
}
