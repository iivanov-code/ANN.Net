using System;
using ANN.Net.Abstractions.Attributes;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// Cardinal cubic B-spline centred at 0: (4 - 6x² + 3|x|³) / 6 for |x| &lt; 1, (2 - |x|)³ / 6 for 1 ≤ |x| &lt; 2,
    /// 0 elsewhere. Twice continuously differentiable; shifted copies one apart add up to 1
    /// </summary>
    [Serializable]
    [ActivationType(Type = ActivationTypes.CubicBSpline)]
    internal class CubicBSplineActivation : BaseActivation<CubicBSplineActivation>, IActivationFunction
    {
        public Quad MinValue => 0;

        public Quad MaxValue => (Quad)(2.0 / 3);

        public Quad Activation(ref Quad x)
        {
            double a = Math.Abs((double)x);
            if (a < 1) return (Quad)((4 - (6 * a * a) + (3 * a * a * a)) / 6);
            if (a < 2)
            {
                double rest = 2 - a;
                return (Quad)(rest * rest * rest / 6);
            }

            return 0;
        }

        public Quad Derivative(Quad x)
        {
            double value = x;
            double a = Math.Abs(value);
            if (a < 1) return (Quad)((-2 * value) + (1.5 * value * a));
            if (a < 2)
            {
                double rest = 2 - a;
                return (Quad)(-Math.Sign(value) * rest * rest / 2);
            }

            return 0;
        }
    }
}
