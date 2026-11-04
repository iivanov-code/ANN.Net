using System;

namespace ANN.Net.Utils
{
    /// <summary>
    /// Gradients through the output transformations of the networks, for <c>IVectorNetwork.BackpropagateGradient</c>
    /// </summary>
    internal static class GradientUtils
    {
        /// <summary>
        /// Gradient with respect to v of y = v / Σv (the "soft-max" of neuron and matrix networks): (g_j - Σ g_i y_i) / Σv
        /// </summary>
        public static Quad[] ThroughDivideBySum(Quad[] gradients, Quad[] values)
        {
            double sum = 0;
            foreach (Quad value in values) sum += (float)value;

            double weighted = 0;
            for (int i = 0; i < values.Length; i++) weighted += (float)gradients[i] * ((float)values[i] / sum);

            var result = new Quad[values.Length];
            for (int i = 0; i < result.Length; i++) result[i] = (Quad)(((float)gradients[i] - weighted) / sum);
            return result;
        }

        /// <summary>
        /// Gradient with respect to z of y = softmax(z), given y: y_j (g_j - Σ g_i y_i)
        /// </summary>
        public static Quad[] ThroughSoftMax(Quad[] gradients, Quad[] outputs)
        {
            double weighted = 0;
            for (int i = 0; i < outputs.Length; i++) weighted += (float)gradients[i] * (float)outputs[i];

            var result = new Quad[outputs.Length];
            for (int i = 0; i < result.Length; i++) result[i] = (Quad)((float)outputs[i] * ((float)gradients[i] - weighted));
            return result;
        }

        public static void CheckLength(Quad[] gradients, int expected, string name)
        {
            ArgumentNullException.ThrowIfNull(gradients, name);
            if (gradients.Length != expected)
                throw new ArgumentException($"Expected {expected} gradients but found {gradients.Length}", name);
        }
    }
}
