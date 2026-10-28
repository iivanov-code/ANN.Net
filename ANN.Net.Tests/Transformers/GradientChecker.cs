using System;
using Xunit;

namespace ANN.Net.Tests.Transformers
{
    /// <summary>
    /// Compares analytic gradients with central finite differences of the loss
    /// L = sum(outputs * upstream). Components must not change their weights between
    /// calls, which holds while every optimizer has the default learning rate of zero
    /// </summary>
    internal static class GradientChecker
    {
        public const double EPSILON = 5e-3;
        public const double TOLERANCE = 2e-2;

        public static Quad[][] RandomSequence(Random random, int length, int dimension, double scale = 1)
        {
            Quad[][] sequence = new Quad[length][];
            for (int t = 0; t < length; t++)
            {
                sequence[t] = new Quad[dimension];
                for (int i = 0; i < dimension; i++)
                {
                    sequence[t][i] = (Quad)((random.NextDouble() * 2 - 1) * scale);
                }
            }

            return sequence;
        }

        public static double Loss(Quad[][] outputs, Quad[][] upstream)
        {
            double loss = 0;
            for (int t = 0; t < outputs.Length; t++)
            {
                for (int i = 0; i < outputs[t].Length; i++)
                {
                    loss += (double)(float)outputs[t][i] * (float)upstream[t][i];
                }
            }

            return loss;
        }

        public static void AssertInputGradients(Func<Quad[][], Quad[][]> propagate,
                                                Func<Quad[][], Quad[][]> backpropagate,
                                                Quad[][] inputs,
                                                Quad[][] upstream)
        {
            double[][] numeric = new double[inputs.Length][];
            for (int t = 0; t < inputs.Length; t++)
            {
                numeric[t] = new double[inputs[t].Length];
                for (int i = 0; i < inputs[t].Length; i++)
                {
                    Quad original = inputs[t][i];

                    inputs[t][i] = original + (Quad)EPSILON;
                    double plus = Loss(propagate(inputs), upstream);

                    inputs[t][i] = original - (Quad)EPSILON;
                    double minus = Loss(propagate(inputs), upstream);

                    inputs[t][i] = original;
                    numeric[t][i] = (plus - minus) / (2 * EPSILON);
                }
            }

            propagate(inputs);
            Quad[][] analytic = backpropagate(upstream);

            Assert.Equal(inputs.Length, analytic.Length);
            for (int t = 0; t < inputs.Length; t++)
            {
                Assert.Equal(inputs[t].Length, analytic[t].Length);
                for (int i = 0; i < inputs[t].Length; i++)
                {
                    AssertClose(numeric[t][i], analytic[t][i], $"input gradient [{t}][{i}]");
                }
            }
        }

        public static void AssertClose(double expected, double actual, string what)
        {
            double difference = Math.Abs(expected - actual);
            double allowed = TOLERANCE * Math.Max(1, Math.Abs(expected));
            Assert.True(difference <= allowed, $"{what}: numeric {expected} vs analytic {actual}");
        }
    }
}
