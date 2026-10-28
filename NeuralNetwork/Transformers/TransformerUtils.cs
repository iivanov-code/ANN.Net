using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Utils;

namespace ANN.Net.Transformers
{
    internal static class TransformerUtils
    {
        /// <summary>
        /// Creates the optimizer of a single trainable value and registers it,
        /// the same way <see cref="NetworkBuilder.ConnectAxon"/> does for a synapse
        /// </summary>
        public static IOptimizer CreateOptimizer(OptimizerTypes optimizerType, ICollection<LearningRateOptimizer> optimizers)
        {
            IOptimizer optimizer = NetworkUtils.GetOptimizer(optimizerType);

            if (optimizer is LearningRateOptimizer learningRateOptimizer)
            {
                optimizers.Add(learningRateOptimizer);
            }

            return optimizer;
        }

        public static Quad InitialWeight(WeightInitType initializationType, int fanIn, int fanOut)
        {
            switch (initializationType)
            {
                case WeightInitType.Xavier:
                    return NetworkUtils.GlorotWeight(fanIn, fanOut);

                case WeightInitType.Random:
                    return NetworkUtils.GetRandomNumber(-1, 1, 4);

                default:
                    throw new ArgumentException("Invalid value", nameof(initializationType));
            }
        }

        public static Quad[][] CreateSequence(int length, int dimension)
        {
            Quad[][] sequence = new Quad[length][];
            for (int i = 0; i < length; i++)
            {
                sequence[i] = new Quad[dimension];
            }

            return sequence;
        }

        /// <summary>
        /// Element-wise sum of two sequences of equal shape
        /// </summary>
        public static Quad[][] SequenceSum(Quad[][] a, Quad[][] b)
        {
            if (a.Length != b.Length)
                throw new ArgumentException("Sequences of different lengths found");

            Quad[][] sum = new Quad[a.Length][];
            for (int i = 0; i < a.Length; i++)
            {
                if (a[i].Length != b[i].Length)
                    throw new ArgumentException("Vectors of different sizes found");

                sum[i] = new Quad[a[i].Length];
                for (int j = 0; j < a[i].Length; j++)
                {
                    sum[i][j] = a[i][j] + b[i][j];
                }
            }

            return sum;
        }

        /// <summary>
        /// In place soft-max. Values equal to <see cref="Quad.NegativeInfinity"/> get probability zero
        /// </summary>
        public static void SoftMax(Quad[] values)
        {
            Quad max = Quad.NegativeInfinity;
            for (int i = 0; i < values.Length; i++)
            {
                if (values[i] > max)
                {
                    max = values[i];
                }
            }

            double[] exponents = new double[values.Length];
            double sum = 0;
            for (int i = 0; i < values.Length; i++)
            {
                exponents[i] = System.Math.Exp(values[i] - max);
                sum += exponents[i];
            }

            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (Quad)(exponents[i] / sum);
            }
        }
    }
}
