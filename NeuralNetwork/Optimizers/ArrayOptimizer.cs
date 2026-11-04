using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using static System.Math;

namespace ANN.Net.Optimizers
{
    /// <summary>
    /// One optimizer for a whole array of trainable values that are always updated together (a weight matrix, a bias
    /// vector, an embedding row). Its state lives in arrays, one entry per value, and the step count is shared, so it
    /// replaces one optimizer object per value. Every update rule is the same as in the per-value optimizer of the type
    /// (<see cref="Adam"/>, <see cref="RMSprop"/>, ...), so the values change exactly as they would there
    /// </summary>
    [Serializable]
    internal sealed class ArrayOptimizer : LearningRateOptimizer
    {
        private const float Epsilon = 1e-8f;

        private static readonly Quad Beta1 = 0.9f;
        private static readonly Quad Beta2 = 0.999f;
        private static readonly Quad Beta = 0.9f;

        private readonly Quad[] m;
        private readonly Quad[] v;
        private readonly Quad[] vHat;
        private uint t = 1;
        private Quad correction1, correction2;

        public ArrayOptimizer(OptimizerTypes type, int length)
        {
            if (length < 0) throw new ArgumentOutOfRangeException(nameof(length), length, "Must not be negative");

            Type = type;
            Length = length;
            switch (type)
            {
                case OptimizerTypes.Normal:
                    break;
                case OptimizerTypes.Momentum:
                case OptimizerTypes.Nesterov:
                case OptimizerTypes.RMSprop:
                    v = new Quad[length];
                    break;
                case OptimizerTypes.Adam:
                case OptimizerTypes.AdaMax:
                case OptimizerTypes.Nadam:
                    m = new Quad[length];
                    v = new Quad[length];
                    break;
                case OptimizerTypes.AmsGrad:
                    m = new Quad[length];
                    v = new Quad[length];
                    vHat = new Quad[length];
                    break;
                default:
                    throw new ArgumentException("Invalid optimizer type", nameof(type));
            }

            Prepare();
        }

        public OptimizerTypes Type { get; }

        public int Length { get; }

        /// <summary>
        /// Values of state kept per trainable value
        /// </summary>
        public int StatePerValue => (m == null ? 0 : 1) + (v == null ? 0 : 1) + (vHat == null ? 0 : 1);

        /// <summary>
        /// Creates the optimizer and registers it, so the learning rate can be set through the collection
        /// </summary>
        public static ArrayOptimizer Create(OptimizerTypes type, int length, ICollection<LearningRateOptimizer> optimizers)
        {
            var optimizer = new ArrayOptimizer(type, length);
            optimizers.Add(optimizer);
            return optimizer;
        }

        /// <summary>
        /// One step of every value: values[i] changes by its gradient gradients[i]
        /// </summary>
        public void Update(Quad[] values, Quad[] gradients)
        {
            for (int i = 0; i < Length; i++)
            {
                values[i] = Calculate(i, gradients[i], values[i]);
            }

            Advance();
        }

        /// <summary>
        /// New value of element <paramref name="index"/>; call <see cref="Advance"/> once all elements of the step are done
        /// </summary>
        public Quad Calculate(int index, Quad gradient, Quad weight)
        {
            switch (Type)
            {
                case OptimizerTypes.Normal:
                    return weight - (gradient * LearningRate);

                case OptimizerTypes.Momentum:
                    v[index] = (Beta * v[index]) + (LearningRate * gradient);
                    return weight - v[index];

                case OptimizerTypes.Nesterov:
                    v[index] = (Beta * v[index]) + gradient;
                    return weight - (LearningRate * (gradient + (Beta * v[index])));

                case OptimizerTypes.RMSprop:
                    v[index] = (Beta * v[index]) + ((1 - Beta) * gradient * gradient);
                    return weight - (LearningRate * gradient / ((Quad)Sqrt(v[index]) + Epsilon));

                case OptimizerTypes.Adam:
                {
                    m[index] = (Beta1 * m[index]) + ((1 - Beta1) * gradient);
                    v[index] = (Beta2 * v[index]) + ((1 - Beta2) * gradient * gradient);
                    Quad mHat = m[index] / correction1;
                    Quad vHatValue = v[index] / correction2;
                    return weight - (LearningRate * mHat / ((Quad)Sqrt(vHatValue) + Epsilon));
                }

                case OptimizerTypes.AdaMax:
                {
                    m[index] = (Beta1 * m[index]) + ((1 - Beta1) * gradient);
                    Quad mHat = m[index] / correction1;
                    v[index] = Max(Beta2 * v[index], Abs(gradient));
                    return weight - (LearningRate * mHat / (v[index] + Epsilon));
                }

                case OptimizerTypes.Nadam:
                {
                    m[index] = (Beta1 * m[index]) + ((1 - Beta1) * gradient);
                    v[index] = (Beta2 * v[index]) + ((1 - Beta2) * gradient * gradient);
                    Quad mHat = (Beta1 * m[index] / correction1) + ((1 - Beta1) * gradient / correction1);
                    Quad vHatValue = v[index] / correction2;
                    return weight - (LearningRate * mHat / ((Quad)Sqrt(vHatValue) + Epsilon));
                }

                case OptimizerTypes.AmsGrad:
                    m[index] = (Beta1 * m[index]) + ((1 - Beta1) * gradient);
                    v[index] = (Beta2 * v[index]) + ((1 - Beta2) * gradient * gradient);
                    vHat[index] = Max(v[index], vHat[index]);
                    return weight - (LearningRate * m[index] / ((Quad)Sqrt(vHat[index]) + Epsilon));

                default:
                    throw new InvalidOperationException($"Unknown optimizer {Type}");
            }
        }

        /// <summary>
        /// Ends a step: the bias corrections of Adam, AdaMax and Nadam move on
        /// </summary>
        public void Advance()
        {
            t++;
            Prepare();
        }

        private void Prepare()
        {
            correction1 = 1 - (Quad)Pow(Beta1, t);
            correction2 = 1 - (Quad)Pow(Beta2, t);
        }
    }
}
