using System;
using System.Collections.Generic;
using System.Linq;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Argument checks shared by the losses of whole batches
    /// </summary>
    internal static class BatchLossChecks
    {
        public static int SameCount<T1, T2>(IReadOnlyList<T1> first, IReadOnlyList<T2> second, string firstName, string secondName)
        {
            ArgumentNullException.ThrowIfNull(first, firstName);
            ArgumentNullException.ThrowIfNull(second, secondName);
            if (first.Count != second.Count)
                throw new ArgumentException($"{firstName} has {first.Count} items but {secondName} has {second.Count}", secondName);
            return first.Count;
        }

        public static void Vectors(IReadOnlyList<Quad[]> vectors, string name)
        {
            if (vectors.Any(x => x == null)) throw new ArgumentException("Every vector must be given", name);
        }

        public static void SameLength(Quad[] first, Quad[] second, string name)
        {
            if (first.Length != second.Length)
                throw new ArgumentException($"Vectors of {first.Length} and {second.Length} values cannot be compared", name);
        }

        public static void Labels(IReadOnlyList<int> labels, string name)
        {
            if (labels.Any(x => x != 1 && x != -1)) throw new ArgumentException("Every label must be 1 (alike) or -1 (different)", name);
        }

        public static void Actions(IReadOnlyList<Quad[]> outputs, IReadOnlyList<int> actions, string name)
        {
            for (int i = 0; i < actions.Count; i++)
            {
                if (outputs[i] == null) throw new ArgumentException("Every output vector must be given", nameof(outputs));
                if (actions[i] < 0 || actions[i] >= outputs[i].Length)
                    throw new ArgumentOutOfRangeException(name, actions[i], $"Sample {i} has {outputs[i].Length} outputs, one per action");
            }
        }

        public static void Margin(float margin, string name)
        {
            if (!float.IsFinite(margin)) throw new ArgumentException("Must be a finite number", name);
        }

        /// <summary>
        /// Squared Euclidean distance
        /// </summary>
        public static double SquaredDistance(Quad[] first, Quad[] second)
        {
            double sum = 0;
            for (int k = 0; k < first.Length; k++)
            {
                double difference = (double)first[k] - second[k];
                sum += difference * difference;
            }

            return sum;
        }

        /// <summary>
        /// ln(1 + exp(z)) without overflow
        /// </summary>
        public static double Softplus(double z) => z > 0 ? z + Math.Log(1 + Math.Exp(-z)) : Math.Log(1 + Math.Exp(z));

        public static double Sigmoid(double z) => z >= 0 ? 1 / (1 + Math.Exp(-z)) : Math.Exp(z) / (1 + Math.Exp(z));
    }
}
