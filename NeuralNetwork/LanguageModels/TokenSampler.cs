using System;
using System.Collections.Generic;

namespace ANN.Net.LanguageModels
{
    /// <summary>
    /// Picks a token from a probability distribution: greedy, with temperature, or from the top-k tokens only
    /// </summary>
    internal static class TokenSampler
    {
        private const double LOG_EPSILON = 1e-7;

        public static int Sample(Quad[] probabilities, float temperature, int topK, Random random)
        {
            if (probabilities == null) throw new ArgumentNullException(nameof(probabilities));
            if (probabilities.Length < 1) throw new ArgumentException("Distribution must not be empty", nameof(probabilities));
            if (float.IsNaN(temperature) || temperature < 0) throw new ArgumentOutOfRangeException(nameof(temperature), "Must not be negative");
            if (topK < 0) throw new ArgumentOutOfRangeException(nameof(topK), "Must not be negative");

            if (temperature == 0)
            {
                return ArgMax(probabilities);
            }

            if (random == null) throw new ArgumentNullException(nameof(random));

            List<int> candidates = Candidates(probabilities, topK);

            // soft-max of log(p) / temperature over the candidates, computed relative to the largest term
            double[] weights = new double[candidates.Count];
            double maxLogit = double.NegativeInfinity;
            for (int i = 0; i < candidates.Count; i++)
            {
                weights[i] = Math.Log(Math.Max((double)(float)probabilities[candidates[i]], LOG_EPSILON)) / temperature;
                maxLogit = Math.Max(maxLogit, weights[i]);
            }

            double total = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = Math.Exp(weights[i] - maxLogit);
                total += weights[i];
            }

            double point = random.NextDouble() * total;
            double cumulative = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                cumulative += weights[i];
                if (point < cumulative)
                {
                    return candidates[i];
                }
            }

            return candidates[candidates.Count - 1];
        }

        public static int ArgMax(Quad[] values)
        {
            int best = 0;
            for (int i = 1; i < values.Length; i++)
            {
                if (values[i] > values[best])
                {
                    best = i;
                }
            }

            return best;
        }

        private static List<int> Candidates(Quad[] probabilities, int topK)
        {
            List<int> candidates = new List<int>(probabilities.Length);
            for (int i = 0; i < probabilities.Length; i++)
            {
                candidates.Add(i);
            }

            if (topK > 0 && topK < probabilities.Length)
            {
                // most likely first, lowest id on ties, keep topK
                candidates.Sort((a, b) =>
                {
                    int byProbability = probabilities[b].CompareTo(probabilities[a]);
                    return byProbability != 0 ? byProbability : a.CompareTo(b);
                });
                candidates.RemoveRange(topK, candidates.Count - topK);
            }

            return candidates;
        }
    }
}
