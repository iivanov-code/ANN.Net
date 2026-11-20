using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Ranking loss of a batch: the network gives every sample one score, and samples of higher relevance should score
    /// higher. The loss goes over all N² ordered pairs (i, j) of the batch, see <see cref="RankingLossTypes"/>. Pairs of
    /// a sample with itself and pairs of equal relevance count as well: they add a constant, and pull scores of equal
    /// relevance to within the margin of each other (to each other without a margin), so the smallest loss is not 0
    /// </summary>
    public sealed class PairwiseRankingLoss
    {
        /// <param name="margin">δ, by how much a more relevant sample should score higher; not used by
        /// <see cref="RankingLossTypes.PairwiseLogistic"/></param>
        public PairwiseRankingLoss(RankingLossTypes type, float margin = 1)
        {
            if (!Enum.IsDefined(type)) throw new ArgumentException($"Unknown ranking loss {type}", nameof(type));
            BatchLossChecks.Margin(margin, nameof(margin));
            Type = type;
            Margin = margin;
        }

        public RankingLossTypes Type { get; }

        public float Margin { get; }

        public Quad CalculateLoss(IReadOnlyList<Quad> scores, IReadOnlyList<Quad> relevance) => Calculate(scores, relevance, null);

        /// <summary>
        /// Gradient of <see cref="CalculateLoss"/> with respect to every score
        /// </summary>
        public Quad[] CalculateGradient(IReadOnlyList<Quad> scores, IReadOnlyList<Quad> relevance)
        {
            var gradient = new double[BatchLossChecks.SameCount(scores, relevance, nameof(scores), nameof(relevance))];
            Calculate(scores, relevance, gradient);
            return gradient.Select(x => (Quad)x).ToArray();
        }

        /// <summary>
        /// One training step of a network with one output, the score, on a batch of samples and their relevances;
        /// see <see cref="IVectorNetwork.TrainOutputs"/>. Returns the loss before the step
        /// </summary>
        public Quad Train(IVectorNetwork network, IReadOnlyList<Quad[]> inputs, IReadOnlyList<Quad> relevance, BatchOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(network);
            BatchLossChecks.SameCount(inputs, relevance, nameof(inputs), nameof(relevance));
            if (network.OutputSize != 1) throw new ArgumentException($"Ranking needs a network with one output, the score, not {network.OutputSize}", nameof(network));

            return network.TrainOutputs(inputs, (outputs, gradients) =>
            {
                Quad[] scores = outputs.Select(x => x[0]).ToArray();
                var gradient = new double[scores.Length];
                Quad loss = Calculate(scores, relevance, gradient);
                for (int i = 0; i < gradient.Length; i++) gradients[i] = new[] { (Quad)gradient[i] };
                return loss;
            }, options);
        }

        private Quad Calculate(IReadOnlyList<Quad> scores, IReadOnlyList<Quad> relevance, double[] gradient)
        {
            int n = BatchLossChecks.SameCount(scores, relevance, nameof(scores), nameof(relevance));
            if (n == 0) return 0;

            double margin = Type == RankingLossTypes.PairwiseLogistic ? 0 : Margin;
            double scale = 1.0 / ((double)n * n);
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double sign = relevance[i] > relevance[j] ? -1 : 1;
                    double z = (sign * ((double)scores[i] - scores[j])) + margin;
                    double slope;
                    if (Type == RankingLossTypes.MarginRanking)
                    {
                        sum += Math.Max(0, z);
                        slope = z > 0 ? 1 : 0;
                    }
                    else
                    {
                        sum += BatchLossChecks.Softplus(z);
                        slope = BatchLossChecks.Sigmoid(z);
                    }

                    if (gradient == null) continue;
                    gradient[i] += slope * sign * scale;
                    gradient[j] -= slope * sign * scale;
                }
            }

            return (Quad)(sum * scale);
        }
    }
}
