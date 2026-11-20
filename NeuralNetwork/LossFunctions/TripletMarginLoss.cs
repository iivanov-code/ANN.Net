using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Triplet margin loss: an anchor should be closer to a positive example than to a negative one by at least the
    /// margin δ. The mean over the triplets of max(d(a, p) - d(a, n) + δ, 0), with d the squared Euclidean distance
    /// </summary>
    public sealed class TripletMarginLoss
    {
        public TripletMarginLoss(float margin = 1)
        {
            BatchLossChecks.Margin(margin, nameof(margin));
            Margin = margin;
        }

        public float Margin { get; }

        public Quad CalculateLoss(IReadOnlyList<Quad[]> anchors, IReadOnlyList<Quad[]> positives, IReadOnlyList<Quad[]> negatives) =>
            Calculate(anchors, positives, negatives, null);

        /// <summary>
        /// Gradient of <see cref="CalculateLoss"/> with respect to every value of the anchors, positives and negatives
        /// </summary>
        public (Quad[][] Anchors, Quad[][] Positives, Quad[][] Negatives) CalculateGradient(IReadOnlyList<Quad[]> anchors, IReadOnlyList<Quad[]> positives,
                                                                                         IReadOnlyList<Quad[]> negatives)
        {
            Check(anchors, positives, negatives);
            var gradients = new Quad[3][][];
            gradients[0] = anchors.Select(x => new Quad[x.Length]).ToArray();
            gradients[1] = positives.Select(x => new Quad[x.Length]).ToArray();
            gradients[2] = negatives.Select(x => new Quad[x.Length]).ToArray();
            Calculate(anchors, positives, negatives, gradients);
            return (gradients[0], gradients[1], gradients[2]);
        }

        /// <summary>
        /// One training step of a network that embeds anchors, positives and negatives; every embedding passes its
        /// gradient back. See <see cref="IVectorNetwork.TrainOutputs"/>. Returns the loss before the step
        /// </summary>
        public Quad Train(IVectorNetwork network, IReadOnlyList<Quad[]> anchorInputs, IReadOnlyList<Quad[]> positiveInputs, IReadOnlyList<Quad[]> negativeInputs,
                          BatchOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(network);
            int n = BatchLossChecks.SameCount(anchorInputs, positiveInputs, nameof(anchorInputs), nameof(positiveInputs));
            BatchLossChecks.SameCount(anchorInputs, negativeInputs, nameof(anchorInputs), nameof(negativeInputs));

            return network.TrainOutputs(anchorInputs.Concat(positiveInputs).Concat(negativeInputs).ToList(), (outputs, gradients) =>
            {
                var parts = new Quad[3][][];
                for (int part = 0; part < 3; part++) parts[part] = Enumerable.Range(part * n, n).Select(i => new Quad[outputs[i].Length]).ToArray();
                Quad loss = Calculate(outputs.Take(n).ToList(), outputs.Skip(n).Take(n).ToList(), outputs.Skip(2 * n).ToList(), parts);
                for (int part = 0; part < 3; part++) parts[part].CopyTo(gradients, part * n);
                return loss;
            }, options);
        }

        private static int Check(IReadOnlyList<Quad[]> anchors, IReadOnlyList<Quad[]> positives, IReadOnlyList<Quad[]> negatives)
        {
            int n = BatchLossChecks.SameCount(anchors, positives, nameof(anchors), nameof(positives));
            BatchLossChecks.SameCount(anchors, negatives, nameof(anchors), nameof(negatives));
            BatchLossChecks.Vectors(anchors, nameof(anchors));
            BatchLossChecks.Vectors(positives, nameof(positives));
            BatchLossChecks.Vectors(negatives, nameof(negatives));
            for (int i = 0; i < n; i++)
            {
                BatchLossChecks.SameLength(anchors[i], positives[i], nameof(positives));
                BatchLossChecks.SameLength(anchors[i], negatives[i], nameof(negatives));
            }

            return n;
        }

        /// <param name="gradients">Gradients of the anchors, positives and negatives, or null</param>
        private Quad Calculate(IReadOnlyList<Quad[]> anchors, IReadOnlyList<Quad[]> positives, IReadOnlyList<Quad[]> negatives, Quad[][][] gradients)
        {
            int n = Check(anchors, positives, negatives);
            if (n == 0) return 0;

            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                Quad[] a = anchors[i], p = positives[i], q = negatives[i];
                double z = BatchLossChecks.SquaredDistance(a, p) - BatchLossChecks.SquaredDistance(a, q) + Margin;
                if (z <= 0) continue;

                sum += z;
                if (gradients == null) continue;

                // d/da = 2 (a - p) - 2 (a - q) = 2 (q - p), d/dp = -2 (a - p), d/dq = 2 (a - q)
                for (int k = 0; k < a.Length; k++)
                {
                    gradients[0][i][k] = (Quad)(2 * ((double)q[k] - p[k]) / n);
                    gradients[1][i][k] = (Quad)(-2 * ((double)a[k] - p[k]) / n);
                    gradients[2][i][k] = (Quad)(2 * ((double)a[k] - q[k]) / n);
                }
            }

            return (Quad)(sum / n);
        }
    }
}
