using System;
using System.Collections.Generic;
using System.Linq;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Abstractions.Settings;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// Contrastive loss of pairs of embeddings, for siamese networks: one network embeds both inputs of a pair, and
    /// pairs labelled 1 should end up alike, pairs labelled -1 different. The mean over the pairs of the loss of
    /// <see cref="EmbeddingLossTypes"/>
    /// </summary>
    public sealed class EmbeddingPairLoss
    {
        private const double Epsilon = 1e-12;

        /// <param name="margin">α of <see cref="EmbeddingLossTypes.CosineEmbedding"/>, between -1 and 1: different
        /// pairs count only while their cosine is above it (0 is common); δ of
        /// <see cref="EmbeddingLossTypes.HingeEmbedding"/>, 0 or more: different pairs count only while their squared
        /// distance is below it</param>
        public EmbeddingPairLoss(EmbeddingLossTypes type, float margin)
        {
            BatchLossChecks.Margin(margin, nameof(margin));
            switch (type)
            {
                case EmbeddingLossTypes.CosineEmbedding when margin < -1 || margin > 1:
                    throw new ArgumentOutOfRangeException(nameof(margin), margin, "The margin of the cosine embedding loss must be between -1 and 1");
                case EmbeddingLossTypes.HingeEmbedding when margin < 0:
                    throw new ArgumentOutOfRangeException(nameof(margin), margin, "The margin of the hinge embedding loss must not be negative");
                case EmbeddingLossTypes.CosineEmbedding:
                case EmbeddingLossTypes.HingeEmbedding:
                    break;
                default:
                    throw new ArgumentException($"Unknown embedding loss {type}", nameof(type));
            }

            Type = type;
            Margin = margin;
        }

        public EmbeddingLossTypes Type { get; }

        public float Margin { get; }

        /// <param name="labels">1 for pairs that should be alike, -1 for pairs that should differ</param>
        public Quad CalculateLoss(IReadOnlyList<Quad[]> first, IReadOnlyList<Quad[]> second, IReadOnlyList<int> labels) =>
            Calculate(first, second, labels, null, null);

        /// <summary>
        /// Gradient of <see cref="CalculateLoss"/> with respect to every value of both embeddings of every pair
        /// </summary>
        public (Quad[][] First, Quad[][] Second) CalculateGradient(IReadOnlyList<Quad[]> first, IReadOnlyList<Quad[]> second, IReadOnlyList<int> labels)
        {
            int n = BatchLossChecks.SameCount(first, second, nameof(first), nameof(second));
            BatchLossChecks.Vectors(first, nameof(first));
            BatchLossChecks.Vectors(second, nameof(second));
            Quad[][] firstGradients = first.Select(x => new Quad[x.Length]).ToArray();
            Quad[][] secondGradients = new Quad[n][];
            for (int i = 0; i < n; i++) secondGradients[i] = new Quad[second[i].Length];

            Calculate(first, second, labels, firstGradients, secondGradients);
            return (firstGradients, secondGradients);
        }

        /// <summary>
        /// One training step of a siamese network on a batch of pairs: the network embeds the first and the second
        /// input of every pair, and both embeddings pass their gradients back; see
        /// <see cref="IVectorNetwork.TrainOutputs"/>. Returns the loss before the step
        /// </summary>
        public Quad Train(IVectorNetwork network, IReadOnlyList<Quad[]> firstInputs, IReadOnlyList<Quad[]> secondInputs, IReadOnlyList<int> labels,
                          BatchOptions options = null)
        {
            ArgumentNullException.ThrowIfNull(network);
            int n = BatchLossChecks.SameCount(firstInputs, secondInputs, nameof(firstInputs), nameof(secondInputs));
            BatchLossChecks.SameCount(firstInputs, labels, nameof(firstInputs), nameof(labels));

            return network.TrainOutputs(firstInputs.Concat(secondInputs).ToList(), (outputs, gradients) =>
            {
                Quad[][] firstGradients = Enumerable.Range(0, n).Select(i => new Quad[outputs[i].Length]).ToArray();
                Quad[][] secondGradients = Enumerable.Range(n, n).Select(i => new Quad[outputs[i].Length]).ToArray();
                Quad loss = Calculate(outputs.Take(n).ToList(), outputs.Skip(n).ToList(), labels, firstGradients, secondGradients);
                firstGradients.CopyTo(gradients, 0);
                secondGradients.CopyTo(gradients, n);
                return loss;
            }, options);
        }

        private Quad Calculate(IReadOnlyList<Quad[]> first, IReadOnlyList<Quad[]> second, IReadOnlyList<int> labels, Quad[][] firstGradients, Quad[][] secondGradients)
        {
            int n = BatchLossChecks.SameCount(first, second, nameof(first), nameof(second));
            BatchLossChecks.SameCount(first, labels, nameof(first), nameof(labels));
            BatchLossChecks.Vectors(first, nameof(first));
            BatchLossChecks.Vectors(second, nameof(second));
            BatchLossChecks.Labels(labels, nameof(labels));
            if (n == 0) return 0;

            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                BatchLossChecks.SameLength(first[i], second[i], nameof(second));
                sum += Type == EmbeddingLossTypes.CosineEmbedding
                    ? Cosine(first[i], second[i], labels[i], firstGradients?[i], secondGradients?[i], n)
                    : Hinge(first[i], second[i], labels[i], firstGradients?[i], secondGradients?[i], n);
            }

            return (Quad)(sum / n);
        }

        /// <summary>
        /// Loss of one pair; adds its share of the mean's gradient. Vectors of length 0 have no direction: their cosine
        /// counts as 0, with no gradient
        /// </summary>
        private double Cosine(Quad[] a, Quad[] b, int label, Quad[] aGradient, Quad[] bGradient, int pairs)
        {
            double dot = 0, aa = 0, bb = 0;
            for (int k = 0; k < a.Length; k++)
            {
                dot += (double)a[k] * b[k];
                aa += (double)a[k] * a[k];
                bb += (double)b[k] * b[k];
            }

            double aNorm = Math.Sqrt(aa), bNorm = Math.Sqrt(bb);
            bool directed = aNorm > Epsilon && bNorm > Epsilon;
            double cosine = directed ? dot / (aNorm * bNorm) : 0;

            // the loss changes with the cosine at this rate
            double loss, slope;
            if (label == 1)
            {
                loss = 1 - cosine;
                slope = -1;
            }
            else
            {
                loss = Math.Max(0, cosine - Margin);
                slope = cosine > Margin ? 1 : 0;
            }

            if (aGradient != null && directed && slope != 0)
            {
                // d cos / da = b / (|a| |b|) - cos · a / |a|²
                for (int k = 0; k < a.Length; k++)
                {
                    aGradient[k] = (Quad)(slope * ((b[k] / (aNorm * bNorm)) - (cosine * a[k] / aa)) / pairs);
                    bGradient[k] = (Quad)(slope * ((a[k] / (aNorm * bNorm)) - (cosine * b[k] / bb)) / pairs);
                }
            }

            return loss;
        }

        private double Hinge(Quad[] a, Quad[] b, int label, Quad[] aGradient, Quad[] bGradient, int pairs)
        {
            double distance = BatchLossChecks.SquaredDistance(a, b);
            double loss, slope;
            if (label == 1)
            {
                loss = distance;
                slope = 1;
            }
            else
            {
                loss = Math.Max(0, Margin - distance);
                slope = distance < Margin ? -1 : 0;
            }

            if (aGradient != null && slope != 0)
            {
                // d distance / da = 2 (a - b)
                for (int k = 0; k < a.Length; k++)
                {
                    double change = 2 * slope * ((double)a[k] - b[k]) / pairs;
                    aGradient[k] = (Quad)change;
                    bGradient[k] = (Quad)(-change);
                }
            }

            return loss;
        }
    }
}
