using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;
using static ANN.Net.Recurrent.RecurrentMath;

namespace ANN.Net.Recurrent
{
    /// <summary>
    /// Gated recurrent unit (Cho et al. 2014): r = σ(W_r [x, h(t-1)] + b_r), u = σ(W_u [x, h(t-1)] + b_u),
    /// n = tanh(W_n [x, r ⊙ h(t-1)] + b_n) and h(t) = (1 - u) ⊙ n + u ⊙ h(t-1). The gate columns are r, u, n
    /// </summary>
    internal sealed class GruLayer : RecurrentLayer
    {
        private readonly float[] dReset;

        /// <param name="updateGateBias">Initial bias of the update gates; a positive bias keeps the state from the start</param>
        public GruLayer(int inputSize, int hiddenSize, float updateGateBias, OptimizerTypes optimizerType, ICollection<LearningRateOptimizer> optimizers)
            : base(inputSize, hiddenSize, 3, false, optimizerType, optimizers)
        {
            dReset = new float[hiddenSize];
            Biases.Weights.Slice(hiddenSize, hiddenSize).Fill(updateGateBias);
        }

        private GruLayer(GruLayer master, ReplicaContext context)
            : base(master, context)
        {
            dReset = new float[master.HiddenSize];
        }

        public override RecurrentLayer Replicate(ReplicaContext context) => new GruLayer(this, context);

        protected override void Forward(ReadOnlySpan<float> x, ReadOnlySpan<float> hPrevious, ReadOnlySpan<float> cPrevious,
                                        Span<float> gate, Span<float> extra, Span<float> h, Span<float> c)
        {
            int n = HiddenSize;
            ReadOnlySpan<float> weights = Weights.Weights;
            Biases.Weights.CopyTo(gate);
            Affine(x, weights, 0, Columns, 0, gate);
            Affine(hPrevious, weights, InputSize, Columns, 0, gate.Slice(0, 2 * n));

            for (int j = 0; j < n; j++)
            {
                gate[j] = Sigmoid(gate[j]);
                gate[n + j] = Sigmoid(gate[n + j]);
                extra[j] = gate[j] * hPrevious[j];
            }

            // the candidate reads the previous hidden vector through the reset gate
            Affine(extra, weights, InputSize, Columns, 2 * n, gate.Slice(2 * n, n));
            for (int j = 0; j < n; j++)
            {
                float candidate = MathF.Tanh(gate[(2 * n) + j]), update = gate[n + j];
                gate[(2 * n) + j] = candidate;
                h[j] = ((1 - update) * candidate) + (update * hPrevious[j]);
            }
        }

        protected override void Backward(int t, ReadOnlySpan<float> dh, Span<float> dc, Span<float> deltas, Span<float> dhPrevious, Span<float> dx)
        {
            int n = HiddenSize;
            ReadOnlySpan<float> gate = Gates(t), gatedHidden = Extra(t), hPrevious = H(t - 1);
            ReadOnlySpan<float> weights = Weights.Weights;
            Span<float> gradients = Weights.Gradients;

            for (int j = 0; j < n; j++)
            {
                float update = gate[n + j], candidate = gate[(2 * n) + j];
                dhPrevious[j] += dh[j] * update;
                deltas[n + j] = dh[j] * (hPrevious[j] - candidate) * update * (1 - update);
                deltas[(2 * n) + j] = dh[j] * (1 - update) * (1 - (candidate * candidate));
            }

            Span<float> candidateDeltas = deltas.Slice(2 * n, n);
            Array.Clear(dReset);
            AffineBackward(gatedHidden, weights, gradients, InputSize, Columns, 2 * n, candidateDeltas, dReset);

            for (int j = 0; j < n; j++)
            {
                float reset = gate[j];
                dhPrevious[j] += dReset[j] * reset;
                deltas[j] = dReset[j] * hPrevious[j] * reset * (1 - reset);
            }

            Axpy(1, deltas, Biases.Gradients);
            AffineBackward(X(t), weights, gradients, 0, Columns, 0, deltas, dx);
            if (t > 0) AffineBackward(hPrevious, weights, gradients, InputSize, Columns, 0, deltas.Slice(0, 2 * n), dhPrevious);
        }
    }
}
