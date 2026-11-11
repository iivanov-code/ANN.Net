using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;
using static ANN.Net.Recurrent.RecurrentMath;

namespace ANN.Net.Recurrent
{
    /// <summary>
    /// Long short-term memory layer. With z = [x, h(t-1)]:
    /// i = σ(W_i z + b_i), f = σ(W_f z + b_f), g = tanh(W_g z + b_g), o = σ(W_o z + b_o),
    /// c(t) = f ⊙ c(t-1) + i ⊙ g and h(t) = o ⊙ tanh(c(t)). With peepholes (Gers and Schmidhuber) the gates also see
    /// the cell: i and f get p_i ⊙ c(t-1) and p_f ⊙ c(t-1), o gets p_o ⊙ c(t). The gate columns are i, f, g, o
    /// </summary>
    internal sealed class LstmLayer : RecurrentLayer
    {
        private readonly RecurrentTensor peepholes;

        public LstmLayer(int inputSize, int hiddenSize, bool peepholes, float forgetGateBias, OptimizerTypes optimizerType,
                         ICollection<LearningRateOptimizer> optimizers)
            : base(inputSize, hiddenSize, 4, true, optimizerType, optimizers)
        {
            if (peepholes) this.peepholes = new RecurrentTensor(3 * hiddenSize, optimizerType, optimizers);
            Biases.Weights.Slice(hiddenSize, hiddenSize).Fill(forgetGateBias);
        }

        private LstmLayer(LstmLayer master, ReplicaContext context)
            : base(master, context)
        {
            peepholes = master.peepholes?.Replicate(context);
        }

        public override RecurrentLayer Replicate(ReplicaContext context) => new LstmLayer(this, context);

        public bool HasPeepholes => peepholes != null;

        /// <summary>
        /// p_i, p_f and p_o one after the other; null without peepholes
        /// </summary>
        public RecurrentTensor Peepholes => peepholes;

        public override IEnumerable<RecurrentTensor> Tensors => HasPeepholes ? new[] { Weights, Biases, peepholes } : base.Tensors;

        protected override void Forward(ReadOnlySpan<float> x, ReadOnlySpan<float> hPrevious, ReadOnlySpan<float> cPrevious,
                                        Span<float> gate, Span<float> extra, Span<float> h, Span<float> c)
        {
            int n = HiddenSize;
            Biases.Weights.CopyTo(gate);
            Sums(x, hPrevious, 0, gate);

            ReadOnlySpan<float> p = HasPeepholes ? peepholes.Weights : ReadOnlySpan<float>.Empty;
            for (int j = 0; j < n; j++)
            {
                float input = gate[j], forget = gate[n + j], output = gate[(3 * n) + j];
                if (HasPeepholes)
                {
                    input += p[j] * cPrevious[j];
                    forget += p[n + j] * cPrevious[j];
                }

                input = Sigmoid(input);
                forget = Sigmoid(forget);
                float candidate = MathF.Tanh(gate[(2 * n) + j]);
                float cell = (forget * cPrevious[j]) + (input * candidate);
                if (HasPeepholes) output += p[(2 * n) + j] * cell;

                output = Sigmoid(output);
                float squashed = MathF.Tanh(cell);

                gate[j] = input;
                gate[n + j] = forget;
                gate[(2 * n) + j] = candidate;
                gate[(3 * n) + j] = output;
                extra[j] = squashed;
                c[j] = cell;
                h[j] = output * squashed;
            }
        }

        protected override void Backward(int t, ReadOnlySpan<float> dh, Span<float> dc, Span<float> deltas, Span<float> dhPrevious, Span<float> dx)
        {
            int n = HiddenSize;
            ReadOnlySpan<float> gate = Gates(t), squashed = Extra(t), cPrevious = C(t - 1), cell = C(t);
            ReadOnlySpan<float> p = HasPeepholes ? peepholes.Weights : ReadOnlySpan<float>.Empty;
            Span<float> peepholeGradients = HasPeepholes ? peepholes.Gradients : Span<float>.Empty;

            for (int j = 0; j < n; j++)
            {
                float input = gate[j], forget = gate[n + j], candidate = gate[(2 * n) + j], output = gate[(3 * n) + j];

                float dOutput = dh[j] * squashed[j] * output * (1 - output);
                float dCell = dc[j] + (dh[j] * output * (1 - (squashed[j] * squashed[j])));
                if (HasPeepholes) dCell += p[(2 * n) + j] * dOutput;

                float dInput = dCell * candidate * input * (1 - input);
                float dForget = dCell * cPrevious[j] * forget * (1 - forget);
                float dCandidate = dCell * input * (1 - (candidate * candidate));

                dc[j] = dCell * forget;
                if (HasPeepholes)
                {
                    dc[j] += (p[j] * dInput) + (p[n + j] * dForget);
                    peepholeGradients[j] += dInput * cPrevious[j];
                    peepholeGradients[n + j] += dForget * cPrevious[j];
                    peepholeGradients[(2 * n) + j] += dOutput * cell[j];
                }

                deltas[j] = dInput;
                deltas[n + j] = dForget;
                deltas[(2 * n) + j] = dCandidate;
                deltas[(3 * n) + j] = dOutput;
            }

            Axpy(1, deltas, Biases.Gradients);
            AffineBackward(X(t), Weights.Weights, Weights.Gradients, 0, Columns, 0, deltas, dx);
            if (t > 0) AffineBackward(H(t - 1), Weights.Weights, Weights.Gradients, InputSize, Columns, 0, deltas, dhPrevious);
        }
    }
}
