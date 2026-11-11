using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;
using static ANN.Net.Recurrent.RecurrentMath;

namespace ANN.Net.Recurrent
{
    /// <summary>
    /// Plain (Elman) recurrent layer: h(t) = tanh(W [x, h(t-1)] + b)
    /// </summary>
    internal sealed class SimpleRecurrentLayer : RecurrentLayer
    {
        public SimpleRecurrentLayer(int inputSize, int hiddenSize, OptimizerTypes optimizerType, ICollection<LearningRateOptimizer> optimizers)
            : base(inputSize, hiddenSize, 1, false, optimizerType, optimizers)
        {
        }

        private SimpleRecurrentLayer(SimpleRecurrentLayer master, ReplicaContext context)
            : base(master, context)
        {
        }

        public override RecurrentLayer Replicate(ReplicaContext context) => new SimpleRecurrentLayer(this, context);

        protected override void Forward(ReadOnlySpan<float> x, ReadOnlySpan<float> hPrevious, ReadOnlySpan<float> cPrevious,
                                        Span<float> gate, Span<float> extra, Span<float> h, Span<float> c)
        {
            Biases.Weights.CopyTo(gate);
            Sums(x, hPrevious, 0, gate);
            for (int j = 0; j < h.Length; j++)
            {
                gate[j] = MathF.Tanh(gate[j]);
                h[j] = gate[j];
            }
        }

        protected override void Backward(int t, ReadOnlySpan<float> dh, Span<float> dc, Span<float> deltas, Span<float> dhPrevious, Span<float> dx)
        {
            ReadOnlySpan<float> gate = Gates(t);
            for (int j = 0; j < deltas.Length; j++) deltas[j] = dh[j] * (1 - (gate[j] * gate[j]));

            Axpy(1, deltas, Biases.Gradients);
            AffineBackward(X(t), Weights.Weights, Weights.Gradients, 0, Columns, 0, deltas, dx);
            if (t > 0) AffineBackward(H(t - 1), Weights.Weights, Weights.Gradients, InputSize, Columns, 0, deltas, dhPrevious);
        }
    }
}
