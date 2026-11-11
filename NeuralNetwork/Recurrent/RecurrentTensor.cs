using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Optimizers;
using ANN.Net.Optimizers;

namespace ANN.Net.Recurrent
{
    /// <summary>
    /// Trainable array of a recurrent network with the gradient collected for it over a sequence. The gradient is
    /// applied once the whole sequence is done, so it can be clipped first
    /// </summary>
    internal sealed class RecurrentTensor
    {
        private readonly Quad[] values;
        private readonly Quad[] gradients;
        private readonly ArrayOptimizer optimizer;

        public RecurrentTensor(int length, OptimizerTypes optimizerType, ICollection<LearningRateOptimizer> optimizers)
        {
            values = new Quad[length];
            gradients = new Quad[length];
            optimizer = ArrayOptimizer.Create(optimizerType, length, optimizers);
        }

        private RecurrentTensor(RecurrentTensor master, ReplicaContext context)
        {
            values = master.values;
            gradients = new Quad[master.values.Length];
            optimizer = context.Link(master.optimizer, master.values);
        }

        /// <summary>
        /// Tensor of a training replica: it shares the values and collects its gradients for the batch
        /// </summary>
        public RecurrentTensor Replicate(ReplicaContext context) => new RecurrentTensor(this, context);

        public int Length => values.Length;

        public Quad[] Values => values;

        public Span<float> Weights => MemoryMarshal.Cast<Quad, float>(values.AsSpan());

        public Span<float> Gradients => MemoryMarshal.Cast<Quad, float>(gradients.AsSpan());

        public double SquaredGradientNorm()
        {
            double sum = 0;
            foreach (float gradient in Gradients) sum += (double)gradient * gradient;
            return sum;
        }

        public void ScaleGradients(float factor)
        {
            foreach (ref float gradient in Gradients) gradient *= factor;
        }

        /// <summary>
        /// One optimizer step with the collected gradient, which is then cleared
        /// </summary>
        public void Apply()
        {
            optimizer.Update(values, gradients);
            Array.Clear(gradients);
        }
    }
}
