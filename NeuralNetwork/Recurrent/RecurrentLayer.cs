using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Optimizers;
using ANN.Net.Transformers;

namespace ANN.Net.Recurrent
{
    /// <summary>
    /// Recurrent layer that runs a whole sequence and trains through it. Every step reads the input x and the state of
    /// the previous step, which is 0 before the first step. The weights are one matrix with a row per value of
    /// [x, h] and a column per gate unit, so a step is a single pass over it. What every step computed is kept for
    /// <see cref="Backpropagate"/>, which goes back through the steps and collects the gradient of every weight
    /// </summary>
    internal abstract class RecurrentLayer
    {
        private float[] inputs = Array.Empty<float>();
        private float[] hidden = Array.Empty<float>();
        private float[] cells = Array.Empty<float>();
        private float[] gates = Array.Empty<float>();
        private float[] extras = Array.Empty<float>();
        private float[] zeros = Array.Empty<float>();
        private int steps;

        protected RecurrentLayer(int inputSize, int hiddenSize, int gatesCount, bool hasCell, OptimizerTypes optimizerType,
                                 ICollection<Abstractions.Optimizers.LearningRateOptimizer> optimizers)
        {
            InputSize = inputSize;
            HiddenSize = hiddenSize;
            Columns = gatesCount * hiddenSize;
            HasCell = hasCell;
            Weights = new RecurrentTensor((inputSize + hiddenSize) * Columns, optimizerType, optimizers);
            Biases = new RecurrentTensor(Columns, optimizerType, optimizers);
        }

        protected RecurrentLayer(RecurrentLayer master, ReplicaContext context)
        {
            InputSize = master.InputSize;
            HiddenSize = master.HiddenSize;
            Columns = master.Columns;
            HasCell = master.HasCell;
            Weights = master.Weights.Replicate(context);
            Biases = master.Biases.Replicate(context);
        }

        public int InputSize { get; }

        public int HiddenSize { get; }

        /// <summary>
        /// Gate units: the columns of the weight matrix
        /// </summary>
        public int Columns { get; }

        /// <summary>
        /// Whether the state has a cell vector besides the hidden vector (an LSTM)
        /// </summary>
        public bool HasCell { get; }

        public RecurrentTensor Weights { get; }

        public RecurrentTensor Biases { get; }

        public virtual IEnumerable<RecurrentTensor> Tensors => new[] { Weights, Biases };

        public abstract RecurrentLayer Replicate(ReplicaContext context);

        /// <summary>
        /// Draws the weights; the biases start at 0
        /// </summary>
        public void Initialize(WeightInitType initializationType)
        {
            Span<float> weights = Weights.Weights;
            for (int i = 0; i < weights.Length; i++)
            {
                weights[i] = TransformerUtils.InitialWeight(initializationType, InputSize + HiddenSize, HiddenSize);
            }
        }

        /// <summary>
        /// Runs <paramref name="count"/> steps of the inputs, stored step after step, and returns the hidden vectors of
        /// all steps the same way
        /// </summary>
        public ReadOnlySpan<float> Propagate(ReadOnlySpan<float> sequence, int count)
        {
            steps = count;
            Ensure(ref inputs, count * InputSize);
            Ensure(ref hidden, count * HiddenSize);
            Ensure(ref gates, count * Columns);
            Ensure(ref extras, count * HiddenSize);
            if (HasCell) Ensure(ref cells, count * HiddenSize);
            Ensure(ref zeros, HiddenSize);

            sequence.Slice(0, count * InputSize).CopyTo(inputs);
            for (int t = 0; t < count; t++)
            {
                Forward(X(t), H(t - 1), C(t - 1), Gates(t), Extra(t), HiddenAt(t), CellAt(t));
            }

            return hidden.AsSpan(0, count * HiddenSize);
        }

        /// <summary>
        /// Takes the gradient of the hidden vectors of every step of the last sequence, adds the gradient of every
        /// weight to <see cref="RecurrentTensor.Gradients"/> and returns the gradient of the inputs
        /// </summary>
        public float[] Backpropagate(ReadOnlySpan<float> hiddenGradients)
        {
            if (steps == 0) throw new InvalidOperationException("Propagate must be called before Backpropagate");

            var inputGradients = new float[steps * InputSize];
            var dh = new float[HiddenSize];
            var dhPrevious = new float[HiddenSize];
            var dc = new float[HiddenSize];
            var deltas = new float[Columns];

            for (int t = steps - 1; t >= 0; t--)
            {
                ReadOnlySpan<float> outer = hiddenGradients.Slice(t * HiddenSize, HiddenSize);
                for (int i = 0; i < HiddenSize; i++) dh[i] = outer[i] + dhPrevious[i];

                Array.Clear(dhPrevious);
                Backward(t, dh, dc, deltas, dhPrevious, inputGradients.AsSpan(t * InputSize, InputSize));
            }

            return inputGradients;
        }

        /// <summary>
        /// Values <see cref="Step"/> keeps for the layer between steps: the hidden vector, then the cell vector
        /// </summary>
        public int StateSize => HasCell ? 2 * HiddenSize : HiddenSize;

        /// <summary>
        /// One step outside training: reads the state of the previous step from <paramref name="previous"/> and writes
        /// the new one to <paramref name="next"/>, both laid out as <see cref="StateSize"/>
        /// </summary>
        public void Step(ReadOnlySpan<float> x, ReadOnlySpan<float> previous, Span<float> next, Span<float> scratch)
        {
            int h = HiddenSize;
            ReadOnlySpan<float> cPrevious = HasCell ? previous.Slice(h, h) : ReadOnlySpan<float>.Empty;
            Span<float> c = HasCell ? next.Slice(h, h) : Span<float>.Empty;
            Forward(x, previous.Slice(0, h), cPrevious, scratch.Slice(0, Columns), scratch.Slice(Columns, h), next.Slice(0, h), c);
        }

        /// <summary>
        /// Scratch values <see cref="Step"/> needs
        /// </summary>
        public int ScratchSize => Columns + HiddenSize;

        /// <summary>
        /// One step: fills the gates (after their activation), the extra values the backward pass needs, the hidden
        /// vector and, with a cell, the cell vector
        /// </summary>
        protected abstract void Forward(ReadOnlySpan<float> x, ReadOnlySpan<float> hPrevious, ReadOnlySpan<float> cPrevious,
                                        Span<float> gate, Span<float> extra, Span<float> h, Span<float> c);

        /// <summary>
        /// Goes back through step t: <paramref name="dh"/> is the whole gradient of its hidden vector and
        /// <paramref name="dc"/> that of its cell vector, which is replaced by the gradient of the previous cell vector.
        /// Adds the gradient of the previous hidden vector to <paramref name="dhPrevious"/> and that of the input to
        /// <paramref name="dx"/>
        /// </summary>
        protected abstract void Backward(int t, ReadOnlySpan<float> dh, Span<float> dc, Span<float> deltas, Span<float> dhPrevious, Span<float> dx);

        protected ReadOnlySpan<float> X(int t) => inputs.AsSpan(t * InputSize, InputSize);

        /// <summary>
        /// Hidden vector of step t; zeros before the first step
        /// </summary>
        protected ReadOnlySpan<float> H(int t) => t < 0 ? zeros : hidden.AsSpan(t * HiddenSize, HiddenSize);

        protected ReadOnlySpan<float> C(int t) => !HasCell ? ReadOnlySpan<float>.Empty : t < 0 ? zeros : cells.AsSpan(t * HiddenSize, HiddenSize);

        protected Span<float> Gates(int t) => gates.AsSpan(t * Columns, Columns);

        protected Span<float> Extra(int t) => extras.AsSpan(t * HiddenSize, HiddenSize);

        private Span<float> HiddenAt(int t) => hidden.AsSpan(t * HiddenSize, HiddenSize);

        private Span<float> CellAt(int t) => HasCell ? cells.AsSpan(t * HiddenSize, HiddenSize) : Span<float>.Empty;

        /// <summary>
        /// Adds x and the previous hidden vector times their weight rows to the gate sums, over the columns
        /// [offset, offset + sums.Length)
        /// </summary>
        protected void Sums(ReadOnlySpan<float> x, ReadOnlySpan<float> hPrevious, int offset, Span<float> sums)
        {
            ReadOnlySpan<float> weights = Weights.Weights;
            RecurrentMath.Affine(x, weights, 0, Columns, offset, sums);
            RecurrentMath.Affine(hPrevious, weights, InputSize, Columns, offset, sums);
        }

        private static void Ensure(ref float[] buffer, int length)
        {
            if (buffer.Length < length) buffer = new float[length];
        }
    }
}
