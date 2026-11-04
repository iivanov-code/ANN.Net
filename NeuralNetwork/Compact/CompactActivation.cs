using System;
using System.Numerics;
using System.Runtime.InteropServices;
using ANN.Net.Abstractions.Enums;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Utils;

namespace ANN.Net.Compact
{
    /// <summary>
    /// Activation of whole arrays: nothing for the identity, SIMD for ReLU, the activation function otherwise
    /// </summary>
    internal sealed class CompactActivation
    {
        private readonly IActivationFunction function;

        private CompactActivation(ActivationTypes type)
        {
            Type = type;
            function = type == ActivationTypes.Identity ? null : NetworkUtils.GetActivation(type);
        }

        public ActivationTypes Type { get; }

        public static CompactActivation Of(ActivationTypes type) => new CompactActivation(type);

        public float Apply(float value)
        {
            if (function == null) return value;

            Quad quad = value;
            return function.Activation(ref quad);
        }

        public void Apply(Span<float> values)
        {
            if (function == null) return;

            int i = 0;
            if (Type == ActivationTypes.RELU)
            {
                // as RELUActivation: negative values become 0, everything else (including NaN and -0) stays
                if (Vector.IsHardwareAccelerated)
                {
                    Span<Vector<float>> vectors = MemoryMarshal.Cast<float, Vector<float>>(values);
                    for (int v = 0; v < vectors.Length; v++)
                    {
                        vectors[v] = Vector.ConditionalSelect(Vector.LessThan(vectors[v], Vector<float>.Zero), Vector<float>.Zero, vectors[v]);
                    }

                    i = vectors.Length * Vector<float>.Count;
                }

                for (; i < values.Length; i++)
                {
                    if (values[i] < 0) values[i] = 0;
                }

                return;
            }

            for (; i < values.Length; i++)
            {
                Quad quad = values[i];
                values[i] = function.Activation(ref quad);
            }
        }
    }
}
