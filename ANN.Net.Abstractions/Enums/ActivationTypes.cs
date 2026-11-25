using System;

namespace ANN.Net.Abstractions.Enums
{
    /// <summary>
    /// Activation of a neuron, applied to the weighted sum of its inputs. Every value except <see cref="SoftMax"/> and
    /// <see cref="BReLU"/> can be used for any neuron, layer or graph node; those two are not functions of one neuron's sum
    /// and are rejected with a <see cref="NotSupportedException"/> when a network is built
    /// </summary>
    [Serializable]
    public enum ActivationTypes
    {
        /// <summary>
        /// Logistic sigmoid 1 / (1 + e^-x)
        /// </summary>
        Sigmoid = 0,

        /// <summary>
        /// ln(1 + e^x)
        /// </summary>
        SoftPlus = 1,

        /// <summary>
        /// Another name of the logistic sigmoid; computes exactly what <see cref="Sigmoid"/> computes
        /// </summary>
        SoftStep = 2,

        HyperbolicTangens = 3,
        ArcusTangens = 4,
        RELU = 5,

        /// <summary>
        /// ReLU with slope 0.01 below 0
        /// </summary>
        LeakyRELU = 6,

        /// <summary>
        /// ReLU with slope 0.25 below 0. The slope is fixed, not learned
        /// </summary>
        ParametricRELU = 7,

        /// <summary>
        /// Not a per-neuron activation: soft-max divides every output by the sum over the whole layer. Building a neuron
        /// with it throws <see cref="NotSupportedException"/>; set <c>OutputApplySoftMax</c> on the network settings instead
        /// </summary>
        SoftMax = 8,

        SWISH = 9,

        /// <summary>
        /// Step: 0 below 0, 1 from 0 on; the same function as <see cref="BinaryStep"/>
        /// </summary>
        BinaryRELU = 10,

        /// <summary>
        /// Step: 0 below 0, 1 from 0 on; its derivative is taken as 0 everywhere
        /// </summary>
        BinaryStep = 11,

        /// <summary>
        /// Not a per-neuron activation: bipolar ReLU flips every second neuron of a layer. Building a neuron with it throws
        /// <see cref="NotSupportedException"/>
        /// </summary>
        BReLU = 12,

        Gaussian = 13,
        ISRU = 14,
        Sinusoid = 15,
        SoftSign = 16,
        SQNL = 17,

        /// <summary>
        /// max(0, min(1, x / 6 + 1 / 2))
        /// </summary>
        HardSigmoid = 18,

        /// <summary>
        /// x - tanh(x)
        /// </summary>
        TanHShrink = 19,

        /// <summary>
        /// x - 0.5 above 0.5, x + 0.5 below -0.5, 0 between
        /// </summary>
        SoftShrink = 20,

        /// <summary>
        /// x outside [-0.5, 0.5], 0 inside
        /// </summary>
        HardShrink = 21,

        GELU = 22,

        /// <summary>
        /// Passes the value through unchanged (linear)
        /// </summary>
        Identity = 23,

        /// <summary>
        /// Cardinal cubic B-spline, the bump every basis function of a uniform cubic spline is a shifted copy of:
        /// (4 - 6x² + 3|x|³) / 6 for |x| &lt; 1, (2 - |x|)³ / 6 for 1 ≤ |x| &lt; 2, 0 elsewhere. The basis functions of
        /// Kolmogorov-Arnold networks (see <c>NetworkFactory.BuildKan</c>)
        /// </summary>
        CubicBSpline = 24
    }
}
