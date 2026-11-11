using System;
using System.ComponentModel;
using ANN.Net.Abstractions.Enums;

namespace ANN.Net.Abstractions.Settings
{
    /// <summary>
    /// Parameters of the losses that have one
    /// </summary>
    public sealed class LossOptions
    {
        /// <summary>
        /// δ of <see cref="LossFunctionTypes.Huber"/> and <see cref="LossFunctionTypes.PseudoHuber"/>, and β of
        /// <see cref="LossFunctionTypes.SmoothL1"/>: the size of error where the loss turns from squared to linear
        /// </summary>
        [DefaultValue(1f)]
        public float Delta { get; set; } = 1;

        /// <summary>
        /// γ of <see cref="LossFunctionTypes.Focal"/>: how much less well classified outputs count; 0 gives the binary
        /// cross entropy weighted by <see cref="FocalAlpha"/>
        /// </summary>
        [DefaultValue(2f)]
        public float FocalGamma { get; set; } = 2;

        /// <summary>
        /// α of <see cref="LossFunctionTypes.Focal"/>: weight of the positive targets; the negative ones get 1 - α
        /// </summary>
        [DefaultValue(0.25f)]
        public float FocalAlpha { get; set; } = 0.25f;

        public void Validate()
        {
            if (!(Delta > 0) || float.IsInfinity(Delta)) throw new ArgumentException("Must be a positive number", nameof(Delta));
            if (!(FocalGamma >= 0) || float.IsInfinity(FocalGamma)) throw new ArgumentException("Must be 0 or a positive number", nameof(FocalGamma));
            if (!(FocalAlpha >= 0 && FocalAlpha <= 1)) throw new ArgumentException("Must be between 0 and 1", nameof(FocalAlpha));
        }
    }
}
