namespace ANN.Net.Abstractions.Enums
{
    /// <summary>
    /// Loss of one sample, with e = predicted - target for every one of the n outputs. Losses marked "mean" divide by
    /// n, the others add up over the outputs. Parameters come from <see cref="Settings.LossOptions"/>
    /// </summary>
    public enum LossFunctionTypes
    {
        /// <summary>
        /// Not a loss: reports the sum of e, which shows whether the outputs are too high or too low, and trains like
        /// <see cref="Squared"/>
        /// </summary>
        Difference,

        /// <summary>
        /// Mean absolute error, mean of |e|
        /// </summary>
        Absolute,

        /// <summary>
        /// Half the sum of e², whose gradient is e
        /// </summary>
        Squared,

        /// <summary>
        /// Mean of max(0, 1 - target · predicted), for targets -1 and 1
        /// </summary>
        Hinge,

        /// <summary>
        /// Mean of δ² (√(1 + (e / δ)²) - 1): squared for small errors, linear for large ones, smooth everywhere
        /// </summary>
        PseudoHuber,

        /// <summary>
        /// Mean squared error, mean of e²
        /// </summary>
        MeanSquared,

        /// <summary>
        /// Mean of e² / 2 where |e| ≤ δ, δ (|e| - δ / 2) elsewhere
        /// </summary>
        Huber,

        /// <summary>
        /// Mean of e² / (2β) where |e| &lt; β, |e| - β / 2 elsewhere: <see cref="Huber"/> divided by β
        /// </summary>
        SmoothL1,

        /// <summary>
        /// Mean of ln(cosh(e))
        /// </summary>
        LogCosh,

        /// <summary>
        /// Mean of (ln(1 + predicted) - ln(1 + target))², for values above -1
        /// </summary>
        MeanSquaredLogarithmic,

        /// <summary>
        /// 100 times the mean of |e| / |target|, for targets that are not 0
        /// </summary>
        MeanAbsolutePercentage,

        /// <summary>
        /// Mean of predicted - target · ln(predicted), for counts and rates: positive predicted values
        /// </summary>
        Poisson,

        /// <summary>
        /// Sum of -target · ln(predicted), for probability outputs such as soft-max outputs and one-hot targets
        /// </summary>
        CrossEntropy,

        /// <summary>
        /// Mean of -(target · ln(predicted) + (1 - target) ln(1 - predicted)), for independent probabilities such as
        /// sigmoid outputs
        /// </summary>
        BinaryCrossEntropy,

        /// <summary>
        /// Sum of target · ln(target / predicted), the divergence of the predicted from the target distribution
        /// </summary>
        KullbackLeibler,

        /// <summary>
        /// Binary focal loss (Lin et al. 2017): mean of -(α target (1 - p)^γ ln(p) + (1 - α)(1 - target) p^γ ln(1 - p)),
        /// which lets well classified outputs count less
        /// </summary>
        Focal,

        /// <summary>
        /// Mean of max(0, 1 - target · predicted)², for targets -1 and 1
        /// </summary>
        SquaredHinge,

        /// <summary>
        /// Mean of ln(1 + exp(-target · predicted)), for targets -1 and 1
        /// </summary>
        SoftMargin,

        /// <summary>
        /// 1 - cos(predicted, target): 1 minus the cosine of the angle between the output and the target vector, so
        /// only their direction counts
        /// </summary>
        CosineDistance,
    }
}
