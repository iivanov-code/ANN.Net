namespace ANN.Net.Abstractions.Enums
{
    /// <summary>
    /// Functions a Kolmogorov-Arnold network adds up to form the learned function of every edge
    /// </summary>
    public enum KanBasisTypes
    {
        /// <summary>
        /// Cubic B-splines on a uniform grid, as in the original KAN (Liu et al. 2024): G + 3 bumps of
        /// <see cref="ActivationTypes.CubicBSpline"/> centred h apart from the grid minimum - h to the grid maximum + h,
        /// h = (maximum - minimum) / G. On the grid they add up to 1
        /// </summary>
        BSpline,

        /// <summary>
        /// Gaussian radial basis functions, as in FastKAN (Li 2024): G + 1 bumps e^(-((x - c) / h)²) with centres c from
        /// the grid minimum to the grid maximum, h apart. Smooth everywhere and close to the B-splines
        /// </summary>
        Gaussian,
    }
}
