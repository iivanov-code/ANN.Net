namespace ANN.Net.Abstractions.Enums
{
    /// <summary>
    /// How a node of a nested network joins the vectors it receives
    /// </summary>
    public enum NodeCombination
    {
        /// <summary>
        /// One after the other, in the order of its inputs: the input size is the sum of their sizes
        /// </summary>
        Concatenate = 0,

        /// <summary>
        /// Added element by element: every input has the input size
        /// </summary>
        Sum = 1
    }
}
