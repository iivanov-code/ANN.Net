namespace ANN.Net.Utils
{
    public static partial class Math
    {
        public static Quad Max(Quad x, Quad y)
        {
            return x > y ? x : y;
        }

        public static Quad Min(Quad x, Quad y)
        {
            return x < y ? x : y;
        }
    }
}
