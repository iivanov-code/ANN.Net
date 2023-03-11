using System;
using ANN.Net.ActivationFunctions;

namespace ANN.Net.Utils
{
    public static class MatrixUtils
    {
        public static Quad[] MatrixToVector(this Quad[,] matrix)
        {
            if (matrix.Rank != 2)
                throw new ArgumentException("Can not convert multidimentional cubes");

            if (matrix.GetLength(1) != 1)
                throw new ArgumentException("Matrix has more than one column");

            Quad[] vector = new Quad[matrix.Length];
            for (int i = 0; i < matrix.Length; i++)
            {
                vector[i] = matrix[i, 0];
            }
            return vector;
        }

        public static Quad[,] VectorToMatrix(this Quad[] vector)
        {
            Quad[,] matrix = new Quad[vector.Length, 1];
            for (int i = 0; i < vector.Length; i++)
            {
                matrix[i, 0] = vector[i];
            }

            return matrix;
        }

        public static Quad[] VectorSum(params Quad[][] vectors)
        {
            int count = vectors[0].Length;
            Quad[] newVector = new Quad[count];

            foreach (var vector in vectors)
            {
                for (int i = 0; i < count; i++)
                {
                    newVector[i] += vector[i];
                }
            }
            return newVector;
        }

        public static Quad[,] MatrixSum(params Quad[][,] matrixes)
        {
            int rows = matrixes[0].GetLength(0);
            int cols = matrixes[0].GetLength(1);

            Quad[,] newMatrix = new Quad[rows, cols];

            foreach (var matrix in matrixes)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int j = 0; j < cols; j++)
                    {
                        newMatrix[i, j] += matrix[i, j];
                    }
                }
            }

            return newMatrix;
        }

        public static Quad[,] MatrixTanh(Quad[,] matrix)
        {
            return matrix.ForEach(x => TanHActivation.Activate(ref x));
        }

        public static Quad[] VectorTanh(Quad[] vector)
        {
            Quad[] newVector = new Quad[vector.Length];
            for (int i = 0; i < vector.Length; i++)
            {
                newVector[i] = TanHActivation.Activate(ref vector[i]);
            }
            return newVector;
        }

        public static Quad[,] MatrixSigmoid(Quad[,] matrix)
        {
            return matrix.ForEach(x => SigmoidActivation.Activate(ref x));
        }

        public static Quad[,] CopyMatrix(Quad[,] matrix)
        {
            return matrix.ForEach(x => x, true);
        }

        public static Quad[,] RandomMatrix(int rows, int cols, Quad fromInclusive, Quad toInclusive)
        {
            Quad[,] matrix = new Quad[rows, cols];
            return matrix.ForEach(x => NetworkUtils.GetRandomNumber(fromInclusive, toInclusive));
        }

        public static Quad[,] MatrixHadamard(Quad[,] a, Quad[,] b)
        {
            int rows = a.GetLength(0);
            int cols = a.GetLength(1);

            Quad[,] result = new Quad[rows, cols];
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    result[i, j] = a[i, j] * b[i, j];
                }
            }

            return result;
        }

        public static Quad[] MatrixHadamard(Quad[] a, Quad[] b)
        {
            Quad[] newMatrix = new Quad[a.Length];
            for (int i = 0; i < a.Length; i++)
            {
                newMatrix[i] = a[i] * b[i];
            }

            return newMatrix;
        }

        public static Quad[,] MatrixProduct(Quad[,] a, Quad[,] b)
        {
            int rowsA = a.GetLength(0);
            int colsA = a.GetLength(1);
            int rowsB = b.GetLength(0);
            int colsB = b.GetLength(1);

            if (colsB == rowsA)
            {
                Swap(ref a, ref b);
                Swap(ref rowsA, ref rowsB);
                Swap(ref colsA, ref colsB);
            }
            else if (colsA != rowsB)
            {
                throw new ArgumentException("Columns in first matrix should be equal to rows in second matrix");
            }

            Quad[,] newMatrix = new Quad[rowsA, colsB];

            for (int i = 0; i < rowsA; i++)
            {
                for (int j = 0; j < colsB; j++)
                {
                    for (int k = 0; k < colsA; k++)
                    {
                        newMatrix[i, j] += a[i, k] * b[k, j];
                    }
                }
            }
            return newMatrix;
        }

        private static Quad[,] ForEach(this Quad[,] matrix, Func<Quad, Quad> func, bool copy = false)
        {
            int rows = matrix.GetLength(0);
            int cols = matrix.GetLength(1);

            Quad[,] newMatrix;

            if (copy)
            {
                newMatrix = new Quad[rows, cols];
            }
            else
            {
                newMatrix = matrix;
            }

            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    newMatrix[i, j] = func(matrix[i, j]);
                }
            }

            return newMatrix;
        }

        private static void Swap<T>(ref T lhs, ref T rhs)
        {
            T temp = lhs;
            lhs = rhs;
            rhs = temp;
        }
    }
}
