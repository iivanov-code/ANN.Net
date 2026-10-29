using System;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.LossFunctions
{
    /// <summary>
    /// One shared instance per loss function type, created on first use in a thread-safe way
    /// </summary>
    internal abstract class BaseLossFunction<T>
        where T : class, ILossFunction
    {
        private static readonly Lazy<T> instance = new Lazy<T>(Activator.CreateInstance<T>);

        public static T Instance => instance.Value;
    }
}
