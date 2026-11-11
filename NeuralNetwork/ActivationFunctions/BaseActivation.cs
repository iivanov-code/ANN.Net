using System;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    /// <summary>
    /// One shared instance per activation type, created on first use in a thread-safe way
    /// </summary>
    [Serializable]
    internal abstract class BaseActivation<T>
         where T : class, IActivationFunction
    {
        private static readonly Lazy<T> instance = new Lazy<T>(Activator.CreateInstance<T>);

        public static T Instance => instance.Value;
    }
}
