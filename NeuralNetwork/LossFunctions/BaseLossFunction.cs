using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.LossFunctions
{
    internal abstract class BaseLossFunction<T>
        where T : class, ILossFunction, new()
    {
        static BaseLossFunction()
        {
            instance = new Dictionary<Type, ILossFunction>();
        }

        private static Dictionary<Type, ILossFunction> instance;

        public static T Instance
        {
            get
            {
                if (!instance.ContainsKey(typeof(T)))
                {
                    instance.Add(typeof(T), new T());
                }

                return instance[typeof(T)] as T;
            }
        }
    }
}
