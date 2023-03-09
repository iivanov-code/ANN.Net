using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    internal abstract class BaseActivation<T>
        where T : class, IActivationFunction, new()
    {
        static BaseActivation()
        {
            instance = new Dictionary<Type, IActivationFunction>();
        }

        private static Dictionary<Type, IActivationFunction> instance;

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
