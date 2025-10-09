using System;
using System.Collections.Generic;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.ActivationFunctions
{
    [Serializable]
    internal abstract class BaseActivation<T>
         where T : class, IActivationFunction
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
                    instance.Add(typeof(T), Activator.CreateInstance<T>());
                }

                return instance[typeof(T)] as T;
            }
        }
    }
}
