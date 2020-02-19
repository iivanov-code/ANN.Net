using NeuralNetwork.Interfaces;
using System.Collections.Generic;

namespace NeuralNetwork.HelperClasses
{
    public static class SynapseExtensions
    {
        public static bool Any(this ISynapseCollection<ISynapse> collection)
        {
            return collection != null && collection.Count > 0;
        }

    }
}
