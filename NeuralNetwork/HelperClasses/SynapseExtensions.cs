using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.HelperClasses
{
    public static class SynapseExtensions
    {
        public static bool Any(this ISynapseCollection<ISynapse> collection)
        {
            return collection != null && collection.Count > 0;
        }

    }
}
