using System.Threading;
using ANN.Net.Abstractions.Interfaces;

namespace ANN.Net.IdentityGenerators
{
    public class IncrementalGenerator : IUniqueIdentityGenerator
    {
        private uint _counter;

        public IncrementalGenerator()
        {
            this._counter = 0;
        }

        public string GenerateUniqueIdentity()
        {
            return Interlocked.Increment(ref _counter).ToString();
        }
    }
}
