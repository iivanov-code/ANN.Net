using ANN.Net.Abstractions.Interfaces;
using ANN.Net.HelperClasses;
using System;

namespace ANN.Net.Neurons
{
    internal class BiasNeuron : INeuron, IInputNeuron, IHiddenNeuron
    {
        public const float VALUE = 1;

        public BiasNeuron()
        {
            Outputs = new SynapseCollection<ISynapse>();
            Inputs = new SynapseCollection<ISynapse>();
        }

        public Guid ID => Guid.NewGuid();

        public ISynapseCollection<ISynapse> Outputs { get; }

        public ISynapseCollection<ISynapse> Inputs { get; }

        public void Backpropagate(float errorSignal, float eWeightedSignal = 0, Action<float> updateWeight = null)
        {
            Outputs.AccountSignal();
            if (Outputs.CheckCountAndReset())
                updateWeight(errorSignal * VALUE);
        }

        public void Propagate(float value)
        {
            if (Inputs.Count > 0)
                Inputs?.AccountSignal();

            if (Inputs.CheckCountAndReset())
                foreach (var output in Outputs)
                {
                    output.Propagate(VALUE);
                }
        }
    }
}
