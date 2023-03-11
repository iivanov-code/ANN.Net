using ANN.Net.Abstractions.Arguments;
using ANN.Net.Abstractions.Collections;
using ANN.Net.Abstractions.Interfaces;
using ANN.Net.Utils;

namespace ANN.Net.Neurons
{
    internal abstract class BaseNeuron
    {
        public BaseNeuron(bool initInputs, bool initOutputs, uint? id = null)
        {
            if (initInputs) this.Inputs = new Synapses(ActivateNeuron);
            if (initOutputs) this.Outputs = new Synapses(FeedbackError);

            if (id == null || !id.HasValue)
            {
                ID = NetworkUtils.NewID();
            }
            else
            {
                ID = id.Value;
            }
        }

        public uint ID { get; }

        public ISynapses Inputs { get; protected set; }

        public ISynapses Outputs { get; protected set; }

        public abstract void Backpropagate(BackpropagateEventArgs errorSignal);

        public abstract void Propagate(NeuronPropagateEventArgs value);

        protected abstract void FeedbackError(BackpropagateEventArgs errorSignal);

        protected abstract void ActivateNeuron(NeuronPropagateEventArgs value);
    }
}
