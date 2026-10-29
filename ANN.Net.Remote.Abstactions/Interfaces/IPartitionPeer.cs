using System.Threading.Tasks;

namespace ANN.Net.Remote.Abstactions.Interfaces
{
    /// <summary>
    /// Contract between the parts of a breadth split: every part holds a slice of the same layers and the
    /// parts exchange their slices layer by layer. All deliveries are one-way; completion follows from the
    /// neurons' own input counting. Part 0 is the coordinator: it receives the inputs for everybody and
    /// collects the outputs and the input gradients
    /// </summary>
    public interface IPartitionPeer
    {
        /// <summary>
        /// Tells the receiving part which part the caller is, so answers can be routed back
        /// </summary>
        Task HelloAsync(int part);

        /// <summary>
        /// Coordinator to part: the full input vector of the partitioned section
        /// </summary>
        void DeliverInputs(double[] values);

        /// <summary>
        /// Part to part: activations of the sender's slice of <paramref name="layer"/>
        /// </summary>
        void DeliverActivations(int fromPart, int layer, double[] slice);

        /// <summary>
        /// Part to coordinator: activations of the sender's slice of the last layer
        /// </summary>
        void DeliverOutputs(int fromPart, double[] slice);

        /// <summary>
        /// Coordinator to part: gradient with respect to the receiver's slice of the last layer
        /// </summary>
        void DeliverOutputGradients(double[] slice);

        /// <summary>
        /// Part to part: the sender's partial gradient with respect to the receiver's slice of <paramref name="layer"/>
        /// </summary>
        void DeliverGradients(int fromPart, int layer, double[] slice);

        /// <summary>
        /// Part to coordinator: the sender's partial gradient with respect to the section inputs
        /// </summary>
        void DeliverInputGradients(int fromPart, double[] values);

        /// <summary>
        /// Part to coordinator: a delivery failed, so the current propagation cannot complete
        /// </summary>
        void ReportFailure(int fromPart, string message);

        Task SetLearningRateAsync(float learningRate);
    }
}
