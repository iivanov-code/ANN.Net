using System.Threading.Tasks;
using ANN.Net.Remote.Abstactions.Models;

namespace ANN.Net.Remote.Abstactions.Interfaces
{
    /// <summary>
    /// Remote procedure contract of a network part that runs in another process. The part receives the values
    /// of the boundary neurons, runs them through its own layers and answers with its outputs; on
    /// backpropagation it trains its own synapses and answers with the gradient for every boundary neuron
    /// </summary>
    public interface IRemoteNetworkPart
    {
        RemoteNetworkInfo GetInfo();

        /// <summary>
        /// One value per boundary neuron, in the order the synapses were attached
        /// </summary>
        RemotePropagationResult Propagate(RemotePropagationDTO[] signals);

        /// <summary>
        /// Targets for the outputs of the remote part; the answer carries the loss and the gradient per boundary neuron
        /// </summary>
        RemoteBackpropagationResult Backpropagate(double[] targets);

        /// <summary>
        /// Sets the learning rate of every optimizer in the remote part
        /// </summary>
        Task SetLearningRateAsync(float learningRate);
    }
}
