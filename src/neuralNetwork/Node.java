package neuralNetwork;

public class Node {
	public float activation; // Activation value of this node (or input value if it's an input node)
	public float unsquishedActivation; // Value of the activation before it was put into the sigmoid function
	public float partialDerivative; // Stores ∂C/∂a for this node (0 for fully connected layer 1)
	  
	//constructor
	public Node() {
		unsquishedActivation = 0;
		activation = 0;
		partialDerivative = 0;
	}
}