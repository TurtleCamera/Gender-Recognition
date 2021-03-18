package neuralNetwork;

public class Convolution {
	public Node [][] nodeMatrix; // Contains the nodes in the convolution (keep in mind that the partial derivatives with respect to the activations are in here too)
	public float [][][] kernelMatrices; // Contains the matrices of the kernels that are used for the convolutions connected to this convolution in the next layer
	public float [][][] stochasticPartialDerivativeKernelMatrices; // Contains the average partial derivative of the cost function with respect to the weights in the kernel matrixes (in code, it doesn't actually take the average until backpropogation)
	public float bias; // The bias attached to this convolution
	public float stochasticPartialDerivativeBias; // The average partial derivative of the cost function with respect to the bias (in code, it doesn't actually take the average until backpropogation)
	
	public Convolution(int nodeMatrixSize, int numConnectedConvolutions, int kernelSize) {
		nodeMatrix = new Node [nodeMatrixSize][nodeMatrixSize]; // nodeMatrixSize is the side of this convolution
		kernelMatrices = new float [numConnectedConvolutions][kernelSize][kernelSize]; // numConnectedConvolutions is the number of convolutions in the next layer that this is connected to while kernel size is self-explanatory
		stochasticPartialDerivativeKernelMatrices = new float [numConnectedConvolutions][kernelSize][kernelSize]; // Still self-explanatory
		
		// Create the nodes in the nodeMatrix
		for(int i = 0; i < nodeMatrixSize; i ++) {
			for(int j = 0; j < nodeMatrixSize; j ++) {
				nodeMatrix[i][j] = new Node();
				Main.loadingCount ++; // This is just for the loading bar
				Main.paint.repaint();
			}                                                        
		}
		
		// Randomly generate weights in the kernels
		for(int i = 0; i < numConnectedConvolutions; i ++) {
			for(int j = 0; j < kernelSize; j ++) {
				for(int k = 0; k < kernelSize; k ++) {
					kernelMatrices[i][j][k] = Main.rng.nextFloat() * 2 - 1;
					Main.loadingCount ++; // This is just for the loading bar
					Main.paint.repaint();
				}
			}
		}
		
		// Randomly generate the bias value from -5 to 5
		bias = Main.rng.nextFloat() * 10 - 5;
		stochasticPartialDerivativeBias = 0;
		Main.loadingCount ++; // This is just for the loading bar
		Main.paint.repaint();
	}
}
