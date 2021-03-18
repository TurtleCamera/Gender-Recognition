package neuralNetwork;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Random;

import javax.imageio.ImageIO;

public class Main {
	
	// Some constants
	public static final int SCREEN_WIDTH = 1920; // Width of the 
	public static final int SCREEN_HEIGHT = 1080; // Height of the frame
	
	public static final int INPUT_CONVOLUTION_SIZE = 74; // Size of input layer matricies
	public static final int KERNEL_SIZE = 7; // Size of each kernel (also used to determine size of the convolution layers after the input layer)
	public static final int INPUT_CONVOLUTION_LAYER_SIZE = 3; // Number of convolutions in the input layer
	public static final int FIRST_HIDDEN_CONVOLUTION_LAYER_SIZE = 3; // Number of convolutions in the first two hidden convolution layers
	public static final int SECOND_HIDDEN_CONVOLUTION_LAYER_SIZE = 6; // Number of convolutions in the first two hidden convolution layers
	public static int vectorizationLayerSize; // Number of nodes in the vectorization layer (can't be final because it's being computed)
	public static final int FULLY_CONNECTED_HIDDEN_LAYER_SIZE = 128; // Number of nodes in the hidden layers
	public static final int GENDER_OUTPUT_LAYER_SIZE = 2; // Number of nodes in the output layer in the gender network
	
	public static final int AVERAGE_STOCHASTIC_DERIVATIVES = 50; // Used to average the weights and biases in the stochastic weight matrices and bias vectors
	public static final int MOD_BY_FIFTY = 50; // Used to check which image we are in within an image group
	
	public static final int GRADIENT_COMPONENT_UNITS = 1; // Multiplied with ∂C/∂b and ∂C/∂w to accelerate learning if these values are small (this is an experimental value, but it should be 1)
	
	public static final int YEAR_TAKEN_INDEX = 0; // Year taken index in the secondary data file
	public static final int GENDER_INDEX = 1; // Gender index in the secondary data file
	public static final int PATH_INDEX = 2; // Path index in the secondary data file
	public static final int NUM_SECONDARY_FILE_ARGS = 3; // The secondary file should have 3 arguments
	
	public static final double MAX_ASPECT_RATIO = 1.3; // Any images with a higher aspect ratio will be ignored
	
	public static final int RED_INDEX = 0; // Red input convolution matrix index
	public static final int GREEN_INDEX = 1; // Green input convolution matrix index
	public static final int BLUE_INDEX = 2; // Blue input convolution matrix index
	
	public static final int TOTAL_STARTUP_ACTIONS = 4092555; // The amount of actions (theoretically) during the startup sequence (used for the loading bar)
	
	public static final int SEED = 1; // Set the RNG for all weights and biases
	public static final Random rng = new Random(SEED); // The RNG that determines all of the weights and biases
													
	
	public static InputFrame frame; // The frame that will display the game
	public static PaintPanel paint; // The panel that draws the objects
	
	public static FileInputStream fisImages; // File input stream for the images
	public static DataInputStream disImages; // Data input stream, which we use to read these pixel values
	public static FileInputStream fisLabels; // File input stream for the labels
	public static DataInputStream disLabels; // Data input stream, which we use to read these label values
	
	public static FileInputStream fisImdbDOB; // File input stream for IMDB DoBs
	public static DataInputStream disImdbDOB; // Data input stream, which we use to check if the dob imdb file exists
	public static FileInputStream fisImdbOther; // File input stream for other IMDB info
	public static DataInputStream disImdbOther; // Data input stream, which we use to check if the second imdb file exists
	public static FileInputStream fisWikiDOB; // File input stream for Wiki DoBs
	public static DataInputStream disWikiDOB; // Data input stream, which we use to check if the dob wiki file exists
	public static FileInputStream fisWikiOther; // File input stream for other Wiki info
	public static DataInputStream disWikiOther; // Data input stream, which we use to check if the second wiki file exists
	public static BufferedReader brImdbDOB; // Used to read the contents of the IMDB dob file
	public static BufferedReader brImdbOther; // Used to read the contents of the second IMDB file
	public static BufferedReader brWikiDOB; // Used to read the contents of the Wiki dob file
	public static BufferedReader brWikiOther; // Used to read the contents of the second Wiki file
	
	public static BufferedImage currentImage; // Current image file we are using
	public static int currentAge; // Stores the rounded age of the current person
	public static String currentBirthDate; // The current person's birth date we are using
	public static int currentYearTaken; // The current photo's year taken
	public static int currentGender; // The current person's gender (1 for male, 0 for female, and -1 for unknown)
	public static String currentPath; // The current image's path in either directory
	
	public static int numWikiImages; // Number of Wiki images
	public static int numImdbImages; // Number of IMDB images
	public static int totalImages; // Number of total images in the training set
	public static int genderCutoffImageValidCount; // Leave 10% of images for testing
	public static int numLabels; // Number of labels in the training file
	public static int loadingCount; // Counts how much valid training data we've gone through
	public static int totalValidGenderCount; // Counts how much valid gender training data we've gone through
	public static int totalValidAgeCount; // Counts how much valid age training data we've gone through
	public static int validGenderCount; // Gender count of the useable images we went through
	public static int validAgeCount; // Age count of the useable images we went through
	public static int genderCount; // Gender count of the all images we went through
	public static int ageCount; // Age count of the all images we went through
	public static int largestGenderKernelWeightMagnitude; // Used only for the purpose of weight intensities on the paint panel
	
	public static Convolution [] genderInputConvolutionLayer; // Input convolutions that store all of the image data in the gender prediction neural network
	public static Convolution [] genderHiddenConvolutionLayer1; // First hidden convolution layer gender prediction neural network
	public static Convolution [] genderHiddenConvolutionLayer2; // Second hidden convolution layer gender prediction neural network
	
	public static Node genderFullyConnectedLayer1[]; // Nodes for the fully connected layer 1 gender prediction neural network
	public static Node genderFullyConnectedLayer2[]; // Nodes for the fully connected layer 1 gender prediction neural network
	public static Node genderFullyConnectedLayer3[]; // Nodes for the fully connected layer 1 gender prediction neural network
	public static Node genderOutputLayer[]; // Nodes for the fully connected layer 1 gender prediction neural network
	
	public static float genderWeightMatrix1[][]; // Weight matrix between fully connected layer 1 and fully connected layer 2 gender prediction neural network
	public static float genderWeightMatrix2[][]; // Weight matrix between fully connected layer 2 and fully connected layer 3 gender prediction neural network
	public static float genderWeightMatrix3[][]; // Weight matrix between fully connected layer 3 and output layer  gender prediction neural network
	
	public static float[] genderBiasVector1; // Bias vector for fully connected layer 2 gender prediction neural network
	public static float[] genderBiasVector2; // Bias vector for fully connected layer 3 gender prediction neural network
	public static float[] genderBiasVector3; // Bias vector for output layer gender prediction neural network
	
	public static float genderStochasticPartialDerivativeWeightMatrix1[][]; // Average of the 10 partial derivatives in the weight matrix between fully connected layer 1 and fully connected layer 2 gender prediction neural network
	public static float genderStochasticPartialDerivativeWeightMatrix2[][]; // Average of the 10 partial derivatives in the weight matrix between fully connected layer 2 and fully connected layer 3 gender prediction neural network
	public static float genderStochasticPartialDerivativeWeightMatrix3[][]; // Average of the 10 partial derivatives in the weight matrix between fully connected layer 3 and output layer  gender prediction neural network
	
	public static float[] genderStochasticPartialDerivativeBiasVector1; // Average of the 10 partial derivatives in the bias vector for fully connected layer 2 gender prediction neural network
	public static float[] genderStochasticPartialDerivativeBiasVector2; // Average of the 10 partial derivatives in the bias vector for fully connected layer 3 gender prediction neural network
	public static float[] genderStochasticPartialDerivativeBiasVector3; // Average of the 10 partial derivatives in the bias vector for output layer gender prediction neural network
	
	public static float genderDesiredOutput[]; // An array (vector) containing all of the desired activations for a given image

	public static boolean genderInvalid = false; // Tells if the current image isn't usable for the gender network
	public static boolean ageInvalid = false; // Tells if the current image isn't usable for the age network
	public static boolean inWiki = true; // If true, we are reading wiki images. Otherwise we are reading IMDB images
	public static boolean inTraining = true; // While this is true, we train the neural network (when false, we enter the accuracy test)
	public static boolean loaded = false; // Tells the PaintPanel when it can start painting
	public static boolean next = false; // Controlled by the user-tells when to load the next image
	public static boolean first = true; // Prevents the first (blank) image to be copied into the lastNineImages arraylist
	public static boolean autoAdvance = false; // Auto-advance option
	public static boolean darkMode = false; // Option to turn on a dark background
	public static boolean disableGraphics = false; // Sometimes it's better to turn this off so the program runs faster
	public static boolean disableWeightRestriction = false; // When this option is enabled, weights are restricted to [-1, 1] (note that it's recommended that this option remain constant throughout training)
	public static boolean inTesting = false; // Tells when the program is testing the neural network
	public static boolean fileWritten = false; // Tells whether a file has been written
	public static boolean useFemale = true; // Keeps track of which gender we should be feeding into the network
	
	public static float genderCost; // The genderCost of the current image displayed
	public static float averageGenderCost; // The total genderCost of the last 50 images displayed
	
	public static int guessedGender = 0; // The number that the neural network guesses after processing an image
	
	public static int numGenderCorrect = 0; // Used during the accuracy test phase to determine the % of images read correctly
	
	public static BufferedWriter networkFile; // Ultimately writes the network and statistics file
	
	public static ArrayList<Age_Person> ageData; // Contains all valid age data
	public static ArrayList<String> femaleData; // Contains all valid female gender data (No need to make a class because we're sorting these file paths into arraylists of differend genders)
	public static ArrayList<String> maleData; // Contains all valid male gender data (No need to make a class because we're sorting these file paths into arraylists of differend genders)
	
	public static void main(String[] args) throws IOException, InterruptedException {	
		// Create the GUI
		createFrame();
		createPanel();
		
		// For the loading percentage, use loadingCount
		loadingCount = 0;
		
		// Load the files
		loadFiles();
		
		// Create the gender network
		loadGenderNetwork();
		
		// Begin painting
		loaded = true; 

		loadingCount = TOTAL_STARTUP_ACTIONS; // Set it to the max loadingCount incase we for some reason we don't hit 100% in the loading bar
		Thread.sleep(500);

		// The loop for training begins
		while(inTraining) {
			// Check if auto-advance is on and automatically set next to true if it's on
			if(autoAdvance) {
				next = true;
			}
			
			// Go to the next image when the user presses space or enter
			if(next) {
				// Redraw the frame again and sleep to prevent instant updating
				paint.repaint();
				Thread.sleep(1);
				
				// Stop the while loop from loading more images
				next = false;
				
				// Don't continue if we're done with wiki images
				if(!checkGenderCompleted() && nextGenderData()) {
					// Get inputs
					loadData();
					
					// Check if we are on the first image of an image group
					if(checkFirstGenderImage()) {
						clearStochasticWeightsAndBiases();
					}
							
					// Feed forward
					feedForward();
							
					// Calculate the genderCost and average genderCost of the current image using the genderCost function
					calculateGenderCost();
					calculateAverageGenderCost();
							
					// Guess the output
					guessedGender = guessGender();
				
					// Finally calculate partial derivatives (and backpropagate when we sum the average gradient vector of the past 50 images)
					backPropagate();
				}
			}
			
			// Redraw the frame
			paint.repaint();
		}
		
		// Training completed. Now we wait for the user to test
		while(!inTesting) {
			// Redraw the frame
			paint.repaint();
		}
		
		// Set the validGenderCount to 0 to count how many images we went through for testing
		validGenderCount = 0;
		
		// Clear the genderCosts and partial derivatives in activations as well
		clearCosts();
		
		// Note that when in the testing phase, we don't use the extra male images (or female if we have more female images, but it's male for the test set provided)
		try {
			// The loop for accuracy testing begins
			while(inTesting) {
				// Check if auto-advance is on and automatically set next to true if it's on
				if(autoAdvance) {
					next = true;
				}
				
				// Go to the next image when the user presses space or enter
				if(next) {
					// Redraw the frame again and sleep to prevent instant updating
					paint.repaint();
					Thread.sleep(1);
					
					// Stop the while loop from loading more images
					next = false;
					
					// Don't continue if we're done testing
					if(!checkTestingCompleted() && nextGenderData()) {
						// Get inputs
						loadData();
						
						// Feed forward
						feedForward();
						
						// Calculate the genderCost and average genderCost of the current image using the genderCost function (Not used, but still good to look at for testing)
						calculateGenderCost();
						calculateAverageGenderCost();
						
						// Guess the output
						guessedGender = guessGender();
							
						// Check this guess
						guessedCorrect();
					}
				}
				
				// Redraw the frame
				paint.repaint();
			}
			
			// At this point, we're done. Wait for the user to print the file
			while(!fileWritten) {
				// Now we just wait for the user to close the program
				paint.repaint();
			}
		}
		catch(Exception e) {
			// Print the file immediately
			e.printStackTrace();
		}
		
		// Write the file
		writeFile();
	
		// Everything is finished. Wait for the user to close the program
		while(true) {
			// Now we just wait for the user to close the program
			paint.repaint();
		}
	}

	public static void calculateAverageGenderCost() {
		// Don't do this if we have an invalid image
		if(!genderInvalid) {
			int n = (validGenderCount - 1) % MOD_BY_FIFTY + 1;
			// Reset the averagegenderCost if we're looking at a new data group
			if(n == 1) {
				Main.averageGenderCost = 0;
			}
			
			// Sum the current genderCost into the average
			averageGenderCost += genderCost;
		}
	}
	
	public static void calculateGenderCost() {
		// Don't do this if we have an invalid image
		if(!genderInvalid) {
			// genderCost is the sum of the squared differences of the network's activation outputs and the desired outputs (use the traditional genderCost formula)
			genderCost = 0;
			for(int i = 0; i < GENDER_OUTPUT_LAYER_SIZE; i ++) {
				genderCost += Math.pow(genderOutputLayer[i].activation - genderDesiredOutput[i], 2);
			}
		}
	}
	
	public static void createPanel() {
		// Create and add the paint panel to the InputFrame
		paint = new PaintPanel();
		paint.setPreferredSize(new Dimension(SCREEN_WIDTH, SCREEN_HEIGHT));
		frame.add(paint);
	    paint.setFocusable(true);
	}
	
	public static void createFrame() {
		//Create the window
	    frame = new InputFrame();
	    frame.setDefaultCloseOperation(InputFrame.EXIT_ON_CLOSE);
	    frame.setSize(SCREEN_WIDTH, SCREEN_HEIGHT);
	    
	    //Display the window.
	    frame.setLocationRelativeTo(null);
	    frame.setVisible(true);     
	    frame.setFocusable(true);
	    frame.addKeyListener(frame);
	}
	
	public static void loadData() throws IOException {
		// Resize the image to a 256x256 image
		currentImage = resizeImage(currentImage, INPUT_CONVOLUTION_SIZE, INPUT_CONVOLUTION_SIZE);
		
		// Collect RGB values
		for(int i = 0; i < INPUT_CONVOLUTION_SIZE; i ++) {
			for(int j = 0; j < INPUT_CONVOLUTION_SIZE; j ++) {
				Color color = new Color(currentImage.getRGB(i, j));
				
				// Get each individual RGB value
				genderInputConvolutionLayer[BLUE_INDEX].nodeMatrix[i][j].unsquishedActivation = color.getBlue();
				genderInputConvolutionLayer[GREEN_INDEX].nodeMatrix[i][j].unsquishedActivation = color.getGreen();
				genderInputConvolutionLayer[RED_INDEX].nodeMatrix[i][j].unsquishedActivation = color.getRed();
			}
		}
		
		// Squish all genderFullyConnectedLayer1 activations
		squishInputActivations();
		
		/* TODO */
		/*
		// Set the rounded age if we can use this data
		if(!ageInvalid) {
			// Assume photo was taken in the middle of the year
			int roundedBirthDate;
			
			currentAge = currentYearTaken - ;
		}
		*/
		
		// Update gender label
		if(!genderInvalid) {
			// Reset the desired output vector and fill the desired node with 1.0
			for(int i = 0; i < genderDesiredOutput.length; i ++) {
				genderDesiredOutput[i] = 0;
			}
			genderDesiredOutput[currentGender] = 1.0f;
		}
	}
	
	@SuppressWarnings("unused")
	public static boolean nextGenderData() {
		// Count regardless of whether image is invalid
		genderCount ++;
		
		// Assume that the gender image is valid
		genderInvalid = false;
		
		String path;
		if(useFemale) {
			path = femaleData.get(0);
			femaleData.remove(0);
			currentGender = 0;
		}
		else {
			path = maleData.get(0);
			maleData.remove(0);
			currentGender = 1;
		}
		
		// Use the opposite gender the next round
		useFemale = !useFemale;

		// Check if the current image is even usable
		File imageFile = new File(path);
		
		// Perform 3 checks: does the image exist (including number directory)? is the image roughly a square? is the image broken (1x1 pixel image)?
		if(imageFile != null) {
			// Safe to load image
			try {
				currentImage = ImageIO.read(imageFile);
			} catch (IOException e) {
				// This failed, so return false
				genderInvalid = true;
				return false;
			}
			
			// Check image's aspect ratio
			if((double) currentImage.getHeight() / currentImage.getWidth() >= MAX_ASPECT_RATIO || (double) currentImage.getWidth() / currentImage.getHeight() >= MAX_ASPECT_RATIO) {
				genderInvalid = true;
				return false;
			}
			
			// Check if broken image
			if(currentImage.getWidth() == 1 || currentImage.getHeight() == 1) {
				genderInvalid = true;
				return false;
			}
		}
		else {
			genderInvalid = true;
			return false;
		}

		// Successfully loaded another image
		validGenderCount ++;
		
		return true;
	}
	
    public static void squishInputActivations() {
    	// NOTE: THIS IS NOT THE SAME AS THE SIGMOID FUNCTION. THIS FUNCTION JUST TAKES THE PROPORTION OF 255 FOR THE UNSQUISHED ACTIVATION 
    	for(int i = 0; i < genderInputConvolutionLayer.length; i ++) {
    		for(int j = 0; j < genderInputConvolutionLayer[i].nodeMatrix.length; j ++) {
    			for(int k = 0; k < genderInputConvolutionLayer[i].nodeMatrix[j].length; k ++) {
    	    		genderInputConvolutionLayer[i].nodeMatrix[j][k].activation = genderInputConvolutionLayer[i].nodeMatrix[j][k].unsquishedActivation / ((float) PaintPanel.MAX_COLOR_VALUE); 
    			}
    		}
    	}
    }
    
    public static void feedForward() {
    	// Feed forward the gender network first
	    if(!genderInvalid) {
	    	// Feed input convolution layer into hidden convolution layer 1
	    	feedForwardConvolutions(genderInputConvolutionLayer, genderHiddenConvolutionLayer1);
	    	
	    	// Feed input hidden convolution layer 1 into hidden convolution layer 2
	    	feedForwardConvolutions(genderHiddenConvolutionLayer1, genderHiddenConvolutionLayer2);
    	}
    	
    	// Vectorize the last hidden convolution layer
    	vectorization();
    	
	    if(!genderInvalid) {
	    	// Feed fully connected layer 1 into fully connected layer 2
	    	matrixMultiplication(genderWeightMatrix1, genderFullyConnectedLayer1, genderBiasVector1, genderFullyConnectedLayer2);
	    	
	    	// Feed fully connected layer 2 into fully connected layer 3
	    	matrixMultiplication(genderWeightMatrix2, genderFullyConnectedLayer2, genderBiasVector2, genderFullyConnectedLayer3);
	    	
	    	// Feed fully connected layer 3 into output layer
	    	matrixMultiplication(genderWeightMatrix3, genderFullyConnectedLayer3, genderBiasVector3, genderOutputLayer);
	    }
    }
    
    public static void matrixMultiplication(float genderWeightMatrix[][], Node layerVector[], float genderBiasVector[], Node resultingVector[]) {
    	// First clear the resulting vector
    	clearResultingVector(resultingVector);
    	
    	// In terms of linear algebra, to calculate the resultingVector (i.e. the layer we are feeding forward to), we simply use the Ax + b = r formula
    	// where A is the weight matrix, x is the layer vector containing the activations of the nodes we want to feed forward, b is the bias vector, and
    	// r is the resulting vector we are feeding into
    	for(int r = 0; r < genderWeightMatrix.length; r ++) {
    		// Columns of genderWeightMatrix, rows of layerVector, rows of genderBiasVector, and rows of resultingVector are assumed to be equal
    		for(int c = 0; c < genderWeightMatrix[0].length; c ++) {
    			// Multiply row of weightMatric to column of layerVector and sum the result
    			resultingVector[r].unsquishedActivation += genderWeightMatrix[r][c] * layerVector[c].activation;
    		}
    		
    		// Then add the bias
    		resultingVector[r].unsquishedActivation += genderBiasVector[r];
    		
    		// Finally, squish the resulting activation value by using the sigmoid function
    		resultingVector[r].activation = sigmoid(resultingVector[r].unsquishedActivation);
    	}
    }
    
    public static void feedForwardConvolutions(Convolution [] inputConvolutionLayer, Convolution [] outputConvolutionLayer) {
    	// First clear the resulting convolution
    	clearResultingConvolution(outputConvolutionLayer);
    	
    	// Feed forward specifically for convolutions
    	int splitSize = inputConvolutionLayer[0].kernelMatrices.length; // Can use 0 because all of them are the same length
    	
    	for(int i = 0; i < inputConvolutionLayer.length; i ++) {
        	for(int j = 0; j < splitSize; j ++) {
        		for(int k = 0; k < outputConvolutionLayer[splitSize * i + j].nodeMatrix.length; k ++) {
	            	for(int l = 0; l < outputConvolutionLayer[splitSize * i + j].nodeMatrix[k].length; l ++) {
			            for(int m = 0; m < inputConvolutionLayer[i].kernelMatrices[j].length; m ++) {
				            for(int n = 0; n < inputConvolutionLayer[i].kernelMatrices[j][m].length; n ++) {
				            	outputConvolutionLayer[splitSize * i + j].nodeMatrix[k][l].unsquishedActivation += inputConvolutionLayer[i].nodeMatrix[k + m][l + n].activation * inputConvolutionLayer[i].kernelMatrices[j][m][n];
				            }
			            }
	            	}
        		}
        	}
    	}
    	
		// Finally, squish all activations
    	squishConvolutionActivations(outputConvolutionLayer);
    }
    
    public static void clearResultingVector(Node resultingVector[]) {
    	for(int i = 0; i < resultingVector.length; i ++) {
    		// Don't need to change squished activation because it will be calculated anyway
    		resultingVector[i].unsquishedActivation = 0;
    	}
    }
    
    public static void clearResultingConvolution(Convolution [] resultingConvolution) {
    	for(int i = 0; i < resultingConvolution.length; i ++) {
    		for(int j = 0; j < resultingConvolution[i].nodeMatrix.length; j ++) {
    			for(int k = 0; k < resultingConvolution[i].nodeMatrix[j].length; k ++) {
    	    		// Don't need to change squished activation because it will be calculated anyway
    				resultingConvolution[i].nodeMatrix[j][k].unsquishedActivation = 0;
    			}
    		}
    	}
    }
    
    public static void squishConvolutionActivations(Convolution [] outputConvolutionLayer) {
    	// We need to squish the convolution activations in a different function and not do it in the nested for loop because we are summing data from all convolutions in the previous layer
    	for(int i = 0; i < outputConvolutionLayer.length; i ++) {
    		for(int j = 0; j < outputConvolutionLayer[i].nodeMatrix.length; j ++) {
    			for(int k = 0; k < outputConvolutionLayer[i].nodeMatrix[j].length; k ++) {
    	    		// Don't need to change squished activation because it will be calculated anyway
    				outputConvolutionLayer[i].nodeMatrix[j][k].activation = sigmoid(outputConvolutionLayer[i].nodeMatrix[j][k].unsquishedActivation + outputConvolutionLayer[i].bias);
    			}
    		}
    	}
    }
    
    public static void vectorization() {
    	// Convert all convolutions in the last hidden convolution layer to a single vector
    	int fullyConnectedLayerIndex = 0; // Keeps track of which index we are on in the first fully connected layer

	    if(!genderInvalid) {
	    	// Start with the gender network first
	    	for(int i = 0; i < genderHiddenConvolutionLayer2.length; i ++) {
	    		for(int j = 0; j < genderHiddenConvolutionLayer2[i].nodeMatrix.length; j ++) {
	    			for(int k = 0; k < genderHiddenConvolutionLayer2[i].nodeMatrix[j].length; k ++) {
	    	    		// Don't need to change squished activation because it will be calculated anyway
	    				genderFullyConnectedLayer1[fullyConnectedLayerIndex ++] = genderHiddenConvolutionLayer2[i].nodeMatrix[j][k];
	    			}
	    		}
	    	}
	    }
    }
    
	// The sigmoid function used to translate the numbers to a value in the range of 0 to 1
	public static float sigmoid(float x) {
		return (float) (1 / (1 + Math.pow((float) Math.E, -1 * x)));
	}
	
	public static int guessGender() {
		// Determine the output node with the highest activation and return its index
		int guessIndex = 0;
		
		for(int i = 0; i < genderOutputLayer.length; i ++) {
			if(genderOutputLayer[i].activation > genderOutputLayer[guessIndex].activation) {
				guessIndex = i;
			}
		}
		
		return guessIndex;
	}
    
	// The derivative of the sigmoid function
	public static float derivativeSigmoid(float x) {
		return (float) (Math.pow((float) Math.E, -1 * x) / Math.pow(1 + Math.pow((float) Math.E, -1 * x), 2));
	}
	
	public static void backPropagate() {
		// Don't do any of this if we got an invalid image
		if(!genderInvalid) {
			// Start propagation from the output weight layer
			deriveGenderWeightMatrix3();
			
			// Propagate the last bias vector before moving on to the next weight layer
			deriveBiasVector3();
			
			// Store summation calculations of ∂C/∂a(L-1) for the fully connected layer 3 activations
			deriveGenderFullyConnectedLayer3Activations();
			
			// Now propagate to the second weight layer
			deriveGenderWeightMatrix2();
			
			// Propagate the second bias vector before moving on to the next weight layer
			deriveBiasVector2();
			
			// Store summation calculations of ∂C/∂a(L-2) for the fully connected layer 2 activations
			deriveGenderFullyConnectedLayer2Activations();
			
			// Now propagate to the first weight layer
			deriveGenderWeightMatrix1();
			
			// Finally, propagate the first bias vector
			deriveBiasVector1();
			
			// Store summation calculations of ∂C/∂a(L-3) for the fully connected layer 1 activations (we will start to use these in the convolution layerz)
			deriveGenderFullyConnectedLayer1Activations();
			
			// Because of the way these convolutions are formatted, we can derive ∂C/∂k, ∂C/∂a, and ∂C/∂b all in a function call
			deriveConvolutionKernelBiasAndActivation(genderHiddenConvolutionLayer1, genderHiddenConvolutionLayer2, true); 
			deriveConvolutionKernelBiasAndActivation(genderInputConvolutionLayer, genderHiddenConvolutionLayer1, false); // Don't derive the activations in the input layer because we won't even use them
			
			// If we traversed 50 images, adjust the weights and biases
			if(checkFiftyGenderImages()) {
				// Now we tell the neural network to learn
				adjustGenderWeightsAndBiases();
				
				// Find the max magnitude of kernel weights
				findMaxKernelWeightMagnitude();
			}
		}
	}
	
	public static void deriveGenderWeightMatrix3() {
		// Calculate ∂C/∂w(L) for all weights in the last weight layer
		// This double for loop will traverse the columns first before moving to the next row in the weight matrix (the indices might look weird because I'm following my notes)
		for(int j = 0; j < genderOutputLayer.length; j ++) {
			for(int i = 0; i < genderFullyConnectedLayer3.length; i ++) {
				// Formula: ∂C/∂w(L) = ∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂w(L)
				genderStochasticPartialDerivativeWeightMatrix3[j][i] += -1 * 2 * (genderOutputLayer[j].activation - genderDesiredOutput[j]) * derivativeSigmoid(genderOutputLayer[j].unsquishedActivation) * genderFullyConnectedLayer3[i].activation;
			}
		}
	}
	
	public static void deriveBiasVector3() {
		// Calculate ∂C/∂b(L) for all weights in the last weight layer
		// This double for loop will traverse the columns first before moving to the next row in the weight matrix (the indices might look weird because I'm following my notes)
		for(int j = 0; j < genderOutputLayer.length; j ++) {
			// Formula: ∂C/∂w(L) = ∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂b(L)
			genderStochasticPartialDerivativeBiasVector3[j] += -1 * 2 * (genderOutputLayer[j].activation - genderDesiredOutput[j]) * derivativeSigmoid(genderOutputLayer[j].unsquishedActivation) * 1;
		}
	}
	
	public static void deriveGenderFullyConnectedLayer3Activations() {
		// Reset the partial derivatives first
		clearPartialDerivativesToActivations(genderFullyConnectedLayer3);
		
		// Calculate ∂C/∂a(L-1) for all activation nodes in fully connected layer 3 (this is just to make calculations easier for later weights, specifically the Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) part in deriveGenderWeightMatrix2())
		// This double for loop will traverse the rows first before moving to the next column in the weight matrix (sum derivatives of each node)
		for(int i = 0; i < genderFullyConnectedLayer3.length; i ++) {
			for(int j = 0; j < genderOutputLayer.length; j ++) {
				// Formula: ∂C/∂a(L-1) = ∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)
				// Don't compute negative gradient for this
				genderFullyConnectedLayer3[i].partialDerivative = 2 * (genderOutputLayer[j].activation - genderDesiredOutput[j]) * derivativeSigmoid(genderOutputLayer[j].unsquishedActivation) * genderWeightMatrix3[j][i];
			}
		}
	}
	
	public static void deriveGenderWeightMatrix2() {
		// Calculate ∂C/∂w(L-1) for all weights in the second weight layer
		// This double for loop will traverse the columns first before moving to the next row in the weight matrix (the indices might look weird because I'm following my notes)
		for(int i = 0; i < genderFullyConnectedLayer3.length; i ++) {
			for(int h = 0; h < genderFullyConnectedLayer2.length; h ++) {
				// Formula: ∂C/∂w(L-1) = Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) * ∂a(L-1)/∂z(L-1) * ∂z(L-1)/∂w(L-1)
				// Remember that genderFullyConnectedLayer3[j].partialDerivative stores ∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1) in the summation part of the formula
				genderStochasticPartialDerivativeWeightMatrix2[i][h] += -1 * genderFullyConnectedLayer3[i].partialDerivative * derivativeSigmoid(genderFullyConnectedLayer3[i].unsquishedActivation) * genderFullyConnectedLayer2[h].activation;
			}
		}
	}
	
	public static void deriveBiasVector2() {
	// Calculate ∂C/∂w(L-1) for all weights in the second weight layer
	// This double for loop will traverse the columns first before moving to the next row in the weight matrix (the indices might look weird because I'm following my notes)
	for(int i = 0; i < genderFullyConnectedLayer3.length; i ++) {
			// Formula: ∂C/∂b(L-1) = Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) * ∂a(L-1)/∂z(L-1) * ∂z(L-1)/∂b(L-1)
			// Remember that genderFullyConnectedLayer3[j].partialDerivative stores ∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1) in the summation part of the formula
			genderStochasticPartialDerivativeBiasVector2[i] += -1 * genderFullyConnectedLayer3[i].partialDerivative * derivativeSigmoid(genderFullyConnectedLayer3[i].unsquishedActivation) * 1;
		}
	}
	
	public static void deriveGenderFullyConnectedLayer2Activations() {
		// Reset the partial derivatives first
		clearPartialDerivativesToActivations(genderFullyConnectedLayer2);
		
		// Calculate ∂C/∂a(L-2) for all activation nodes in fully connected layer 2 (this is just to make calculations easier for later weights, specifically the Σ(Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) * ∂a(L-1)/∂z(L-1) * ∂z(L-1)/∂a(L-2)) part in deriveGenderWeightMatrix1())
		// This double for loop will traverse the rows first before moving to the next column in the weight matrix (sum derivatives of each node)
		for(int h = 0; h < genderFullyConnectedLayer2.length; h ++) {
			for(int i = 0; i < genderFullyConnectedLayer3.length; i ++) {
				// Formula: ∂C/∂a(L-2) = Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) * ∂a(L-1)/∂z(L-1) * ∂z(L-1)/∂a(L-2)
				// Don't compute negative gradient for this
				genderFullyConnectedLayer2[h].partialDerivative = genderFullyConnectedLayer3[i].partialDerivative * derivativeSigmoid(genderFullyConnectedLayer3[i].unsquishedActivation) * genderWeightMatrix2[i][h];
			}
		}
	}
	
	public static void deriveGenderWeightMatrix1() {
		// Calculate ∂C/∂w(L-1) for all weights in the second weight layer
		// This double for loop will traverse the columns first before moving to the next row in the weight matrix (the indices might look weird because I'm following my notes)
		for(int h = 0; h < genderFullyConnectedLayer2.length; h ++) {
			for(int g = 0; g < genderFullyConnectedLayer1.length; g ++) {
				// Formula: ∂C/∂w(L-2) = Σ(Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) * ∂a(L-1)/∂z(L-1) * ∂z(L-1)/∂a(L-2)) * ∂a(L-2)/∂z(L-2) * ∂z(L-2)/∂w(L-2)
				// Remember that genderFullyConnectedLayer2[j].partialDerivative stores Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) * ∂a(L-1)/∂z(L-1) * ∂z(L-1)/∂a(L-2) in the summation part of the formula
				genderStochasticPartialDerivativeWeightMatrix1[h][g] += -1 * genderFullyConnectedLayer2[h].partialDerivative * derivativeSigmoid(genderFullyConnectedLayer2[h].unsquishedActivation) * genderFullyConnectedLayer1[g].activation;
			}
		}
	}
	
	public static void deriveBiasVector1() {
	// Calculate ∂C/∂w(L-1) for all weights in the second weight layer
	// This double for loop will traverse the columns first before moving to the next row in the weight matrix (the indices might look weird because I'm following my notes)
	for(int h = 0; h < genderFullyConnectedLayer2.length; h ++) {
			// Formula: ∂C/∂b(L-2) = Σ(Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) * ∂a(L-1)/∂z(L-1) * ∂z(L-1)/∂a(L-2)) * ∂a(L-2)/∂z(L-2) * ∂z(L-2)/∂b(L-2)
			// Remember that genderFullyConnectedLayer2[j].partialDerivative stores Σ(∂C/∂a(L) * ∂a(L)/∂z(L) * ∂z(L)/∂a(L-1)) * ∂a(L-1)/∂z(L-1) * ∂z(L-1)/∂a(L-2) in the summation part of the formula
			genderStochasticPartialDerivativeBiasVector1[h] += -1 * genderFullyConnectedLayer2[h].partialDerivative * derivativeSigmoid(genderFullyConnectedLayer2[h].unsquishedActivation) * 1;
		}
	}
	
	public static void deriveGenderFullyConnectedLayer1Activations() {
		// Reset the partial derivatives first
		clearPartialDerivativesToActivations(genderFullyConnectedLayer1);
		
		// Calculate ∂C/∂a(L-3) for all activation nodes in fully connected layer 1
		for(int g = 0; g < genderFullyConnectedLayer1.length; g ++) {
			for(int h = 0; h < genderFullyConnectedLayer2.length; h ++) {
				// No need for a formula because it's the same pattern over and over again
				// Note that vectorization copies the reference, so we don't need to copy all of these partial derivatives back to the last hidden convolution layer
				// Don't compute negative gradient for this
				genderFullyConnectedLayer1[g].partialDerivative = genderFullyConnectedLayer2[h].partialDerivative * derivativeSigmoid(genderFullyConnectedLayer2[h].unsquishedActivation) * genderWeightMatrix1[h][g];
			}
		}
	}
	
	public static void deriveConvolutionKernelBiasAndActivation(Convolution [] inputConvolutionLayer, Convolution [] outputConvolutionLayer, boolean deriveActivation) {
		// Reset the partial derivatives first
		clearConvolutionPartialDerivativesToActivations(inputConvolutionLayer);
		
		// This function will derive the cost with respect to the weights in the kernels of the input convolution layer
    	int splitSize = inputConvolutionLayer[0].kernelMatrices.length; // Can use 0 because all of them are the same length
    	
    	for(int i = 0; i < inputConvolutionLayer.length; i ++) {
        	for(int j = 0; j < splitSize; j ++) {
        		for(int k = 0; k < outputConvolutionLayer[splitSize * i + j].nodeMatrix.length; k ++) {
	            	for(int l = 0; l < outputConvolutionLayer[splitSize * i + j].nodeMatrix[k].length; l ++) {
        				// Get the ∂C/∂a value
        				float partialDerivativeActivation = outputConvolutionLayer[splitSize * i + j].nodeMatrix[k][l].partialDerivative;
        				
			            for(int m = 0; m < inputConvolutionLayer[i].kernelMatrices[j].length; m ++) {
				            for(int n = 0; n < inputConvolutionLayer[i].kernelMatrices[j][m].length; n ++) {
        						// Note that we're using the stochastic partial derivative kernel immediately. If you sum everything in the next 50 valid images and divide all of these millions of values by 50, it's still mathematically the same as finding each partial derivative before summing them up
        						inputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices[j][m][n] += -1 * partialDerivativeActivation * derivativeSigmoid(outputConvolutionLayer[splitSize * i + j].nodeMatrix[k][l].unsquishedActivation) * inputConvolutionLayer[i].nodeMatrix[k + m][l + n].activation;

        						// The only time we don't derive ∂C/∂a is if this convolution layer is the first (input) convolution layer of the neural network
        						if(deriveActivation) {
	        						// While we're here, might as well just calculate ∂C/∂a as well (This variable isn't used for stochastic gradient descent, so we can change it every time we calculate these derivatives for each image)
        							// Don't compute negative gradient for this
	        						inputConvolutionLayer[i].nodeMatrix[k + m][l + n].partialDerivative += partialDerivativeActivation * derivativeSigmoid(outputConvolutionLayer[splitSize * i + j].nodeMatrix[k][l].unsquishedActivation) * inputConvolutionLayer[i].kernelMatrices[j][m][n];
        						}
				            }
			            }
        				
        				// While we're here, might as well just calculate ∂C/∂b as well
        				outputConvolutionLayer[splitSize * i + j].stochasticPartialDerivativeBias += -1 * partialDerivativeActivation * derivativeSigmoid(outputConvolutionLayer[splitSize * i + j].nodeMatrix[k][l].unsquishedActivation) * 1;
	            	}
        		}
        	}
    	}
	}
	
	public static void clearPartialDerivativesToActivations(Node [] layer) {
		// Set all of the partial derivatives with respect to the activations to 0 for resetting
		for(int i = 0; i < layer.length; i ++) {
			layer[i].partialDerivative = 0;
		}
	}
	
	public static void clearConvolutionPartialDerivativesToActivations(Convolution [] convolution) {
		// Set all of the partial derivatives with respect to the activations to 0 for resetting
		for(int i = 0; i < convolution.length; i ++) {
			for(int j = 0; j < convolution[i].nodeMatrix.length; j ++) {
				for(int k = 0; k < convolution[i].nodeMatrix[j].length; k ++) {
					convolution[i].nodeMatrix[j][k].partialDerivative = 0;
				}
			}
		}
	}
	
	public static boolean checkFiftyGenderImages() {
		// Checks if we traversed 50 images
		if(validGenderCount % MOD_BY_FIFTY == 0) {
			return true;
		}
		
		return false;
	}	
	
	public static boolean checkFiftyAgeImages() {
		// Checks if we traversed 50 images
		if(validAgeCount % MOD_BY_FIFTY == 0) {
			return true;
		}
		
		return false;
	}	
	
	public static boolean checkFirstGenderImage() {
		// Checks if we hit the first image in an image group
		if(validGenderCount % MOD_BY_FIFTY == 1) {
			return true;
		}
		
		return false;
	}
	
	public static void adjustGenderWeightsAndBiases() {
		// Use partial derivatives to change the weights and biases (For this project, I'll adjust the weights and biases by moving them by GRADIENT_COMPONENT_UNITS unit(s), which is simply changing the weight and bias by the value of the partial derivative)
		// Edit the convolution kernel in input layer
		for(int i = 0; i < genderInputConvolutionLayer.length; i ++) {
			for(int j = 0; j < genderInputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices.length; j ++) {
				for(int k = 0; k < genderInputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices[j].length; k ++) {
					for(int l = 0; l < genderInputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices[j][k].length; l ++) {
						genderInputConvolutionLayer[i].kernelMatrices[j][k][l] += genderInputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices[j][k][l] / AVERAGE_STOCHASTIC_DERIVATIVES * GRADIENT_COMPONENT_UNITS;
						
						// Keep the value within -1 and 1
						genderInputConvolutionLayer[i].kernelMatrices[j][k][l] = boundValue(genderInputConvolutionLayer[i].kernelMatrices[j][k][l]);
					}
				}
			}
		}

		// Edit the convolution kernel in hidden layer 1
		for(int i = 0; i < genderHiddenConvolutionLayer1.length; i ++) {
			for(int j = 0; j < genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeKernelMatrices.length; j ++) {
				for(int k = 0; k < genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeKernelMatrices[j].length; k ++) {
					for(int l = 0; l < genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeKernelMatrices[j][k].length; l ++) {
						genderHiddenConvolutionLayer1[i].kernelMatrices[j][k][l] += genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeKernelMatrices[j][k][l] / AVERAGE_STOCHASTIC_DERIVATIVES * GRADIENT_COMPONENT_UNITS;
						
						// Keep the value within -1 and 1
						genderHiddenConvolutionLayer1[i].kernelMatrices[j][k][l] = boundValue(genderHiddenConvolutionLayer1[i].kernelMatrices[j][k][l]);
					}
				}
			}
		}

		// Edit the convolution kernel in hidden layer 2
		for(int i = 0; i < genderHiddenConvolutionLayer2.length; i ++) {
			for(int j = 0; j < genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeKernelMatrices.length; j ++) {
				for(int k = 0; k < genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeKernelMatrices[j].length; k ++) {
					for(int l = 0; l < genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeKernelMatrices[j][k].length; l ++) {
						genderHiddenConvolutionLayer2[i].kernelMatrices[j][k][l] += genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeKernelMatrices[j][k][l] / AVERAGE_STOCHASTIC_DERIVATIVES * GRADIENT_COMPONENT_UNITS;
						
						// Keep the value within -1 and 1
						genderHiddenConvolutionLayer2[i].kernelMatrices[j][k][l] = boundValue(genderHiddenConvolutionLayer2[i].kernelMatrices[j][k][l]);
					}
				}
			}
		}

		// Edit the convolution bias in input layer
		for(int i = 0; i < genderInputConvolutionLayer.length; i ++) {
			genderInputConvolutionLayer[i].bias += genderInputConvolutionLayer[i].stochasticPartialDerivativeBias / AVERAGE_STOCHASTIC_DERIVATIVES * GRADIENT_COMPONENT_UNITS;
			
			genderInputConvolutionLayer[i].bias = boundValue(genderInputConvolutionLayer[i].bias);
		}

		// Edit the convolution bias in hidden layer 1
		for(int i = 0; i < genderHiddenConvolutionLayer1.length; i ++) {
			genderHiddenConvolutionLayer1[i].bias += genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeBias / AVERAGE_STOCHASTIC_DERIVATIVES * GRADIENT_COMPONENT_UNITS;
			
			genderHiddenConvolutionLayer1[i].bias = boundValue(genderHiddenConvolutionLayer1[i].bias);
		}

		// Edit the convolution bias in hidden layer 2
		for(int i = 0; i < genderHiddenConvolutionLayer2.length; i ++) {
			genderHiddenConvolutionLayer2[i].bias += genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeBias / AVERAGE_STOCHASTIC_DERIVATIVES * GRADIENT_COMPONENT_UNITS;
			
			genderHiddenConvolutionLayer2[i].bias = boundValue(genderHiddenConvolutionLayer2[i].bias);
		}
		
		// Edit the first weight matrix
		for(int i = 0; i < genderWeightMatrix1.length; i ++) {
			for(int j = 0; j < genderWeightMatrix1[0].length; j ++) {
				genderWeightMatrix1[i][j] += (genderStochasticPartialDerivativeWeightMatrix1[i][j] / AVERAGE_STOCHASTIC_DERIVATIVES) * GRADIENT_COMPONENT_UNITS;
				
				// Keep the value within -1 and 1
				genderWeightMatrix1[i][j] = boundValue(genderWeightMatrix1[i][j]);
			}
		}

		// Edit the second weight matrix
		for(int i = 0; i < genderWeightMatrix2.length; i ++) {
			for(int j = 0; j < genderWeightMatrix2[0].length; j ++) {
				genderWeightMatrix2[i][j] += (genderStochasticPartialDerivativeWeightMatrix2[i][j] / AVERAGE_STOCHASTIC_DERIVATIVES) * GRADIENT_COMPONENT_UNITS;
				
				// Keep the value within -1 and 1
				genderWeightMatrix2[i][j] = boundValue(genderWeightMatrix2[i][j]);
			}
		}
		
		// Edit the third weight matrix
		for(int i = 0; i < genderWeightMatrix3.length; i ++) {
			for(int j = 0; j < genderWeightMatrix3[0].length; j ++) {
				genderWeightMatrix3[i][j] += (genderStochasticPartialDerivativeWeightMatrix3[i][j] / AVERAGE_STOCHASTIC_DERIVATIVES) * GRADIENT_COMPONENT_UNITS;
				
				// Keep the value within -1 and 1
				genderWeightMatrix3[i][j] = boundValue(genderWeightMatrix3[i][j]);
			}
		}
		
		// Edit the first bias vector
		for(int i = 0; i < genderBiasVector1.length; i ++) {
			genderBiasVector1[i] += (genderStochasticPartialDerivativeBiasVector1[i] / AVERAGE_STOCHASTIC_DERIVATIVES) * GRADIENT_COMPONENT_UNITS;
				
			// Keep the value within -1 and 1
			genderBiasVector1[i] = boundValue(genderBiasVector1[i]);
		}
		
		// Edit the second bias vector
		for(int i = 0; i < genderBiasVector2.length; i ++) {
			genderBiasVector2[i] += (genderStochasticPartialDerivativeBiasVector2[i] / AVERAGE_STOCHASTIC_DERIVATIVES) * GRADIENT_COMPONENT_UNITS;
				
			// Keep the value within -1 and 1
			genderBiasVector2[i] = boundValue(genderBiasVector2[i]);
		}
		
		// Edit the third bias vector
		for(int i = 0; i < genderBiasVector3.length; i ++) {
			genderBiasVector3[i] += (genderStochasticPartialDerivativeBiasVector3[i] / AVERAGE_STOCHASTIC_DERIVATIVES) * GRADIENT_COMPONENT_UNITS;
				
			// Keep the value within -1 and 1
			genderBiasVector3[i] = boundValue(genderBiasVector3[i]);
		}
	}
	
	public static float boundValue(float value) {
		// Check if this feature is enabled
		if(!disableWeightRestriction) {
			// Don't let the value go above 1
			if(value > 1) {
				return 1;
			}
			else if(value < -1) {
				return -1;
			}
			else {
				return value;
			}
		}
		else {
			// If disabled, then just return the value
			return value;
		}
	}
	
	public static void clearStochasticWeightsAndBiases() {
		// Clears the stochastic weight matrices and biases for the next 50 images
		// Change Convolution kernel in input layer
		for(int i = 0; i < genderInputConvolutionLayer.length; i ++) {
			for(int j = 0; j < genderInputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices.length; j ++) {
				for(int k = 0; k < genderInputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices[j].length; k ++) {
					for(int l = 0; l < genderInputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices[j][k].length; l ++) {
						genderInputConvolutionLayer[i].stochasticPartialDerivativeKernelMatrices[j][k][l] = 0;
					}
				}
			}
		}
		
		// Change Convolution bias in input layer
		for(int i = 0; i < genderInputConvolutionLayer.length; i ++) {
			genderInputConvolutionLayer[i].stochasticPartialDerivativeBias = 0;
		}
		
		// Change Convolution kernel in hidden layer 1
		for(int i = 0; i < genderHiddenConvolutionLayer1.length; i ++) {
			for(int j = 0; j < genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeKernelMatrices.length; j ++) {
				for(int k = 0; k < genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeKernelMatrices[j].length; k ++) {
					for(int l = 0; l < genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeKernelMatrices[j][k].length; l ++) {
						genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeKernelMatrices[j][k][l] = 0;
					}
				}
			}
		}
		
		// Change Convolution bias in hidden layer 1
		for(int i = 0; i < genderHiddenConvolutionLayer1.length; i ++) {
			genderHiddenConvolutionLayer1[i].stochasticPartialDerivativeBias = 0;
		}
		
		// Change Convolution kernel in hidden layer 2
		for(int i = 0; i < genderHiddenConvolutionLayer2.length; i ++) {
			for(int j = 0; j < genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeKernelMatrices.length; j ++) {
				for(int k = 0; k < genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeKernelMatrices[j].length; k ++) {
					for(int l = 0; l < genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeKernelMatrices[j][k].length; l ++) {
						genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeKernelMatrices[j][k][l] = 0;
					}
				}
			}
		}
		
		// Change Convolution bias in hidden layer 2
		for(int i = 0; i < genderHiddenConvolutionLayer2.length; i ++) {
			genderHiddenConvolutionLayer2[i].stochasticPartialDerivativeBias = 0;
		}
		
		// Change weight matrix 1
		for(int i = 0; i < genderStochasticPartialDerivativeWeightMatrix1.length; i ++) {
			for(int j = 0; j < genderStochasticPartialDerivativeWeightMatrix1[0].length; j ++) {
				genderStochasticPartialDerivativeWeightMatrix1[i][j] = 0;
			}
		}
		
		// Change bias vector 1
		for(int i = 0; i < genderStochasticPartialDerivativeBiasVector1.length; i ++) {
			genderStochasticPartialDerivativeBiasVector1[i] = 0;
		}
		
		// Change weight matrix 2
		for(int i = 0; i < genderStochasticPartialDerivativeWeightMatrix2.length; i ++) {
			for(int j = 0; j < genderStochasticPartialDerivativeWeightMatrix2[0].length; j ++) {
				genderStochasticPartialDerivativeWeightMatrix2[i][j] = 0;
			}
		}
		
		// Change bias vector 2
		for(int i = 0; i < genderStochasticPartialDerivativeBiasVector2.length; i ++) {
			genderStochasticPartialDerivativeBiasVector2[i] = 0;
		}
		
		// Change weight matrix 3
		for(int i = 0; i < genderStochasticPartialDerivativeWeightMatrix3.length; i ++) {
			for(int j = 0; j < genderStochasticPartialDerivativeWeightMatrix3[0].length; j ++) {
				genderStochasticPartialDerivativeWeightMatrix3[i][j] = 0;
			}
		}
		
		// Change bias vector 3
		for(int i = 0; i < genderStochasticPartialDerivativeBiasVector3.length; i ++) {
			genderStochasticPartialDerivativeBiasVector3[i] = 0;
		}
	}
	
	public static boolean checkGenderCompleted() {
		// We are done if we have less than 10% of the images remaining
		if(genderCount >= genderCutoffImageValidCount) {
			inTraining = false; 
			
			return true;
		}
		
		return false;
	}
	
	public static boolean checkTestingCompleted() {
		// Testing is completed if we traversed through all images
		if(Math.min(femaleData.size(), maleData.size()) <= 0) {
			inTesting = false; 
			
			return true;
		}
		
		return false;
	}
	
	public static void guessedCorrect() {
		// Checks if the program guessed the value correctly and add 1 to the numGenderCorrect loadingCount if correct (used during accuracy test)
		if(!genderInvalid) {
			if((guessedGender == currentGender)) {
				numGenderCorrect ++;
			}
			
			validGenderCount ++;
		}
	}	
	
	public static void clearCosts() {
		// Clears both genderCost functions in preparation for accuracy testing
		averageGenderCost = 0;
		genderCost = 0;
	}
	
	public static void findMaxKernelWeightMagnitude() {
		// If we restrict weights, then the max is just 1
		if(!disableWeightRestriction) {
			largestGenderKernelWeightMagnitude = 1;
		}
		else {
			// We need to find it
			// Check input layer
			findLargestKernelWeight(genderInputConvolutionLayer);
			
			// Check hidden layer 1
			findLargestKernelWeight(genderHiddenConvolutionLayer1);
		}
	}
	
	public static void findLargestKernelWeight(Convolution [] convolutionToCheck) {
		for(int i = 0; i < convolutionToCheck.length; i ++) {
			for(int j = 0; j < convolutionToCheck[i].kernelMatrices.length; j ++) {
				for(int k = 0; k < convolutionToCheck[i].kernelMatrices[j].length; k ++) {
					for(int l = 0; l < convolutionToCheck[i].kernelMatrices[j][k].length; l ++) {
						if(Math.abs(convolutionToCheck[i].kernelMatrices[j][k][l]) > largestGenderKernelWeightMagnitude) {
							largestGenderKernelWeightMagnitude = (int) Math.abs(convolutionToCheck[i].kernelMatrices[j][k][l]);
						}
					}
				}
			}
		}
	}
	
	public static void writeFile() throws IOException {
		// Writes the network and statistics file
		System.out.println("Writing file... Please wait.");
		
		// Get the current date and time
		DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd_HH.mm.ss");
		LocalDateTime date = LocalDateTime.now();
		
		// Name this file
		networkFile = new BufferedWriter(new FileWriter("neuralNetwork_" + formatter.format(date) + ".txt"));
		
		// Write the following:
		// Total wiki images
		networkFile.write(numWikiImages + "\n");
		
		// Total imdb images
		networkFile.write(numImdbImages + "\n");
		
		// Total images
		networkFile.write(totalImages + "\n");
		
		// Total usable gender images
		networkFile.write(totalValidGenderCount + "\n");
		
		// Total usable age images
		networkFile.write(totalValidGenderCount + "\n");
		
		// Total gender images used for testing
		networkFile.write(validGenderCount + "\n");
		
		// Number of images guessed correctly
		networkFile.write(numGenderCorrect + "\n");
		
		// Accuracy
		String accuracy = ((PaintPanel.CONVERT_TO_PERCENTAGE * Main.numGenderCorrect / Main.validGenderCount) * PaintPanel.ROUNDING) / PaintPanel.ROUNDING + "";
		networkFile.write(accuracy + "\n");
		
		// Input convolution size
		networkFile.write(genderInputConvolutionLayer[0].nodeMatrix.length + "\n");
		
		// Input convolution kernel count
		networkFile.write(genderInputConvolutionLayer[0].kernelMatrices.length + "\n");
		
		// Input convolution kernel size
		networkFile.write(genderInputConvolutionLayer[0].kernelMatrices[0].length + "\n");
		
		// Hidden convolution layer 1 size
		networkFile.write(genderHiddenConvolutionLayer1[0].nodeMatrix.length + "\n");
		
		// Hidden convolution layer 1 kernel count
		networkFile.write(genderHiddenConvolutionLayer1[0].kernelMatrices.length + "\n");
		
		// Hidden convolution layer 2 kernel size
		networkFile.write(genderHiddenConvolutionLayer1[0].kernelMatrices[0].length + "\n");
		
		// Hidden convolution layer 2 size
		networkFile.write(genderHiddenConvolutionLayer2[0].nodeMatrix.length + "\n");
		
		// Hidden convolution layer 2 kernel count
		networkFile.write(genderHiddenConvolutionLayer2[0].kernelMatrices.length + "\n");
		
		// Hidden convolution layer 2 kernel size
		networkFile.write(0 + "\n");
		
		// Input node loadingCount
		networkFile.write(vectorizationLayerSize + "\n");

		// fully connected layer 2 node loadingCount
		networkFile.write(FULLY_CONNECTED_HIDDEN_LAYER_SIZE + "\n");

		// fully connected layer 3 node loadingCount
		networkFile.write(FULLY_CONNECTED_HIDDEN_LAYER_SIZE + "\n");

		// Output layer node loadingCount
		networkFile.write(GENDER_OUTPUT_LAYER_SIZE + "\n");
		
		// Kernel matrices in input convolution layer (use strings since we can't write floats)
		String weights = "";
		for(int i = 0; i < genderInputConvolutionLayer.length; i ++) {
			for(int j = 0; j < genderInputConvolutionLayer[i].kernelMatrices.length; j ++) {
				for(int k = 0; k < genderInputConvolutionLayer[i].kernelMatrices[j].length; k ++) {
					weights = "";
					for(int l = 0; l < genderInputConvolutionLayer[i].kernelMatrices[j][k].length; l ++) {
						weights += genderInputConvolutionLayer[i].kernelMatrices[j][k][l];
						
						// Don't add space after the last value
						if(l < genderInputConvolutionLayer[i].kernelMatrices[j][k].length - 1) {
							weights += " ";
						}
					}
					networkFile.write(weights + "\n");
				}
			}
		}
		
		// Kernel matrices in hidden convolution layer 1
		for(int i = 0; i < genderHiddenConvolutionLayer1.length; i ++) {
			for(int j = 0; j < genderHiddenConvolutionLayer1[i].kernelMatrices.length; j ++) {
				for(int k = 0; k < genderHiddenConvolutionLayer1[i].kernelMatrices[j].length; k ++) {
					weights = "";
					for(int l = 0; l < genderHiddenConvolutionLayer1[i].kernelMatrices[j][k].length; l ++) {
						weights += genderHiddenConvolutionLayer1[i].kernelMatrices[j][k][l];
						
						// Don't add space after the last value
						if(l < genderHiddenConvolutionLayer1[i].kernelMatrices[j][k].length - 1) {
							weights += " ";
						}
					}
					networkFile.write(weights + "\n");
				}
			}
		}
		
		// Bias values in hidden conovlution layer 1
		for(int i = 0; i < genderHiddenConvolutionLayer1.length; i ++) {
			networkFile.write(genderHiddenConvolutionLayer1[i].bias + "\n");
		}
		
		// Bias values in hidden conovlution layer 2
		for(int i = 0; i < genderHiddenConvolutionLayer2.length; i ++) {
			networkFile.write(genderHiddenConvolutionLayer2[i].bias + "\n");
		}
		
		// Weight matrix 1 values
		for(int r = 0; r < genderWeightMatrix1.length; r ++) {
			weights = "";
			for(int c = 0; c < genderWeightMatrix1.length; c ++) {
				weights += genderWeightMatrix1[r][c];
				
				// Don't add space after the last value
				if(c < genderWeightMatrix1.length - 1) {
					weights += " ";
				}
			}
			networkFile.write(weights + "\n");
		}
		
		// Weight matrix 2 values
		for(int r = 0; r < genderWeightMatrix2.length; r ++) {
			weights = "";
			for(int c = 0; c < genderWeightMatrix2.length; c ++) {
				weights += genderWeightMatrix2[r][c];
				
				// Don't add space after the last value
				if(c < genderWeightMatrix2.length - 1) {
					weights += " ";
				}
			}
			networkFile.write(weights + "\n");
		}
		
		// Weight matrix 3 values
		for(int r = 0; r < genderWeightMatrix3.length; r ++) {
			weights = "";
			for(int c = 0; c < genderWeightMatrix3.length; c ++) {
				weights += genderWeightMatrix3[r][c];
				
				// Don't add space after the last value
				if(c < genderWeightMatrix3.length - 1) {
					weights += " ";
				}
			}
			networkFile.write(weights + "\n");
		}
		
		// Bias vector 1 values
		for(int r = 0; r < genderBiasVector1.length; r ++) {
			networkFile.write(genderBiasVector1[r] + "\n");
		}
		
		// Bias vector 2 values
		for(int r = 0; r < genderBiasVector2.length; r ++) {
			networkFile.write(genderBiasVector2[r] + "\n");
		}
		
		// Bias vector 3 values
		for(int r = 0; r < genderBiasVector3.length; r ++) {
			networkFile.write(genderBiasVector3[r] + "\n");
		}
		
		// Close and complete the buffered writer
		networkFile.close();
		
		// Conclude the program
		System.out.println("File written. Thank you for using this neural network trainer!");
	}
	
	public static BufferedImage resizeImage(BufferedImage current, int width, int height) {
        BufferedImage resizedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        final Graphics2D g2D = resizedImage.createGraphics();
        g2D.setComposite(AlphaComposite.Src);
        
        // Use RenderingHints for better quality
        g2D.setRenderingHint(RenderingHints.KEY_INTERPOLATION,RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2D.setRenderingHint(RenderingHints.KEY_RENDERING,RenderingHints.VALUE_RENDER_QUALITY);
        g2D.setRenderingHint(RenderingHints.KEY_ANTIALIASING,RenderingHints.VALUE_ANTIALIAS_ON);
        
        // Create the new image
        g2D.drawImage(current, 0, 0, width, height, null);
        g2D.dispose();
        
        return resizedImage;
	}
	
	public static void loadFiles() throws InterruptedException {
		// Startup sequence for loading up files
		// Check to make sure the directories are there
		File metaDirectory = new File("metaData");
		if(!metaDirectory.exists()) {
			System.out.println("ERROR: Directory \"meta data\" is missing. Closing program...");
			Thread.sleep(5000);
			System.exit(0);
		}
		loadingCount ++;
		paint.repaint();

		File imagesDirectory = new File("images");
		if(!imagesDirectory.exists()) {
			System.out.println("ERROR: Directory \"images\" is missing. Closing program...");
			Thread.sleep(5000);
			System.exit(0);
		}
		loadingCount ++;
		paint.repaint();
		
		// Check to make sure all meta data files are there
		try {
			fisImdbDOB = new FileInputStream("metaData/imdb_data_dob.csv");
		} catch (FileNotFoundException e) {
			System.out.println("ERROR: File \"imdb_data_dob.csv\" is missing. Closing program...");
			Thread.sleep(5000);
			System.exit(0);
		}
		disImdbDOB = new DataInputStream(fisImdbDOB);
		loadingCount ++;
		paint.repaint();

		try {
			fisImdbOther = new FileInputStream("metaData/imdb_data_photo_taken_gender_path.csv");
		} catch (FileNotFoundException e) {
			System.out.println("ERROR: File \"imdb_data_photo_taken_gender_path.csv\" is missing. Closing program...");
			Thread.sleep(5000);
			System.exit(0);
		}
		disImdbOther = new DataInputStream(fisImdbOther);
		loadingCount ++;
		paint.repaint();
		
		try {
			fisWikiDOB = new FileInputStream("metaData/wiki_data_dob.csv");
		} catch (FileNotFoundException e) {
			System.out.println("ERROR: File \"wiki_data_dob.csv\" is missing. Closing program...");
			Thread.sleep(5000);
			System.exit(0);
		}
		disWikiDOB = new DataInputStream(fisWikiDOB);
		loadingCount ++;
		paint.repaint();

		try {
			fisWikiOther = new FileInputStream("metaData/wiki_data_photo_taken_gender_path.csv");
		} catch (FileNotFoundException e) {
			System.out.println("ERROR: File \"wiki_data_photo_taken_gender_path.csv\" is missing. Closing program...");
			Thread.sleep(5000);
			System.exit(0);
		}
		disWikiOther = new DataInputStream(fisWikiOther);
		loadingCount ++;
		paint.repaint();
		
		// Check to make sure the crop directories are there
		File wikiCropDirectory = new File("images/wiki_crop");
		if(!wikiCropDirectory.exists()) {
			Thread.sleep(5000);
			System.exit(0);
		}
		loadingCount ++;
		paint.repaint();
		
		File imdbCropDirectory = new File("images/imdb_crop");
		if(!imdbCropDirectory.exists()) {
			System.out.println("ERROR: Directory \"imdb_crop\" is missing. Closing program...");
			Thread.sleep(5000);
			System.exit(0);
		}
		loadingCount ++;
		paint.repaint();
		
		// If we got here, then we have all of the files. Now loadingCount how many images there are
		File [] wikiImageFolders = wikiCropDirectory.listFiles();
		for(int i = 0; i < wikiImageFolders.length; i ++) {
			numWikiImages += wikiImageFolders[i].listFiles().length;
			loadingCount ++;
			paint.repaint();
		}
		
		File [] imdbImageFolders = imdbCropDirectory.listFiles();
		for(int i = 0; i < imdbImageFolders.length; i ++) {
			numImdbImages += imdbImageFolders[i].listFiles().length;
			loadingCount ++;
			paint.repaint();
		}
		
		totalImages = numImdbImages + numWikiImages;
		loadingCount ++;
		paint.repaint();

		// Create the buffered readers for each file
		brImdbDOB = new BufferedReader(new InputStreamReader(disImdbDOB));
		brImdbOther = new BufferedReader(new InputStreamReader(disImdbOther));
		brWikiDOB = new BufferedReader(new InputStreamReader(disWikiDOB));
		brWikiOther = new BufferedReader(new InputStreamReader(disWikiOther));
		loadingCount += 4;
		paint.repaint();
	}
	
	public static void loadGenderNetwork() throws IOException {
		// Create the data arraylists
		ageData = new ArrayList<Age_Person>();
		femaleData = new ArrayList<String>();
		maleData = new ArrayList<String>();
		loadingCount += 3;
		paint.repaint();
		
		// Set the counts
		totalValidGenderCount = 0;
		totalValidAgeCount = 0;
		
		// Create the convolution layers
		int decreaseSize = KERNEL_SIZE - 1; // Since the kernels create a smaller convolution in the next layer, each convolution's dimensions decrease by kernel - 1
		int convolutionSize = INPUT_CONVOLUTION_SIZE; // The size of the convolution in the current layer we're looking at
		
		genderInputConvolutionLayer = new Convolution[INPUT_CONVOLUTION_LAYER_SIZE];
		for(int i = 0; i < genderInputConvolutionLayer.length; i ++) {
			genderInputConvolutionLayer[i] = new Convolution(convolutionSize, FIRST_HIDDEN_CONVOLUTION_LAYER_SIZE / INPUT_CONVOLUTION_LAYER_SIZE, KERNEL_SIZE);
			
			// Bias is unused in this layer
			genderInputConvolutionLayer[i].bias = 0;
		}
		convolutionSize -= decreaseSize;
		
		genderHiddenConvolutionLayer1 = new Convolution[FIRST_HIDDEN_CONVOLUTION_LAYER_SIZE];
		for(int i = 0; i < genderHiddenConvolutionLayer1.length; i ++) {
			genderHiddenConvolutionLayer1[i] = new Convolution(convolutionSize, SECOND_HIDDEN_CONVOLUTION_LAYER_SIZE / FIRST_HIDDEN_CONVOLUTION_LAYER_SIZE, KERNEL_SIZE);
		}
		convolutionSize -= decreaseSize;
		
		genderHiddenConvolutionLayer2 = new Convolution[SECOND_HIDDEN_CONVOLUTION_LAYER_SIZE];
		for(int i = 0; i < genderHiddenConvolutionLayer2.length; i ++) {
			genderHiddenConvolutionLayer2[i] = new Convolution(convolutionSize, 0, KERNEL_SIZE);
		}
		vectorizationLayerSize = convolutionSize * convolutionSize * SECOND_HIDDEN_CONVOLUTION_LAYER_SIZE;
		
		// Create the fully connected layers
		genderFullyConnectedLayer1 = new Node[vectorizationLayerSize];
		for(int i = 0; i < genderFullyConnectedLayer1.length; i ++) {
			genderFullyConnectedLayer1[i] = new Node();
			loadingCount ++;
			paint.repaint();
		}
		
		genderFullyConnectedLayer2 = new Node[FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		for(int i = 0; i < genderFullyConnectedLayer2.length; i ++) {
			genderFullyConnectedLayer2[i] = new Node();
			loadingCount ++;
			paint.repaint();
		}
		
		genderFullyConnectedLayer3 = new Node[FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		for(int i = 0; i < genderFullyConnectedLayer3.length; i ++) {
			genderFullyConnectedLayer3[i] = new Node();
			loadingCount ++;
			paint.repaint();
		}
		
		genderOutputLayer = new Node[GENDER_OUTPUT_LAYER_SIZE];
		for(int i = 0; i < genderOutputLayer.length; i ++) {
			genderOutputLayer[i] = new Node();
			loadingCount ++;
			paint.repaint();
		}
		
		// Create genderDesiredOutput array
		genderDesiredOutput = new float[GENDER_OUTPUT_LAYER_SIZE];
		loadingCount ++;
		paint.repaint();

		// Create and randomize weight matrices (rows correspond to the node number at the end of the weight connection while columns correspond to the node number at the beginning of the weight connection
		genderWeightMatrix1 = new float[FULLY_CONNECTED_HIDDEN_LAYER_SIZE][vectorizationLayerSize];
		for(int i = 0; i < genderWeightMatrix1.length; i ++) {
			for(int j = 0; j < genderWeightMatrix1[i].length; j ++) {
				genderWeightMatrix1[i][j] = (float) ((rng.nextFloat() * 2) - 1);
				loadingCount ++;
				paint.repaint();
			}
		}
		
		genderWeightMatrix2 = new float[FULLY_CONNECTED_HIDDEN_LAYER_SIZE][FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		for(int i = 0; i < genderWeightMatrix2.length; i ++) {
			for(int j = 0; j < genderWeightMatrix2[i].length; j ++) {
				genderWeightMatrix2[i][j] = (float) ((rng.nextFloat() * 2) - 1);
				loadingCount ++;
				paint.repaint();
			}
		}
		
		genderWeightMatrix3 = new float[GENDER_OUTPUT_LAYER_SIZE][FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		for(int i = 0; i < genderWeightMatrix3.length; i ++) {
			for(int j = 0; j < genderWeightMatrix3[i].length; j ++) {
				genderWeightMatrix3[i][j] = (float) ((rng.nextFloat() * 2) - 1);
				loadingCount ++;
				paint.repaint();
			}
		}
		
		// Create and randomize the bias vectors with biases in the range -5 to 5
		genderBiasVector1 = new float[FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		for(int i = 0; i < genderBiasVector1.length; i ++) {
			genderBiasVector1[i] = rng.nextFloat() * 10 - 5;
			loadingCount ++;
			paint.repaint();
		}
		
		genderBiasVector2 = new float[FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		for(int i = 0; i < genderBiasVector2.length; i ++) {
			genderBiasVector2[i] = rng.nextFloat() * 10 - 5;
			loadingCount ++;
			paint.repaint();
		}
		
		genderBiasVector3 = new float[GENDER_OUTPUT_LAYER_SIZE];
		for(int i = 0; i < genderBiasVector3.length; i ++) {
			genderBiasVector3[i] = rng.nextFloat() * 10 - 5;
			loadingCount ++;
			paint.repaint();
		}	
		
		// For this project, we only have the stochastic gradient partial derivative matrices and vectors this time because we don't need to display that much debug
		// Initialize the stochastic weight matrices (what we actually use to adjust the weight values)
		genderStochasticPartialDerivativeWeightMatrix1 = new float[FULLY_CONNECTED_HIDDEN_LAYER_SIZE][vectorizationLayerSize];
		genderStochasticPartialDerivativeWeightMatrix2 = new float[FULLY_CONNECTED_HIDDEN_LAYER_SIZE][FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		genderStochasticPartialDerivativeWeightMatrix3 = new float[GENDER_OUTPUT_LAYER_SIZE][FULLY_CONNECTED_HIDDEN_LAYER_SIZE]; 
		loadingCount +=3;
		paint.repaint();
		
		// Initialize the stochastic bias vectors (what we actually use to adjust the bias values)
		genderStochasticPartialDerivativeBiasVector1 = new float[FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		genderStochasticPartialDerivativeBiasVector2 = new float[FULLY_CONNECTED_HIDDEN_LAYER_SIZE];
		genderStochasticPartialDerivativeBiasVector3 = new float[GENDER_OUTPUT_LAYER_SIZE];
		loadingCount +=3;
		paint.repaint();
		
		// Set the default max kernel weight magnitude
		largestGenderKernelWeightMagnitude = 1;
		// Reset checks
		ageInvalid = false;
		genderInvalid = false;
		
		// Load all of the data into the array lists
		String readLineSecondary;
		String readLineDob;
		
		while((readLineSecondary = brWikiOther.readLine()) != null && (readLineDob = brWikiDOB.readLine()) != null) {
			loadingCount += 2;
			paint.repaint();
			
			processData(readLineDob, readLineSecondary, "images/wiki_crop/");
		}

		while((readLineSecondary = brImdbOther.readLine()) != null && (readLineDob = brImdbDOB.readLine()) != null) {
			loadingCount += 2;
			paint.repaint();
			
			processData(readLineDob, readLineSecondary, "images/imdb_crop/");
		}

		// Create the gender neural network
		genderCutoffImageValidCount = (int) (Math.min(femaleData.size(), maleData.size()) * 2 * 0.9);
		
		// Set more counts
		validGenderCount = 0;
		validAgeCount = 0;
		loadingCount ++;
		paint.repaint();
	}
	
	public static void processData(String dob, String other, String directoryPath) {
		try {
			String [] tempOtherRead = other.split(",");
			
			// Safe to use image
			ageData.add(new Age_Person(dob, Integer.parseInt(tempOtherRead[YEAR_TAKEN_INDEX]), directoryPath + tempOtherRead[PATH_INDEX]));
			totalValidAgeCount ++;
						
			if(Integer.parseInt(tempOtherRead[GENDER_INDEX]) == 0) {
				femaleData.add(directoryPath + tempOtherRead[PATH_INDEX]);
				totalValidGenderCount ++;
			}
			else if(Integer.parseInt(tempOtherRead[GENDER_INDEX]) == 1) {
				// Use else if instead of else because sometimes the person could have -1 as their gender, which means unidentified
				maleData.add(directoryPath + tempOtherRead[PATH_INDEX]);
				totalValidGenderCount ++;
			}
		}
		catch(Exception e) {
			// Just skip that data
		}
	}
}
