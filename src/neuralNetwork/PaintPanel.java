package neuralNetwork;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.Graphics2D;
import javax.swing.JPanel;

@SuppressWarnings("serial")
public class PaintPanel extends JPanel {

	// Some constants
	public static final int NODE_SIZE = 30; // Node size
	public static final int TEXT_SPACE = 10; // Space between node and text
	public static final int INPUT_TEXT_SPACE = 17; // Space between input number and node
	public static final int WEIGHT_LABEL_SPACE = 140; // Space between each weight label
	public static final int COST_TEXT_SPACE = 27; // Space to lower the cost to the equal sign in the cost function image/node
	public static final int AVERAGE_COST_TEXT_SPACE = 20; // Space to lower the cost to the equal sign in the average cost function image/node
	public static final int TRAVERSED_TEXT_SPACE = 30; // Space to lower the cost to the equal sign in the cost function image
	public static final int COST_FUNCTION_SPACE = 80; // Space between both cost function images
	public static final int LABEL_SPACE = 35; // Space between the stacked labels (next to the last nine images)
	public static final float ROUNDING = 100000.0f; // Used to round numbers to the 5th decimal place
	public static final float CONVERT_TO_PERCENTAGE = 100.0f; // Used to convert to percentage format
	public static final int WEIGHT_SPACE_LEFT = 200; // Amount of space the input weight is written on the left side
	public static final int WEIGHT_SPACE_RIGHT = 200; // Amount of space the output weight is written on the right side
	public static final int FIFTY_IMAGES = 50;
	public static final int CUTOFF = 15; // Columns that are cut off on the right side of the image
	public static final int MAX_COLOR_VALUE = 255; // Max color value in RGB
	public static final int DOUBLE = 2; // Doubles a value
	public static final int HALF = 2; // Halves a value
	public static final int TRIPLE = 3; // Triples a value
	public static final int QUADRUPLE = 4; // Quadruples a value
	public static final int QUINTUPLE = 5; // Quintuples a value
	public static final int SEXTUPLE = 6; // Sextuples a value
	public static final int SEPTUPLE = 7; // Septuples a value
	public static final int OCTUPLE = 8; // Octuples a value
	public static final int NONUPLE = 9; // Nonuples a value
	public static final int DECUPLE = 10; // Decuples a value
	public static final int UNDECUPLE = 11; // Undecuples a value
	public static final int DUODECUPLE = 12; // Duodecuples a value
	public static final int TREDECUPLE = 13; // Tredecuples a value
	public static final int QUATTUORDECUPLE = 14; // Quattuordecuples a value
	public static final int QUINDECUPLE = 15; // Quindecuples a value
	public static final int SEXDECUPLE = 16; // Sexdecuples a value
	public static final int SEPTENDECUPLE = 17; // Septdendecuples a value
	public static final int OCTODECUPLE = 18; // Octodecuples a value
	
	public static final int CONVOLUTION_PIXEL_SIZE = 1; // Size of pixels drawn for convolution layers 
	public static final int KERNEL_PIXEL_SIZE = 16; // Size of pixels drawn for kernels
	public static final int CONVOLUTION_START_X = 650; // Starting x location of the input convolution layer
	public static final int CONVOLUTION_START_Y = 0; // Starting y location of all convolution layers
	public static final int CONVOLUTION_SPACE = 10; // Space between convolutions and kernels
	
	public static final int LOADING_TEXT_X = 840; // X position of the loading text
	public static final int LOADING_TEXT_Y = 460; // X position of the loading text
	
	public static final int LOADING_PERCENTAGE_TEXT_X = 10; // X position of the loading percentage text
	public static final int LOADING_PERCENTAGE_TEXT_Y = 1030; // X position of the loading percentage text
	
	public static final int LOADING_BAR_X = 480; // X position of the loading bar
	public static final int LOADING_BAR_Y = 500; // Y position of the loading bar
	public static final int LOADING_BAR_WIDTH = 960; // Width of the loading bar
	public static final int LOADING_BAR_HEIGHT = 40; // Height of the loading bar
	
	public static final int FONT_SIZE_THIRTY = 30; // Font size 30
	public static final int FONT_SIZE_SIXTY = 60; // Font size 60
	
	public static final int CURRENT_IMAGE_X = 1920 - (Main.INPUT_CONVOLUTION_SIZE * DOUBLE) - CUTOFF; // Starting X position of the current image
	public static final int CURRENT_IMAGE_Y = 0; // Starting Y position of the current image
	
	public static final int DEBUG_LABEL_START_X = 5; // Starting x position of debug information
	public static final int DEBUG_LABEL_START_Y = 28; // Starting y position of debug information
	public static final int FIRST_CONTROL_LABEL_X = DEBUG_LABEL_START_X; // Starting x position of the control labels
	public static final int FIRST_CONTROL_LABEL_Y = Main.SCREEN_HEIGHT - 50; // Starting x position of the control labels
	
	public static final int INPUT_LAYER = 0; // First convolution layer index
	public static final int HIDDEN_LAYER1 = 1; // First convolution layer index
	public static final int HIDDEN_LAYER2 = 2; // First convolution layer index
	public static final int HIDDEN_LAYER3 = 3; // First convolution layer index
	public static final int HIDDEN_LAYER4 = 4; // First convolution layer index
	public static final int HIDDEN_LAYER5 = 5; // First convolution layer index
	
	public static final int OUTPUT_START_X = 1160; // X position for HIDDEN1 nodes
	public static final int LAYER_SPACING = 170; // Space between layers
	public static final int HIDDEN1_START_Y = 40; // Starting Y value for the HIDDEN1 nodes
	public static final int HIDDEN1_END_Y = Main.SCREEN_HEIGHT - HIDDEN1_START_Y - DOUBLE * NODE_SIZE; // Ending Y value for the HIDDEN1 nodes
	public static final int HIDDEN_SPACING_Y = (int) Math.ceil((float) (HIDDEN1_START_Y + HIDDEN1_END_Y) / (Main.FULLY_CONNECTED_HIDDEN_LAYER_SIZE)); // Space between HIDDEN1 nodes
	public static final int OUTPUT_START_Y = 10; // Starting Y value for the HIDDEN1 nodes
	public static final int OUTPUT_SPACING = 50; // Space between HIDDEN1 nodes
	
	// Draws all of the objects on the panel (which is added to the JFrame)
	public void paint(Graphics g) {
		// Call the super class
		super.paintComponent(g);
		
		// Determine background setting
		if(Main.darkMode) {
			this.setBackground(Color.BLACK);
		}
		else {
			this.setBackground(Color.WHITE);
		}
		
		if(Main.loaded) {
			// Set the font
			g.setFont(new Font("TimesRoman", Font.PLAIN, FONT_SIZE_THIRTY)); 
			
			// Set color of text depending on whether dark mode is enabled
			if(Main.darkMode) {
				g.setColor(Color.WHITE);
			}
			else {
				g.setColor(Color.BLACK);
			}	
			
			// Draw debug information
			if(!Main.genderInvalid) {
			    g.drawString("Gender cost: c = " + Main.genderCost, DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y);
			    g.drawString("Average gender cost: c = " + Main.averageGenderCost / ((Main.validGenderCount - 1) % Main.MOD_BY_FIFTY + 1), DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE);
			}
			else {
			    g.drawString("Gender cost: Invalid image", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y);
			    g.drawString("Average gender cost: Invalid image", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE);
			}
		    
		    if(!Main.inTesting) {
			    g.drawString("Images until gender backpropagation: " + (FIFTY_IMAGES - ((Main.validGenderCount - 1) % Main.MOD_BY_FIFTY + 1)), DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * DOUBLE);
			    g.drawString("Images until age backpropagation: " + (FIFTY_IMAGES - ((Main.validAgeCount - 1) % Main.MOD_BY_FIFTY + 1)), DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * TRIPLE);
		    	g.drawString("Gender backpropogation count: " + Main.validGenderCount / FIFTY_IMAGES, DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * QUADRUPLE);
		    	g.drawString("Age backpropogation count: " + Main.validAgeCount / FIFTY_IMAGES, DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * QUINTUPLE);
		    }
		    else {
			    g.drawString("Images until gender backpropagation: Disabled", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * DOUBLE);
			    g.drawString("Images until age backpropagation: Disabled", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * TRIPLE);
		    	g.drawString("Gender backpropogation: Disabled", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * QUADRUPLE);
		    	g.drawString("Age backpropogation: Disabled", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * QUINTUPLE);
		    }
		    
		    if(!Main.genderInvalid) {
			    if(Main.guessedGender == 0) {
			    	g.drawString("Gender guess: Female", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * SEXTUPLE);
			    }
			    else {
			    	g.drawString("Gender guess: Male", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * SEXTUPLE);
			    }
		    }
		    else {
			    g.drawString("Gender guess: Invalid image", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * SEXTUPLE);
		    }
		    
		    if(!Main.genderInvalid) {
			    if(Main.currentGender == 0) {
			    	g.drawString("Labeled gender: Female", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * SEPTUPLE);
			    }
			    else {
			    	g.drawString("Labeled gender: Male", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * SEPTUPLE);
			    }
		    }
		    else {
			    g.drawString("Labeled gender: Invalid image", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * SEPTUPLE);
		    }
		    g.drawString("Valid gender images traversed: " + Main.validGenderCount, DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * OCTUPLE);
		    
		    /*
		    if(!Main.ageInvalid) {
			    if(Main.guessedAge == 0) {
			    	g.drawString("Gender guess: Female", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * OCTUPLE);
			    }
			    else {
			    	g.drawString("Gender guess: Male", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * OCTUPLE);
			    }
		    }
		    else {
			    g.drawString("Age guess: Invalid image", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * OCTUPLE);
		    }
		    
		    if(!Main.ageInvalid) {
			    if(Main.currentAge == 0) {
			    	g.drawString("Labeled gender: Female", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * NONUPLE);
			    }
			    else {
			    	g.drawString("Labeled gender: Male", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * NONUPLE);
			    }
		    }
		    else {
			    g.drawString("Age gender: Invalid image", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * NONUPLE);
		    }
		    */
		    
		    // Tell whether weight restriction is on
		    if(Main.disableWeightRestriction) {
		    	g.drawString("Weight restriction: Disabled", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * TREDECUPLE);
		    }
		    else {
		    	g.drawString("Weight restriction: Enabled", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * TREDECUPLE);
		    }
	    	g.drawString("Total remaining male count: " + Main.maleData.size(), DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * QUATTUORDECUPLE);
	    	g.drawString("Total remaining female count: " + Main.femaleData.size(), DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * QUINDECUPLE);
		    
		    // Don't print testing info if we're training
		    if(!Main.checkTestingCompleted() && Main.inTesting) {
			    g.drawString("Gender images guessed correctly: " + Main.numGenderCorrect, DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * SEXDECUPLE);
			    g.drawString("Total gender testing images remaining: " + (int) (Math.min(Main.femaleData.size(), Main.maleData.size())), DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * SEPTENDECUPLE);
			    
			    // Change the color temporarily to highlight the accuracy value when we finished testing
			    if(Main.checkTestingCompleted()) {
			    	g.setColor(Color.MAGENTA);
			    }
			    g.drawString("Gender accuracy: " + (Math.round((CONVERT_TO_PERCENTAGE * Main.numGenderCorrect / Main.validGenderCount) * ROUNDING) / ROUNDING) + "%", DEBUG_LABEL_START_X, DEBUG_LABEL_START_Y + LABEL_SPACE * OCTODECUPLE);
				if(Main.darkMode) {
					g.setColor(Color.WHITE);
				}
				else {
					g.setColor(Color.BLACK);
				}
		    }
		    
		    // Draw the controls (from bottom to top)
		    g.drawString("Toggle graphics: G", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y);
		    g.drawString("Toggle dark mode: D", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - LABEL_SPACE);
		    g.drawString("Toggle automatic image advancement: A", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - DOUBLE * LABEL_SPACE);
		    g.drawString("Load next image manually: Space or Enter", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - TRIPLE * LABEL_SPACE);
		    g.drawString("Toggle weight restriction (limit to [-1, 1]): B", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - QUADRUPLE * LABEL_SPACE);
		    g.drawString("Controls", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - QUINTUPLE * LABEL_SPACE);
		    
		    // Training and testing complete messages
		    if(!Main.inTesting && Main.checkTestingCompleted()) {
		    	// Print this text if we finished testing
		    	// Change the text according to whether a file was printed
		    	g.setColor(Color.GREEN);
		    	if(!Main.fileWritten) {
				    g.drawString("the network file.", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - SEXTUPLE * LABEL_SPACE);
				    g.drawString("Testing Completed! Press W to write", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - SEPTUPLE * LABEL_SPACE);
		    	}
		    	else {
					g.drawString("this CNN trainer.", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - SEXTUPLE * LABEL_SPACE);
					g.drawString("File written! Thank you for using", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - SEPTUPLE * LABEL_SPACE);
			    }
		    }
		    else if(!Main.inTesting && Main.checkGenderCompleted()) {
				// Print this text if we finished training
				g.setColor(Color.GREEN);
				g.drawString("an accuracy test.", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - SEXTUPLE * LABEL_SPACE);
				g.drawString("Training Completed! Press T to perform", FIRST_CONTROL_LABEL_X, FIRST_CONTROL_LABEL_Y - SEPTUPLE * LABEL_SPACE);
			}
			
		    if(!Main.disableGraphics) {
				// Draw the image in the top right corner of the screen
				g.drawImage(Main.currentImage, CURRENT_IMAGE_X, CURRENT_IMAGE_Y, Main.INPUT_CONVOLUTION_SIZE * DOUBLE, Main.INPUT_CONVOLUTION_SIZE * DOUBLE, null);	
				
				// Draw the gender input convolution
				drawConvolutionLayer(Main.genderInputConvolutionLayer, INPUT_LAYER, g, true);
				
				// Draw the gender hidden layer 1
				drawConvolutionLayer(Main.genderHiddenConvolutionLayer1, HIDDEN_LAYER1, g, false);
				
				// Draw the gender hidden layer 2
				drawConvolutionLayer(Main.genderHiddenConvolutionLayer2, HIDDEN_LAYER2, g, false);
				
				// Draw gender kernel in input convolution layer
				drawKernelLayer(Main.genderInputConvolutionLayer, INPUT_LAYER, g);
				
				// Draw gender kernel in hidden convolution layer 1
				drawKernelLayer(Main.genderHiddenConvolutionLayer1, HIDDEN_LAYER1, g);
				
				// Draw gender kernel in hidden convolution layer 2
				drawKernelLayer(Main.genderHiddenConvolutionLayer2, HIDDEN_LAYER2, g);
			    
				// Draw the output layer
				g.setFont(new Font("TimesRoman", Font.PLAIN, 30));
				Graphics2D g2D = (Graphics2D) g;
				g2D.setStroke(new BasicStroke(1)); 
				
				for(int i = 0; i < Main.genderOutputLayer.length; i ++) {
					int grayscale = (int) (Main.genderOutputLayer[i].activation * 255);
					
					// Because of the way doubles work, it's possible to get a value very slightly above 255 or below 0
					if(grayscale > 255) {
						grayscale = 255;
					}
					else if(grayscale < 0) {
						grayscale = 0;
					}
					g.setColor(new Color(grayscale, grayscale, grayscale));
					g.fillOval(OUTPUT_START_X , OUTPUT_START_Y + OUTPUT_SPACING * i, NODE_SIZE, NODE_SIZE);
					if(Main.darkMode) {
						g.setColor(Color.WHITE);
						g.drawOval(OUTPUT_START_X, OUTPUT_START_Y + OUTPUT_SPACING * i, NODE_SIZE, NODE_SIZE);
					}
					g.setColor(Color.GRAY);
					g.drawString(Math.round(Main.genderOutputLayer[i].activation * ROUNDING) / ROUNDING + "", OUTPUT_START_X + NODE_SIZE + TEXT_SPACE, OUTPUT_START_Y + (NODE_SIZE / HALF) + TEXT_SPACE + OUTPUT_SPACING * i);	
				}
		    }
		}
		else {
			// Set the font and stroke size
			g.setFont(new Font("TimesRoman", Font.PLAIN, FONT_SIZE_SIXTY)); 
			
			// The thickness of the line doesn't exist in Graphics 1D
			Graphics2D g2D = (Graphics2D)g;
		    g2D.setStroke(new BasicStroke(10));
	
			// Draw loading bar
			g.setColor(Color.RED);
			g.fillRect(LOADING_BAR_X, LOADING_BAR_Y, (int) (((double) Main.loadingCount / Main.TOTAL_STARTUP_ACTIONS) * LOADING_BAR_WIDTH), LOADING_BAR_HEIGHT);

			// Set color of text depending on whether dark mode is enabled
			if(Main.darkMode) {
				g.setColor(Color.WHITE);
			}
			else {
				g.setColor(Color.BLACK);
			}
			g.drawRect(LOADING_BAR_X, LOADING_BAR_Y, LOADING_BAR_WIDTH, LOADING_BAR_HEIGHT);
			
			// Draw loading text
			g.drawString("Loading...", LOADING_TEXT_X, LOADING_TEXT_Y);
			
			// Draw the percentage loaded
			g.drawString((int) (((float) Main.loadingCount / Main.TOTAL_STARTUP_ACTIONS) * CONVERT_TO_PERCENTAGE) + "%", LOADING_PERCENTAGE_TEXT_X, LOADING_PERCENTAGE_TEXT_Y);
		}
	}

	public int determineBrightness(float weight) {
		// Returns a thickness value for the connection drawing based on the weight
		weight = (weight / Main.largestGenderKernelWeightMagnitude) * MAX_COLOR_VALUE;
		
		if(weight < -1 * MAX_COLOR_VALUE) {
			weight = -1 * MAX_COLOR_VALUE;
		}
		
		if(weight > MAX_COLOR_VALUE) {
			weight = MAX_COLOR_VALUE;
		}
		
		return (int) weight;
	}
	
	public void drawConvolutionLayer(Convolution [] layerToDraw, int layer, Graphics g, boolean isInputLayer) {
		// Draw the layer specified
		for(int i = 0; i < layerToDraw.length; i ++) {
			for(int j = 0; j < layerToDraw[i].nodeMatrix.length; j ++) {
				for(int k = 0; k < layerToDraw[i].nodeMatrix[j].length; k ++) {
					if(isInputLayer) {
						int color = (int) layerToDraw[i].nodeMatrix[j][k].unsquishedActivation;
						if(i == Main.RED_INDEX) {
							g.setColor(new Color(color, 0, 0));
						}
						else if(i == Main.GREEN_INDEX) {
							g.setColor(new Color(0, color, 0));
						}
						else if(i == Main.BLUE_INDEX) {
							g.setColor(new Color(0, 0, color));
						}
					}
					else {
						int color = (int) (layerToDraw[i].nodeMatrix[j][k].activation * MAX_COLOR_VALUE);
						g.setColor(new Color(color, color, color));
					}

					g.fillRect(CONVOLUTION_START_X + CONVOLUTION_PIXEL_SIZE * j + layer * (Main.INPUT_CONVOLUTION_SIZE * CONVOLUTION_PIXEL_SIZE + Main.KERNEL_SIZE * KERNEL_PIXEL_SIZE + CONVOLUTION_SPACE * DOUBLE), CONVOLUTION_START_Y + CONVOLUTION_PIXEL_SIZE * k + layerToDraw[i].nodeMatrix.length * i * CONVOLUTION_PIXEL_SIZE + CONVOLUTION_SPACE * i, CONVOLUTION_PIXEL_SIZE, CONVOLUTION_PIXEL_SIZE);
				}
			}
			
			// Draw a box around each convolution
			if(Main.darkMode) {
				g.setColor(Color.GRAY);
				g.drawRect(CONVOLUTION_START_X + layer * (Main.INPUT_CONVOLUTION_SIZE * CONVOLUTION_PIXEL_SIZE + Main.KERNEL_SIZE * KERNEL_PIXEL_SIZE + CONVOLUTION_SPACE * DOUBLE), CONVOLUTION_START_Y + layerToDraw[i].nodeMatrix.length * i * CONVOLUTION_PIXEL_SIZE + CONVOLUTION_SPACE * i, layerToDraw[i].nodeMatrix.length * CONVOLUTION_PIXEL_SIZE, layerToDraw[i].nodeMatrix.length * CONVOLUTION_PIXEL_SIZE);
			}
		}
	}
	
	public void drawKernelLayer(Convolution [] layerToDraw, int layer, Graphics g) {
		// Draw the kernels specified
		for(int i = 0; i < layerToDraw.length; i ++) {
			for(int j = 0; j < layerToDraw[i].kernelMatrices.length; j ++) {
				for(int k = 0; k < layerToDraw[i].kernelMatrices[j].length; k ++) {
					for(int l = 0; l < layerToDraw[i].kernelMatrices[j][k].length; l ++) {
						int color = determineBrightness(layerToDraw[i].kernelMatrices[j][k][l]);
						if(color < 0) {
							g.setColor(new Color(-1 * color, 0, 0));
						}
						else {
							g.setColor(new Color(0, 0, color));
						}

						g.fillRect(CONVOLUTION_START_X + KERNEL_PIXEL_SIZE * k + layerToDraw[0].nodeMatrix.length * CONVOLUTION_PIXEL_SIZE + CONVOLUTION_SPACE + layer * (Main.INPUT_CONVOLUTION_SIZE * CONVOLUTION_PIXEL_SIZE + Main.KERNEL_SIZE * KERNEL_PIXEL_SIZE + CONVOLUTION_SPACE * DOUBLE), CONVOLUTION_START_Y + KERNEL_PIXEL_SIZE * l + layerToDraw[i].kernelMatrices[j].length * j * KERNEL_PIXEL_SIZE + (CONVOLUTION_SPACE + layerToDraw[i].kernelMatrices[j].length * layerToDraw[i].kernelMatrices.length * KERNEL_PIXEL_SIZE) * i, KERNEL_PIXEL_SIZE, KERNEL_PIXEL_SIZE);
					}
				}
				
				// Draw a box around each convolution
				if(Main.darkMode) {
					g.setColor(Color.GRAY);
					g.drawRect(CONVOLUTION_START_X + layerToDraw[0].nodeMatrix.length * CONVOLUTION_PIXEL_SIZE + CONVOLUTION_SPACE + layer * (Main.INPUT_CONVOLUTION_SIZE * CONVOLUTION_PIXEL_SIZE + Main.KERNEL_SIZE * KERNEL_PIXEL_SIZE + CONVOLUTION_SPACE * DOUBLE), CONVOLUTION_START_Y + layerToDraw[i].kernelMatrices[j].length * j * KERNEL_PIXEL_SIZE + (CONVOLUTION_SPACE + layerToDraw[i].kernelMatrices[j].length * layerToDraw[i].kernelMatrices.length * KERNEL_PIXEL_SIZE) * i, Main.KERNEL_SIZE * KERNEL_PIXEL_SIZE, Main.KERNEL_SIZE * KERNEL_PIXEL_SIZE);
				}
			}
		}	
	}
}
