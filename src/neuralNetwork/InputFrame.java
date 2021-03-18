package neuralNetwork;

import java.awt.event.KeyEvent;
import java.awt.event.KeyListener;

import javax.swing.JFrame;

@SuppressWarnings("serial")
public class InputFrame extends JFrame implements KeyListener {
	
	public static int layerSelected = 0; // Tells which layer the user is currently on
	
	@Override
	public void keyPressed(KeyEvent e) {
		// Advances training data
		if(e.getKeyCode() == KeyEvent.VK_ENTER || e.getKeyCode() == KeyEvent.VK_SPACE) {
			Main.next = true;
		}	// When toggled, training data advances automatically
		else if(e.getKeyCode() == KeyEvent.VK_A) {
			Main.autoAdvance = !Main.autoAdvance;
		}	// When toggled, weights are no longer restricted to the range [-1, 1]
		else if(e.getKeyCode() == KeyEvent.VK_B) {
			Main.disableWeightRestriction = !Main.disableWeightRestriction;
		}
		else if(e.getKeyCode() == KeyEvent.VK_G) {
			Main.disableGraphics = !Main.disableGraphics;
		}	// Can only be pressed after training completes-enters accuracy test mode
		else if(e.getKeyCode() == KeyEvent.VK_T && Main.checkGenderCompleted() && !Main.inTesting) {
			Main.inTesting = true;
		}	// Write a file (once again, can only be pressed after testing has completed
		else if(e.getKeyCode() == KeyEvent.VK_W && Main.checkTestingCompleted() && !Main.inTesting) {
			Main.fileWritten = true;
		}
	}

	@Override
	public void keyReleased(KeyEvent e) {
		// Toggles between light and dark mode
		if(e.getKeyCode() == KeyEvent.VK_D) {
			Main.darkMode = !Main.darkMode;
		}
	}

	@Override
	public void keyTyped(KeyEvent e) {
		// Nothing needed here
	}
}
