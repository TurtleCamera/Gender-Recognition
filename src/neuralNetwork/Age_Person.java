package neuralNetwork;

public class Age_Person {
	public String dob; // Date of birth
	public int yearTaken; // Year when the photo was taken (assume taken in middle of year)
	public String path; // File path
	
	public Age_Person(String date, int year, String filePath) {
		dob = date;
		yearTaken = year;
		path = filePath;
	}
}