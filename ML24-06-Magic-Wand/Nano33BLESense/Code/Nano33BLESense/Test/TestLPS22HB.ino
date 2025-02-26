/**
*
*  @file TestLPS22HB.ino
*
*  @brief Example code for using the pressure sensor LPS22HB with the Arduino Nano 33 BLE Sense board.
*	
*
*  
*  @details We include the necessary libraries: Wire.h for I2C communication and LPS22HB.h for the sensor LPS22HB.
*  We create an instance of the class LPS22HB called lps.
*  In the function setup(), we initialize the serial communication at 9600 baud and start the I2C bus.
*  We call the function begin() on the lps instance to initialize the sensor.
*  In the function loop(), we read the pressure and temperature data using the readPressure() and readTemperature() functions, respectively.
*  We print the data to the serial monitor using Serial.print() and Serial.println().
*  We add a delay of 1 second to update the data every second.
*  
*  Note:
*  
*  Make sure to connect the sensor LPS22HB to the board Arduino Nano 33 BLE Sense correctly.
*  This example assumes that the sensor is calibrated and ready to use.
*  You can modify the code to suit your specific needs, such as changing the update rate or adding more functionality.
*  
*  Tips:
*  
*  Use the function Wire.begin() to initialize the I2C bus before using the  class LPS22HB.
*  Use the function lps.begin() to initialize the sensor before reading data.
*  Use the function delay()  to add a delay between readings, or use a timer to update the data at a specific interval.
*/

#include <Wire.h>
#include <LPS22HB.h>

/**
 * @class LPS22HB
 *
 * @brief Class for interacting with the pressure sensor LPS22HB.
 */
LPS22HB lps;

/**
 * @brief Setup function for the Arduino board.
 *
 * @details Initializes the serial communication and I2C bus.
 */
void setup() {
  // Initialize serial communication at 9600 baud
  Serial.begin(9600);

  // Initialize I2C bus
  Wire.begin();

  // Initialize the sensor LPS22HB
  lps.begin();
}

/**
 * @brief Main loop function for the Arduino board.
 *
 * @details Reads pressure and temperature data from the sensor LPS22HB and prints it to the serial monitor.
 */
void loop() {
  // Read pressure data from the sensor LPS22HB
  int32_t pressure = lps.readPressure();

  // Read temperature data from the sensor LPS22HB
  int32_t temperature = lps.readTemperature();

  // Print pressure and temperature data to the serial monitor
  Serial.print("Pressure: ");
  Serial.print(pressure);
  Serial.println(" hPa");
  Serial.print("Temperature: ");
  Serial.print(temperature);
  Serial.println(" °C");

  // Add a delay to update the data every second
  delay(1000);
}