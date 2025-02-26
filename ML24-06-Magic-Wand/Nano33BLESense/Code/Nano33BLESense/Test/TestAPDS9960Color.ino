/**
*
*  @file TestAPDS9960Color.ino
*
*  @brief This sketch tests the color sensor
*	
*  The sketches reads the color and print in the serial monitor the rgb values
*  
*/


#include <Arduino_APDS9960.h>   /*< Library for APDS9960 sensor */


int r; /*< Measured value for red */
int g; /*< Measured value for green */
int b; /*< Measured value for blue */

/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  standard function of Arduino sketches
*  
*  Initialization of the 
*  - serial communication; here: 9600 Baud
*  - the sensor 
*
*  @param ---
*
*  @return void
*/  
void setup() 
{
  Serial.begin(9600);

  // initialze the sensor
  if (!APDS.begin()) {
    Serial.println("Error initializing APDS9960 sensor.");
  }
}

/** 
*  @brief the loop function runs over and over again forever
*  
*  standard function of Arduino sketches
*  
*  The function checks, if a sensor value, here color, is available
*  if a color is available, then the function reads the color and sends the value to the serial monitor
*
*  @param ---
*
*  @return void
*/
void loop() 
{

  // check whether color detection is available
  if(APDS.colorAvailable()) 
  {  
    // read the measured values
    APDS.readColor(r, g, b);  
    // print the measured values
    Serial.println("Red = ",r, ", Green = ",g, ", BLUE = ", b );
 	
  }  
}