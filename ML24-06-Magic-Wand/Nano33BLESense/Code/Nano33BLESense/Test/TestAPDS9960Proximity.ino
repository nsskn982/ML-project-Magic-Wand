/**
*
*  @file TestAPDS9960Proximity.ino
*
*  @brief This sketch tests the proximity sensor
*	
*  The sketches reads the distance and print the distance in the serial monitor 
*  
*/

#include <Arduino_APDS9960.h>   /*< Library for APDS9960 sensor */


int proximity; /*< Measured value for the distance */


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
*  The function checks, if a sensor value, here proximity, is available
* if a proximity is available, then the function reads the distance and sends the value to the serial monitor
*
*  @param ---
*
*  @return void
*/
void loop() 
{
  // check whether distance measurement is available, if not 5 ms
  if (APDS.proximityAvailable()) 
  { 
    // read the Measured value 
    proximity = APDS.readProximity();
    // print the measured value
    if (proximity != -1) {
      Serial.print("Distance: ");
      Serial.println(proximity);
    }
  }
 
}