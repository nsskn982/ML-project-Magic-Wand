/**
*
*  @file TestLED.ino
*
*  @brief Simple program for testing an external LED
*	
*
*  How to control an external LED with the Arduino Nano 33 BLE Sense
*
* The LED is switched on for 1 second and switched off 
* for 1 second so that the LED flashes accordingly.
*
*
*  Author: Elmar Wings
*  Created: 06.08.2024
*/


#include <../LEDs/LED.h>
#include <../LEDs/SignsOfLife.h>

#define LED_EXT 14 /*< Define the pin number  for the builtin-LED */

#define CycleTimeOn 1000 /*< Duty cycle [ms] */

#define CycleTimeOff 1000 /*< Switch-off time [ms] */

 
 /**  
*  @brief the setup function runs once when you press reset or power the board
*
*  standard function of Arduino sketches
*  
*  Initialization of the pin LED_EXT as output
*
*  @param ---
*
*  @return void
*/  
void setup() {
    // Initialize the function SignsOfLife
	SignsOfLifeInit(LED_EXT, CycleTimeOn, CycleTimeOff)
	
	// Application
	// ...
}
        
/** 
*  @brief the loop function runs over and over again forever
*  
*  standard function of Arduino sketches
*  
*  swichting the led on / off for 1sec.
*
*  @param ---
*
*  @return void
*/  
void loop() {
 	// Switch builtin LED on/off
	SignsOfLife();
	
	// Application
	// ...
}