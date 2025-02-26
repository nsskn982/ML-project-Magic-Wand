/**
*
*  @file TestLEDRGBApplication.ino
*
*  @brief Simple application using the built-in RGB-LED of the Arduino Nano 33 BLE Sense
*	
*
*  @details The red part of the RGB-LED is switched on for 1 second and switched off for 29 second so that red part of the LED flashes accordingly.

*  
*/

#include <../LEDs/RGBLED.h>
#include <../LEDs/LED.h>
#include <../LEDs/SignsOfLife.h>


#define CycleTimeOn 1000 /*< Duty cycle[ms] */

#define CycleTimeOff 29000 /*< Switch-off time [ms] */



/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  standard function of Arduino sketches
*  
*  Initialization of the pin LED_RED, the red part of the builtin RGB-LED for the signs of life
*
*  @param ---
*
*  @return void
*/  
void setup() {
    // Initialize the function SignsOfLife
	SignsOfLifeInit(LED_RED, CycleTimeOn, CycleTimeOff)
	
	// Application
	// ...
}
 

   
/** 
*  @brief the loop function runs over and over again forever
*  
*  standard function of Arduino sketches
*  
*  swichting the led on / off for the signs of life
*
*  @param ---
*
*  @return void
*/  
void loop() {
	// Switch red part of the RGB LED on/off
	SignsOfLife();
	
	// Application
	// ...
}