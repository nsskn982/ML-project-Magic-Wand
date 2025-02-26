/**
*  @file TestLEDPowerBrightness.ino
*
*  @brief Simple program for controlling the brightness of the power LED of the Arduino Nano 33 BLE Sense
*	
*
*  Using PWM, the LED's brightness is increasing until full and reverse.
*
*/

#incude <../LEDs/PowerLED.h>
#incude <../LEDs/LED.h>

int Brightness = 0; /*< Define the initial brightness values  (0-255) */

int redStep = 5; /*< Define the increment/decrement value  */

  
/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  standard function of Arduino sketches
*  
*  Initialization of the pin LED_BUILTIN as output
*
*  @param ---
*
*  @return void
*/  
void setup() {
    // Initialize the pin as an output
    PowerLEDinit();
}
  
/** 
*  @brief the loop function runs over and over again forever
*  
*  standard function of Arduino sketches
*  
* changing continuously the LED's brightness
*
*  @param ---
*
*  @return void
*/  
void loop() {
  // Write the PWM values to the LED pin
  LEDBrightness(LED_PWR, redBrightness);

  // Update the brightness values 
  Brightness += Step;

  // Check if the brightness values are out of range and reverse the direction
  if (Brightness <= 0 || Brightness >= 255) {
    Step = -Step;
  }

  // Wait for 10 milliseconds
  delay(10);
}
