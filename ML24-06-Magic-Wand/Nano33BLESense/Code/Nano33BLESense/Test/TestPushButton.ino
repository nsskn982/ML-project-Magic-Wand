/**
*
*  @file TestPushButton.ino
*
*  @brief Simple application reading in the built-in push button states. 
*	
*
*  If the built-in push button is pressed, the the internal built-in LED is turn on for 2 seconds

*  
*/

#include <LED.h>
#include <BuiltinLED.h>

#define BUTTON_PIN  BUTTON_B /*< Use the onboard push button (BUTTON_B) */

 
int buttonState = 0; /*< state of the built-in push button */
 
/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  standard function of Arduino sketches
*  
*  Initialization of the built-inb LED and the push button
*
*  @param ---
*
*  @return void
*/  
 
void setup() {
  // Initialize the pin as an output
  BuiltinLEDinit();
  // Initialize the pin as an input
  pinMode(BUTTON_PIN, INPUT_PULLUP);
}

        
/** 
*  @brief the loop function runs over and over again forever
*  
*  standard function of Arduino sketches
*  
*  swichting the built-in led on for 2 sec, if the push button is pressed
*
*  @param ---
*
*  @return void
*/  
void loop() 
{
  buttonState = digitalRead(BUTTON_PIN);
  if (buttonState == HIGH) 
  {	
    // Turn the LED on
	BuiltinLED(SET_ON);
    // Wait for two second
    delay(2000);
    // Turn the LED off
	BuiltinLED(SET_OFF);
    // Wait for one second
    delay(1000);
	buttonState = LOW;
  }	
}