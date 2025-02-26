/**
*
*  @file TestPushButtonInterrupt.ino
*
*  @brief Simple application reading in the built-in push button states using an interrupt
*	
*
*  If the builtin push button is pressed, the built-in LED is switched on  for 2 second and switched off again.
*  But in this example, an interrupt is used.
*  
*/

#include <LED.h>
#include <BuiltinLED.h>

#define BUTTON_PIN  BUTTON_B /*< Use the onboard push button (BUTTON_B) */



 
// Initialize variables
//
volatile bool pushPressed = false; /*< // Flag, whether the button is pressed. Declare as volatile for interrupt.  safety */

int ledState = 0; /*<  LED-Status zur Verarbeitng */
 
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
  // Initialize the interrupt function    
  attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), buttonPressed, FALLING);
}
        

/**  
*  @brief Interrupt service function
*
*  Attention: as short as possible
*
*  The function changes just one flag.
*/
void buttonPressed() 
{
  if (pushPressed == false) 
  {
    pushPressed = true;
  }	
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
  if (pushPressed)
  {
    // Turn the LED on
    BuildinLED(SET_ON);
    // Wait for one second
    delay(2000);
    // Turn the LED off
    BuildinLED(SET_OFF);
    pushPressed = false;	  
  }	
  // ... 
}