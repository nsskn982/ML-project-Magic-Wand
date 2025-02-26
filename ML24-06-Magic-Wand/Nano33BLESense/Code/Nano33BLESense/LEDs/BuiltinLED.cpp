/**
*
*  @file BuiltinLED.cpp
*
*  @brief file of the library for using the built-in LED
*	
*
*  
*/


#include <LED.h>
#include <BuiltinLED.h>


/**  
*  @brief initialize digital pin LED_BUILTIN as an output.
*
*  call the function once when you press reset or power the board
*
*  Initialization of the pin LED_BUILTIN as output
*
*  @param ---
*
*  @return 0 - success
*  @return 1 - error
*/  
int PowerLEDinit() {
  // initialize digital pin LED_BUILTIN as an output.
  LEDinit(LED_BUILTIN);
  
  return 0;
}

/** 
*  @brief function for switching on or off the LED
*  
*  
*  swichting the led on / off 
*
*  @param On - true: switch on, false: switch off
*
*  @return 0 - success
*  @return 1 - error
*/  
int BuiltinLED(bool On) 
{
  LED(LED_BUILTIN, On);   // turn the LED on = true or off = false
  
  return 0:
}