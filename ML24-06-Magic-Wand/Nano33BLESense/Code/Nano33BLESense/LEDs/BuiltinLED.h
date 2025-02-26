/**
*
*  @file BuiltinLED.h
*
*  @brief header file of the library for using the built-in LED
*	
*/

#ifndef BuiltinLED_h
#define BuiltinLED_h

#define LED_BUILTIN 13 /*< Define the pin for the builtin LED */


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
int BuiltinLEDinit();

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
int BuiltinLED(bool On);

#endif