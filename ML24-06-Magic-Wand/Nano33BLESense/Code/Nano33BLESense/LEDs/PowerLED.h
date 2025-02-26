/**
*
*  @file PowerLED.h
*
*  @brief header file of the library for using the power LED
*	
*/

#ifndef PowerLED_h
#define PowerLED_h

#define LED_PWR 25 /*< Define the pin for the power LED */


/**  
*  @brief initialize digital pin LED_PWR as an output.
*
*  call the function once when you press reset or power the board
*
*  Initialization of the pin LED_PWR as output
*
*  @param ---
*
*  @return 0 - success
*  @return 1 - error
*/  
int PowerLEDinit();

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
int PowerLED(bool On);

#endif