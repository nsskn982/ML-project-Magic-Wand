/**
*
*  @file LED.h
*
*  @brief header file for the library for using  LEDs
*	
*
*  
*/

#ifndef LED_h
#define LED_h

#define SET_ON  true  /*< Define flag for switching on  */
#define SET_OFF false /*< Define flag for switching off */


/**  
*  @brief initialize digital pin of the LED as an output.
*
*  call the function once when you press reset or power the board
*
*  Initialization of the pin  as  output
*
*  The program doesn't check if the parameter PinNo is reasonable.
*
*  @param PinNo - Nummer des Pins
*
*  @return 0 - success
*  @return 1 - error
*/  
int LEDinit(int PinNo);

/** 
*  @brief function for switching on or off the LED
*  
*  
*  swichting the led on / off 
*
*  @param PinNo - Nummer des Pins
*  @param On - true: switch on, false: switch off
*
*  @return 0 - success
*  @return 1 - error
*/  
int LED(int PinNo, bool On);


/** 
*  @brief setting the brightness 
*  
*  
*  setting the brightness 
*
*  The program checks if the parameter setBrightness is rasonable.
*
*  @param setBrightness: 0 - aus; 255 - maximal brightness
*
*  @return 0 - success
*  @return 1 - error
*/  
int LEDBrightness(int PinNo, int setBrightness);



#endif