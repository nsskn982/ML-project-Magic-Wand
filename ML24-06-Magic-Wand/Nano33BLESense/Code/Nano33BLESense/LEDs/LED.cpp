/**
*
*  @file LED.cpp
*
*  @brief file of the library for using  LEDs
*	
*/

#include "LED.h"


/**  
*  @brief initialize digital pin of the LED as an output.
*
*  call the function once in the function setup when you press reset or power the board
*
*  Initialization of the pin  as output
*
*  The program doesn't check if the parameter PinNo is reasonable.
*
*  @param PinNo - Nummer des Pins
*
*  @return 0 - success
*  @return 1 - error
*/  
int LEDinit(int PinNo) {
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(PinNo, OUTPUT);
  
  return 0;
}

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
int LED(int PinNo, bool On) 
{
  if (On)
  {	  
    digitalWrite(PinNo, HIGH);   // turn the LED on (HIGH is the voltage level)
  } else
  {
    digitalWrite(PinNo, LOW);    // turn the LED off by making the voltage LOW
  }
  
  return 0:
}

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
int LEDBrightness(int PinNo, int setBrightness)
{
  int b;

  if (setBrightness <= 0) {
    b = 0;  
  } else
  if (setBrightness >= 255) {
    b = 255;  
  } else
  {	  
    b = setBrightness;
  }	 
  
  analogWrite(PinNo, b);
  
  return 0;
}
