/**
*
*  @file SignsOfLife.cpp
*
*  @brief file of the library  signs of life with a LED
*	
*
*  
*/

#include "LED.h"
#include "SignsOfLife.h"


/**  
*  @brief initialize digital pin of the LED as an output.
*
*  call the function once, in the function setup, when you press reset or power the board
*
*  Initialization of the pin  as an output
*
*  The program doesn't check if the parameter PinNo is reasonable.
*
*  @param PinNo - number of pin for the LED
*  @param CycleTimeOn - duration of the LED is on
*  @param CycleTimeOff - duration of the LED is off
*
*  @return true - LED's actual state is on
*  @return false - LED's actual state is off
*/  
bool SignsOfLifeInit(int PinNo, int CycleTimeOn, int CycleTimeOff)
{
  LEDinit(PinNo);
  
  SignsOfLifePinNo = PinNo; 
	
  SignsOfLifeCycleTimeOn  = CycleTimeOn; 

  SignsOfLifeCycleTimeOff = CycleTimeOff; 
	
  SignsOfLifeCurrentMillis = millis();  
		
  SignsOfLifePreviousMillis = millis();


  SignsOfLifeState = false; 

  LED(SignsOfLifePinNo, SignsOfLifeState);
  
  return SignsOfLifeState;
}

/** 
*  @brief function for blinking the LED
*  
*  
*  The function checks the duration of the state. If the state has to change, then the function changes the state.
*   

*
*  @return true - LED's actual state is on
*  @return false - LED's actual state is off
*/  
bool SignsOfLife()
{

  // Check if CycleTimeOff have passed since the last LED turn on
  SignsOfLifeCurrentMillis = millis();
  if (SignsOfLifeState == true){
    if (SignsOfLifeCurrentMillis - SignsOfLifePreviousMillis >= SignsOfLifeCycleTimeOn) {
      LED(SignsOfLifePinNo, SET_OFF );   // Turn LED off
      SignsOfLifePreviousMillis = SignsOfLifeCurrentMillis;
	  SignsOfLifeState = false;
    }
  }
  else {
    if (SignsOfLifeCurrentMillis - SignsOfLifePreviousMillis >= SignsOfLifeCycleTimeOff) {
      LED(SignsOfLifePinNo, SET_ON );   // Turn LED on
      SignsOfLifePreviousMillis = SignsOfLifeCurrentMillis;
	  SignsOfLifeState = true;
    }
  }
  return SignsOfLifeState;
}

