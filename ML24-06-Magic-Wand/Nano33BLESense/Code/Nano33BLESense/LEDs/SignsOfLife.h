/**
*
*  @file SignsOfLife.h
*
*  @brief header file for the function signs of life with a LED
*	
*
*  
*/

#ifndef SignsOfLife_h
#define SignsOfLife_h

int SignsOfLifePinNo = -1;  /*<  Stores the pin number  */	
	
int SignsOfLifeCycleTimeOn  = 1000;  /*<  Stores the time which the LED is on  */	

int SignsOfLifeCycleTimeOff = 1000;  /*<  Stores the time which the LED is off  */	
	
unsigned long SignsOfLifeCurrentMillis = 0;  /*<  Stores the last time the LED was turned on */
		
unsigned long SignsOfLifePreviousMillis = 0; /*< Stores the last time the LED was turned on */

bool SignsOfLifeState = false; /*< Stores the actual state (On or Off)  */


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
*  @return 0 - success
*  @return 1 - error
*/  
bool SignsOfLifeInit(int PinNO, int CycleTimeOn, int CycleTimeOff);

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
bool SignsOfLife();




#endif