/**
*  @file TestLEDRGB.ino
*
*  @brief Simple program for testing the RGB-LED
*	
*
*  Turns the RGB-LED on for one second, then off for one second, repeatedly.
*
*  The LED is switched on for 1 second and switched off 
*  for 1 second so that the LED flashes accordingly.
*
*  On the Arduino Nano 33 BLE Sense, it is attached to digital pin 22, 23, 24
*
*/


#incude <../LEDs/RGBLED.h>
#incude <../LEDs/LED.h>

/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  Standard function of Arduino sketches
*  
*  Initialization of the pin  22, pin 23, and 24 as output
*
*  @param ---
*
*  @return void
*/  
void setup() {
    // Initialize the pin as an output
    RGBLEDinit();
}



/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  Standard function of Arduino sketches
*  
*  In each loop, the red part is switched on for 1 sec, 
*  then the green part is switched on for 1 sec, and
*  at last the blue part is switched on.
*
*  @param ---
*
*  @return void
*/  
void loop() {
    // Turn the red LED on
    RGBLED_Red(SET_ON);
    // Wait for one second
    delay(1000);
    // Turn the LED off
    RGBLED_Red(SET_OFF);
    // Turn the green LED on
    RGBLED_Green(SET_ON);
    // Wait for one second
    delay(1000);
    // Turn the LED off
    RGBLED_Green(SET_OFF);
    // Turn the blue LED on
    RGBLED_Blue(SET_ON);
    // Wait for one second
    delay(1000);
    // Turn the LED off
    RGBLED_BLue(SET_OFF);
    // Wait for one second
    delay(1000);
}
