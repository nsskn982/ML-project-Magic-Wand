/**
*  @file TestLEDPower.ino
*
*  @brief Simple program for testing the power LED
*	
*
*  Turns the power LED on for one second, then off for one second, repeatedly.
*
*  The LED is switched on for 1 second and switched off 
*  for 1 second so that the LED flashes accordingly.
*
*  On the Arduino Nano 33 BLE Sense, it is attached to digital pin 25
*
*/

#incude <../LEDs/PowerLED.h>
#incude <../LEDs/LED.h>
 
/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  standard function of Arduino sketches
*  
*  Initialization of the pin LED_BUILTIN as output
*
*  @param ---
*
*  @return void
*/  
void setup() {
    // Initialize the pin as an output
    PowerLEDinit();
}
        
/** 
*  @brief the loop function runs over and over again forever
*  
*  standard function of Arduino sketches
*  
*  swichting the led on / off for 1sec.
*
*  @param ---
*
*  @return void
*/  
void loop() {
    // Turn the LED on
    PowerLED(SET_ON);
    // Wait for one second
    delay(1000);
    // Turn the LED off
    PowerLED(SET_OFF);
    // Wait for one second
    delay(1000);
}
