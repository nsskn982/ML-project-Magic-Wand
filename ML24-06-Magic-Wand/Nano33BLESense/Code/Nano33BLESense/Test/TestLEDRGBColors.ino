/**
*  @file TestLEDRGBColors.ino
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
  // Set the LED pins as outputs
  RGBLEDinit();
}



/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  Standard function of Arduino sketches
*  
*  In each loop, different colors are tested
*
*  @param ---
*
*  @return void
*/  
void loop() {
	// red
	RGBLED_Color(255,0,0);
    // Wait for one second
    delay(1000);
	// green
	RGBLED_Color(0,255,0);
    // Wait for one second
    delay(1000);
	// blue
	RGBLED_Color(0,0,255);
    // Wait for one second
    delay(1000);
	// yellow
	RGBLED_Color(255,255,0);
    // Wait for one second
    delay(1000);
	// cyan
	RGBLED_Color(0,255,255);
    // Wait for one second
    delay(1000);
	// magenta
	RGBLED_Color(255,0,255);
    // Wait for one second
    delay(1000);
	// white
	RGBLED_Color(255,255,255);
    // Wait for one second
    delay(1000);
	// black
	RGBLED_Color(0,0,0);
    // Wait for one second
    delay(1000);
	// orange
	RGBLED_Color(255,127,0);
    // Wait for one second
    delay(1000);
	// pink
	RGBLED_Color(255,192,203);
    // Wait for one second
    delay(1000);
	// purple
	RGBLED_Color(120,0,128);
    // Wait for one second
    delay(1000);
}
