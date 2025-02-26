/**
*  @file TestLEDRGBBrightness.ino
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

int redBrightness = 0;   /*< Define the initial brightness values for red (0-255)   */
int greenBrightness = 0; /*< Define the initial brightness values for green (0-255) */
int blueBrightness = 0;  /*< Define the initial brightness values for blue (0-255)  */
        

int redStep = 5;   /*< Define the increment/decrement value for red   */
int greenStep = 3; /*< Define the increment/decrement value for green */
int blueStep = 7;  /*< Define the increment/decrement value for blue  */


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
*  In each loop, the color is changing
*
*  @param ---
*
*  @return void
*/  
void loop() {
    // Write the PWM values to the LED pins
    RGBLED_Colors(redBrightness, greenBrightness, blueBrightness);
           
    // Update the brightness values for each color
    redBrightness += redStep;
    greenBrightness += greenStep;
    blueBrightness += blueStep;
            
    // Check if the brightness values are out of range and reverse the direction
    if (redBrightness <= 0 || redBrightness >= 255) {
       redStep = -redStep;
    }
    if (greenBrightness <= 0 || greenBrightness >= 255) {
        greenStep = -greenStep;
    }
    if (blueBrightness <= 0 || blueBrightness >= 255) {
        blueStep = -blueStep;
    }
                    
    // Wait for 10 milliseconds
    delay(10);
}
