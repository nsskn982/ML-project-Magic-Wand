/**
*
*  @file TestAPDS9960Gesture.ino
*
*  @brief This sketch tests the gesture sensor; not ready!
*	
*  This example initializes the APDS9960 sensor and enables the gesture detection. It then reads the gesture sensor data in the loop() function and prints the detected gesture type to the serial console.
*
*Note that the Arduino_APDS9960 library provides a more straightforward API for working with the gesture sensor, and it's a good choice if you want to use the library's built-in gesture recognition algorithms.
*
*Also, keep in mind that the Arduino_APDS9960 library is a third-party library, so you'll need to install it separately using the Arduino Library Manager.
+

*  
*/

#include <Arduino_APDS9960.h>   /*< Library for APDS9960 sensor */


APDS9960 apds; /*< variable for the the seonsor */


/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  standard function of Arduino sketches
*  
*  Initialization of the 
*  - serial communication; here: 9600 Baud
*  - the sensor 
*
*  @param ---
*
*  @return void
*/  
void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Initialize the APDS9960 sensor
  if (apds.begin()) {
    Serial.println("APDS9960 initialization complete");
  } else {
    Serial.println("APDS9960 initialization failed");
    while (1);
  }

  // Enable the gesture detection
  apds.enableGestureSensor();
}
/** 
*  @brief the loop function runs over and over again forever
*  
*  standard function of Arduino sketches
*  
*  The function checks, if a sensor value, here gesture, is available
* if a gesture is available, then the function reads the gesture and sends the value to the serial monitor
*
*  @param ---
*
*  @return void
*/
void loop() {
  // Read the gesture sensor data
  int gesture = apds.readGesture();

  // Print the gesture type
  switch (gesture) {
    case APDS9960_GESTURE_NONE:
      Serial.println("No gesture");
      break;
    case APDS9960_GESTURE_LEFT:
      Serial.println("Swipe left");
      break;
    case APDS9960_GESTURE_RIGHT:
      Serial.println("Swipe right");
      break;
    case APDS9960_GESTURE_UP:
      Serial.println("Swipe up");
      break;
    case APDS9960_GESTURE_DOWN:
      Serial.println("Swipe down");
      break;
    case APDS9960_GESTURE_DUAL_TAP:
      Serial.println("Double tap");
      break;
    case APDS9960_GESTURE_SHAKE:
      Serial.println("Shake");
      break;
  }

  delay(100);
}

