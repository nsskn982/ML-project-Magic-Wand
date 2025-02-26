/**
*  
*  @file TestLEDPowerBattery.ino
*  
*  @brief Simple program for checking the battery state
*
*   if the battery state less 20%, the power LED will blink.
*
*/

#include <Arduino.h>
#incude <../LEDs/PowerLED.h>
#incude <../LEDs/LED.h>

const int BATTERY_PIN = A0; /*< Battery measurement pin */

const float REFERENCE_VOLTAGE = 3.3; /*< Reference voltage for 3.3V (adjust if using external power) */


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

  SignsOfLifeInit(LED_PWR, 500, 500);
  PowerLED(true);
}

/** 
*  @brief function for checking the battery state
*  
*  
*  swichting the led on / off 
*
*  @param SendMessages - true: messages are sent, false: messages are not sent
*
*  @return 0 - success
*  @return 1 - battery state less than 20 %
*/  
int BatteryState(bool SendMessages)
{
  int ret;
  // Read battery voltage from analog pin
  float rawVoltage = analogRead(BATTERY_PIN) * (REFERENCE_VOLTAGE / 1023.0);

  // Calculate percentage based on reference voltage (adjust based on battery specs)
  float batteryPercentage = (rawVoltage / 4.2) * 100.0;

  if (SendMessages)
  {
    // Print battery voltage and percentage (for debugging)
    Serial.print("Battery voltage: ");
    Serial.print(rawVoltage);
    Serial.println(" V");
    Serial.print("Battery percentage: ");
    Serial.print(batteryPercentage);
    Serial.println("%");
  }
  
  // Optional: blink LED based on battery level (adjust thresholds)
  if (LED_PWR >= 0) {
    if (batteryPercentage < 20) {
      digitalWrite(LED_PWR, HIGH);
      delay(500);
      digitalWrite(LED_PWR, LOW);
      delay(500);
	  ret = 1;
    } else {
      digitalWrite(LED_PWR, HIGH);
	  ret = 0;
    }
  }
  else
  {
	ret = 0;
  }
  return ret;  
}  
	

/** 
*  @brief the loop function runs over and over again forever
*  
*  inside the loop the function for checkuíng the battery state is called.
*
*  @param ---
*
*  @return void
*/  
void loop() {
 
  BatteryState(false);
  
  // Delay between measurements
  delay(5000);
}
