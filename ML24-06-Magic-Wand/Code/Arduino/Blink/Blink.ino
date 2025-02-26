/**
*
*  @file Blink.ino
*
*  @brief Simple program for testing the configuration
*	
*
*  Turns an LED on for one second, then off for one second, repeatedly.
*
*  Most Arduinos have an on-board LED you can control. On the UNO, MEGA and ZERO
*  it is attached to digital pin 13, on MKR1000 on pin 6. LED_BUILTIN is set to
*  the correct LED pin independent of which board is used.
*  If you want to know what pin the on-board LED is connected to on your Arduino
*  model, check the Technical Specs of your board at:
*  
*  https://www.arduino.cc/en/Main/Products
*
*  modified 8 May 2014
*  by Scott Fitzgerald
*  modified 2 Sep 2016
*  by Arturo Guadalupi
*  modified 8 Sep 2016
*  by Colby Newman
*
*  This example code is in the public domain.
*
*  https://www.arduino.cc/en/Tutorial/BuiltInExamples/Blink
*/

/**
* 
*
* @mainpage Hello World
*
*  Date created: 28.8.2024
*
*  Path: Code/Arduino/Blink.ino
*
*  Version: 2.0
*
*  Reviewed by: 
*
*  Review Date: 
*
* @section description Introduction
*
* If you a new hardware based of Arduino, you have to configure the Arduino IDE. After connecting the hardware with the PC, you have to configure the IDE. If you have to test the communication between PC and the hardware, you have to test it with a simple program. If the program works, you are sure that the process 'Creating a sketch' -> 'Downloading a sketch' -> 'Running a sketch'. If the program works, then the LED is switching on and off every 1sec.
*
* @section Short description
*  
*  swichting the led on / off for 1sec.
*/



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
  // initialize digital pin LED_BUILTIN as an output.
  pinMode(LED_BUILTIN, OUTPUT);
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
  digitalWrite(LED_BUILTIN, HIGH);   // turn the LED on (HIGH is the voltage level)
  delay(1000);                       // wait for a second
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  delay(1000);                       // wait for a second
}
