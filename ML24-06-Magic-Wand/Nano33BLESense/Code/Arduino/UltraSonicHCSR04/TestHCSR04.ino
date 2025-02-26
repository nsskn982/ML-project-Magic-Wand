/**
* @file UltraschallsensorTest.ino
* @author Wings
* @date 20.05.2024
* @details The sketch is used to test the functionality of the ultrasonic sensor. The transit time of the ultrasonic signal is measured
* The distance is then calculated from the transit time and the speed of sound and displayed on the serial monitor.
* Projekt: Abstandssensor
**/

/** Pin D6 is referenced as an echo pin for the ultrasonic sensor. */
#define Echopin D6
/** Pin D9 is referenced as the trigger pin for the ultrasonic sensor. */
#define Triggerpin D9

/** Der Parameter MaxReichweite definiert die obere Grenze des Messbereichs von dem Ultraschallsensor und ist hier in cm angegeben. */
int MaxReichweite = 215;
/** Der Parameter MinReichweite definiert die untere Grenze des Messbereichs von dem Ultraschallsensor und ist hier in cm angegeben. */
int MinReichweite = 3;
/** The parameter Anstand  stands for the distance between a wall and the ultrasonic sensor and is shown on the display in cm. */
double Abstand;
/** The duration parameter stands for the runtime of the ultrasonic signal and is used here with the unit microseconds. */
double Dauer;
/** The  parameter Schallgeschwindigkeit stands for the speed of sound in the medium air at a temperature of 20C and is specified in m/s. */
double Schallgeschwindigkeit = 343.2;

/**
* @brief Starten der seriellen Kommunikation und festlegen der Pinmodi
* @details The function is executed when the programme is started. The serial interface to the computer is initialised at 9600 baud, the pin mode for the trigger pin is defined as OUTPUT 
* and the pin mode for the echo pin is defined as INPUT 
* There is no return value.
*/
void setup() {
  Serial.begin(9600);
  pinMode(Triggerpin, OUTPUT);
  pinMode(Echopin, INPUT);
}

/**
* @brief Abstand messen und anzeigen
* @details The function is performed continuously. An ultrasonic signal is sent out and the runtime is measured until the signal arrives back at the ultrasonic sensor. 
* The distance in cm is calculated using the duration of the runtime and the speed of sound and then displayed on the serial monitor. 
* There is no return value.
*/
void loop() {
  digitalWrite(Triggerpin, LOW); // Der Triggerpin wird auf Low gestellt
  delayMicroseconds(2); // Es wird 2 Millisekunden gewartet
  digitalWrite(Triggerpin, HIGH); // Das Senden eines Ultraschallsignals wird initiiert
  delayMicroseconds(10); // Es wird 10 Millisekunden gewartet
  digitalWrite(Triggerpin, LOW); // Der Triggerpin wird auf Low gestellt
  Dauer = pulseIn(Echopin, HIGH); // Measure the time until the signal returns
  Abstand = 0.5 * Schallgeschwindigkeit * Dauer * pow(10,-6) * 100; // The distance to the reflective surface is calculated in cm
  Serial.print(Abstand);
  Serial.println(" cm");
  delay(5000);
}
