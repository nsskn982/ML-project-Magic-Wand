#include <Wire.h>

#define LSM9DS1_M_ADDR 0x1E // I2C Address of the LSM9DS1 magnetometer module
#define LSM9DS1_AG_ADDR 0x6B // I2C Address of the LSM9DS1 accelerometer-gyroscope module

void setup() {
    Serial.begin(9600); // Initialize serial communication
    Wire.begin(); // Initialize Wire library
    // Initialize LSM9DS1
    initLSM9DS1();
}

void loop() {
    // Read LSM9DS1 data
    readLSM9DS1();
    delay(1000); // Pause for one second between readings
}

void initLSM9DS1() {
    // Initialize magnetometer module
    Wire.beginTransmission(LSM9DS1_M_ADDR);
    Wire.write(0x20); // Control register address for mode
    Wire.write(0x1C); // Activate continuous mode at 10 Hz
    Wire.endTransmission();

    // Initialize accelerometer-gyroscope module
    Wire.beginTransmission(LSM9DS1_AG_ADDR);
    Wire.write(0x10); // Control register address for gyro mode
    Wire.write(0x38); // Activate continuous m
