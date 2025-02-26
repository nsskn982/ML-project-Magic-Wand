/*
  OV767X - Camera Capture Raw Bytes

  This sketch reads a frame from the OmniVision OV7670 camera
  and writes the bytes to the Serial port.

  This sketch waits for the letter 'c' on the Serial Monitor,
  it then reads a frame from the OmniVision OV7670 camera and 
  prints the data to the Serial Monitor as a series of bytes.

  The website https://rawpixels.net - can be used the visualize the data:
    width: 176
    height: 144
    offset: 0
    Predefined Format: RGB565
    Pixel Format: RGBA
    Ignore Alpha checked
    Little Endian not checked

  Circuit:
    - Arduino Nano 33 BLE board
    - OV7670 camera module:
      - 3.3 connected to 3.3
      - GND connected GND
      - SIOC connected to A5
      - SIOD connected to A4
      - VSYNC connected to 8
      - HREF connected to A1
      - PCLK connected to A0
      - XCLK connected to 9
      - D7 connected to 4
      - D6 connected to 6
      - D5 connected to 5
      - D4 connected to 3
      - D3 connected to 2
      - D2 connected to 0 / RX
      - D1 connected to 1 / TX
      - D0 connected to 10

  This example code is in the public domain.
*/

#include <Arduino_OV767X.h>

long bytesPerFrame; // variable to hold total number of bytes in image
long numPixels;

// Declare a byte array to hold the raw pixels recieved from the OV7670
// Array size is set for QCIF; if other format requied, change size
// QCIF: 176x144 X 2 bytes per pixel (RGB565)
byte data[176 * 144 * 2]; 

void setup() {
  Serial.begin(115200);
  while (!Serial);

  // Begin the OV7670 specifing resoultion (QCIF, Pixel format RGB565 and frames per second)
  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
  
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  numPixels = Camera.width() * Camera.height();

  // Optionally, enable the test pattern for testing
  // If the next line is uncommented, the OV7670 will output a test pattern
  //Camera.testPattern();
}

void loop() {
  // Wait for a 'c' from Serial port before taking frame
  if (Serial.read() == 'c') {
    
    // Read frame from OV7670 into byte array
    Camera.readFrame(data);

    // Write out each byte of the array to the serial port
    // Probaly a quicker way to do this
    for (int i = 0; i < bytesPerFrame; i++){
      Serial.write(data[i]);
    }

    // Write out a FF byte to tell receiving program that data is finished
    // Somewhat dangerous - but has worked so far
    delay(100);
    Serial.write(0xFF);
  }
}
