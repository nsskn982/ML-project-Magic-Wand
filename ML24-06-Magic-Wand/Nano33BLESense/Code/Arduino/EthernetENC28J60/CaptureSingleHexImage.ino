/*
  CaptureSingleHexImage

  Dieser Code nimmt einzelne bilder auf die als 
  Hexadezimal-String gespeichert werden.

*/

#include <Arduino_OV767X.h>

unsigned short pixels[176 * 144]; // QCIF: 176x144 X 2 bytes per pixel (RGB565)

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("OV767X Camera Capture");
  Serial.println();

  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }

  Serial.println("Camera settings:");
  Serial.print("\twidth = ");
  Serial.println(Camera.width());
  Serial.print("\theight = ");
  Serial.println(Camera.height());
  Serial.print("\tbits per pixel = ");
  Serial.println(Camera.bitsPerPixel());
  Serial.println();

  Serial.println("Send the 'c' character to read a frame ...");
  Serial.println();
}

void loop() {
  if (Serial.read() == 'c') {
    Serial.println("Reading frame");
    Serial.println();
    Camera.readFrame(pixels);

    int numPixels = Camera.width() * Camera.height();

    for (int i = 0; i < numPixels; i++) {
      unsigned short p = pixels[i];

      if (p < 0x1000) {
        Serial.print('0');
      }

      if (p < 0x0100) {
        Serial.print('0');
      }

      if (p < 0x0010) {
        Serial.print('0');
      }

      Serial.print(p, HEX);
    }
  }
}
