/*
  TestCameraRawBites320x240x2

  Dieser Sketch liest ein Bild von der OmniVision OV7670-Kamera
  und schreibt die Bytes an den seriellen Port. Verwenden Sie den 
  Processing-Sketch "CameraVisualizerHochkant320x240x2"
  um die Kameraausgabe zu visualisieren.
*/

#include <Arduino_OV767X.h>

int bytesPerFrame;
// Definiere die Pin-Nummer fuer die LED
const int ledPin = A6;



byte data[320 * 240 * 2]; // QVGA: 320x240 X 2 bytes per pixel (RGB565)

void setup() {
  Serial.begin(9600);
  while (!Serial);
    // Setze den Pin-Modus als Ausgang
    pinMode(ledPin, OUTPUT);

  if (!Camera.begin(QVGA, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);

      digitalWrite(ledPin, LOW);
  }

  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
  

  // Optionally, enable the test pattern for testing
  // Camera.testPattern();
}
  void loop() {
  Camera.readFrame(data);

  Serial.write(data, bytesPerFrame);
    digitalWrite(ledPin, HIGH);
}
