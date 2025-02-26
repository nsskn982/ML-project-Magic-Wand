/*
  CameraVisualizerHochkant320x240x2

Dieser Code liest ein Bild von einer OV7670-Kamera ueber die 
serielle Schnittstelle ein,konvertiert die empfangenen RGB565-Daten
 RGB-Daten und zeigt das Bild um 90 Grad gedreht 
auf dem Bildschirm an.

*/

import processing.serial.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

Serial myPort;

// must match resolution used in the sketch
final int cameraWidth = 320;
final int cameraHeight = 240;
final int cameraBytesPerPixel = 2;
final int bytesPerFrame = cameraWidth * cameraHeight * cameraBytesPerPixel;

PImage myImage;
byte[] frameBuffer = new byte[bytesPerFrame];

void setup()
{
  size(240, 320);

  // Seriellen Port auswaehlen
  myPort = new Serial(this, Serial.list()[0], 9600);          

  // if you know the serial port name
  //myPort = new Serial(this, "COM5", 9600);                    // Windows
  //myPort = new Serial(this, "/dev/ttyACM0", 9600);            // Linux
 // myPort = new Serial(this, "/dev/cu.usbmodem14401", 9600);     // Mac

  // wait for full frame of bytes
  myPort.buffer(bytesPerFrame);  

  myImage = createImage(cameraWidth, cameraHeight, RGB);
}

void draw()
{
  // Rotate the image by 90 degrees clockwise
  pushMatrix();
  translate(width / 2, height / 2);
  rotate(HALF_PI);
  image(myImage, -myImage.width / 2, -myImage.height / 2);
  popMatrix();
}

void serialEvent(Serial myPort) {
  // read the saw bytes in
  myPort.readBytes(frameBuffer);

  // access raw bytes via byte buffer
  ByteBuffer bb = ByteBuffer.wrap(frameBuffer);
  bb.order(ByteOrder.BIG_ENDIAN);

  int i = 0;

  while (bb.hasRemaining()) {
    // read 16-bit pixel
    short p = bb.getShort();

    // convert RGB565 to RGB 24-bit
    int r = ((p >> 11) & 0x1f) << 3;
    int g = ((p >> 5) & 0x3f) << 2;
    int b = ((p >> 0) & 0x1f) << 3;

    // set pixel color
    myImage .pixels[i++] = color(r, g, b);
  }
 myImage .updatePixels();

}