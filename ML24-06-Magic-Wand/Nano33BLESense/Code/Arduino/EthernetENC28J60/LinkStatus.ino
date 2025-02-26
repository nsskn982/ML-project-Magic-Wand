/*
  LinkStatus

    Dieser Sketch gibt den Ethernet-Verbindungsstatus aus. Wenn das
  Ethernet-Kabel angeschlossen ist, sollte der Verbindungsstatus 
  "ON" anzeigen.
*/

#include <SPI.h>
#include <EthernetENC.h>

void setup() {

  Ethernet.init(10);  // Most Arduino shields

  Serial.begin(9600);
}

void loop() {
  auto link = Ethernet.linkStatus();
  Serial.print("Link status: ");
  switch (link) {
    case Unknown:
      Serial.println("Unknown");
      break;
    case LinkON:
      Serial.println("ON");
      break;
    case LinkOFF:
      Serial.println("OFF");
      break;
  }
  delay(1000);
}
