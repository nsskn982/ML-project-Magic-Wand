/*
  TestENC

Dieser Code implementiert einen Webserver auf einem Arduino 
mit einem Ethernet-Shield, der es ermoeglicht, ueber eine 
Webschnittstelle eine LED zu steuern und Aktionen auszuloesen.

*/

#include <SPI.h>
#include <EthernetENC.h>

#define ON HIGH
#define OFF LOW
#define RELAY A6

byte mac[] = { 0xDE, 0xAD, 0xBE, 0xEF, 0xFE, 0xED }; // MAC-Adresse des Ethernet-Shields
IPAddress ip(192, 168, 0, 141); // Statische IP-Adresse des Arduinos

EthernetServer server(80); // Ethernet-Server auf Port 80

String displayText = "Initialer Text"; // Variable fuer den anzuzeigenden Text
bool testExecuted = false; // Variable zum Speichern des Teststatus

