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

void setup() {
  pinMode(RELAY, OUTPUT);
  
  // LED Steuerung beim Start
  digitalWrite(RELAY, ON);  // LED an fuer 1 Sekunde
  delay(1000);
  digitalWrite(RELAY, OFF); // LED aus fuer 0,5 Sekunden
  delay(500);
  digitalWrite(RELAY, ON);  // LED an fuer 1 Sekunde
  delay(1000);
  digitalWrite(RELAY, OFF); // LED aus bleiben
  
  Serial.begin(9600); // Initialisierung der seriellen Kommunikation
  //while (!Serial) { ; } // Warte auf Verbindung zur seriellen Schnittstelle

  Ethernet.init(10); // Ethernet-Initialisierung mit CS-Pin 10
  if (Ethernet.begin(mac) == 0) { // Versuche DHCP-Konfiguration
    Serial.println("Failed to configure Ethernet using DHCP");
    Ethernet.begin(mac, ip); // Bei Fehler, verwende statische IP
  }

  server.begin(); // Start des Ethernet-Servers
  Serial.print("Server is at ");
  Serial.println(Ethernet.localIP()); // Ausgabe der IP-Adresse des Servers
}

void loop() {
  EthernetClient client = server.available(); // Warte auf Verbindung vom Client
  if (client) {
    Serial.println("New client");
    bool currentLineIsBlank = true;
    String request = "";

    while (client.connected()) {
      if (client.available()) {
        char c = client.read();
        request += c;

        if (c == '\n' && currentLineIsBlank) {
          Serial.println(request);

          // HTML-Seite senden
          client.println("HTTP/1.1 200 OK");
          client.println("Content-Type: text/html");
          client.println();
          client.println("<html><body style='background-color: #ADD8E6;'><h2 style='font-size: 48px;'>Livecam</h2></br>");
          client.println("<input type=submit value='Bildaufnahme' style='width:100px;height:45px' onClick=location.href='/?CAPTURE'>&nbsp&nbsp&nbsp");
          client.println("<input type=submit value='Test' style='width:100px;height:45px' onClick=location.href='/?TEST'><br/>");

          // Anzeigen des Textfelds
          client.println("<p>Hier wird ihr Bild in einem Hexadezimal-String ausgegeben:</p>");
          client.println("<p>");
          client.println(displayText);
          client.println("</p>");

          // HTTP-Anfrage analysieren
          if (request.indexOf("/?CAPTURE") != -1) {
            // Funktion fuer den Capture-Button: Funktion Capture aufrufen
            captureText();
            client.println("<p>Bild wurde erfasst</p>");
          } else if (request.indexOf("/?TEST") != -1) {
            // Funktion fuer den Test-Button: LED schnell blinken lassen
            blinkLED();
            testExecuted = true; // Teststatus aktualisieren
          }

          if (testExecuted) {
            client.println("<p>Test erfolgreich ausgefuehrt</p>");
            testExecuted = false; // Teststatus zuruecksetzen
          } else {
            // client.println("<p>Keine Aktion durchgefuehrt</p>");
          }

          client.println("</body></html>");
          break;
        }
        if (c == '\n') {
          currentLineIsBlank = true;
        } else if (c != '\r') {
          currentLineIsBlank = false;
        }
      }
    }
    delay(1); // Kurze Pause, bevor die Verbindung geschlossen wird
    client.stop(); // Schliesse Verbindung zum Client
    Serial.println("Client disconnected");
  }
}

// Funktion um die LED schnell zu blinken
void blinkLED() {
  for (int i = 0; i < 5; i++) {
    digitalWrite(RELAY, ON);
    delay(100);
    digitalWrite(RELAY, OFF);
    delay(100);
  }
}

// Funktion fuer den Capture-Button
void captureText() {
  digitalWrite(RELAY, ON);
  delay(2000);
  digitalWrite(RELAY, OFF);
  delay(200);
  digitalWrite(RELAY, ON);
  delay(2000);
  digitalWrite(RELAY, OFF);
  delay(200);
  
  // Hier koennte der Hexadezimal-String ausgegeben werden
  displayText = "0x8B42, 0x6B42, 0x8A42, 0x8A42, 0x8A42, 0x8B42, 0x6B42, 0x8B42, 0xAB42, 0xAB42, 0xCB42, 0xAB42, 0xAB42, 0xAC42, 0xAB42, 0xAC42, 0x8C42, 0x6C42, 0x6B42, 0xAB4A,";
}
