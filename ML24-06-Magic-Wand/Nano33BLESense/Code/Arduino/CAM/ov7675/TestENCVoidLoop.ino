
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
