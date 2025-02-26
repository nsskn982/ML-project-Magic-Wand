
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
