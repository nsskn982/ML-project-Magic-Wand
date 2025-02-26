#include <SPI.h>
#include <SD.h>

const int chipSelect = 10;

void setup() {
 
  Serial.begin(9600);
  while (!Serial) {
    ;
  }

  Serial.print("Initialisiere SD-Karte...");

  if (!SD.begin(chipSelect)) {
    Serial.println("SD-Karten Initialisierung fehlgeschlagen!");
    return;
  }
  Serial.println("SD-Karte erfolgreich initialisiert.");

  Serial.println("Erstelle Testdatei...");
  File testFile = SD.open("test.txt", FILE_WRITE);

  if (testFile) {
    Serial.println("Schreibe auf die Testdatei...");
    testFile.println("Hallo, dies ist ein Test, um die SD-Karte zu ueberpruefen.");
    testFile.close();
    Serial.println("Schreiben abgeschlossen.");
  } else {
    Serial.println("Fehler beim Oeffnen der Testdatei.");
  }

  Serial.println("Lese die Testdatei...");
  testFile = SD.open("test.txt");
  if (testFile) {
    Serial.println("test.txt Inhalt:");
    while (testFile.available()) {
      Serial.write(testFile.read());
    }
    testFile.close();
  } else {
    Serial.println("Fehler beim Oeffnen der Testdatei.");
  }
}

void loop() {
}
