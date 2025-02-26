#include <Arduino.h>
#include <U8g2lib.h>
#include <SPI.h>
#include <SD.h>

#ifdef U8X8_HAVE_HW_SPI
#include <SPI.h>
#endif
#ifdef U8X8_HAVE_HW_I2C
#include <Wire.h>
#endif

U8G2_SH1106_128X64_NONAME_F_HW_I2C u8g2(U8G2_R0, /* reset=*/ U8X8_PIN_NONE, /* clock=*/ A5, /* data=*/ A4);

const int heartRatePin = A0;
const int batteryPin = A1;
const int chipSelect = 10;

int heartRateValue = 0;
int lastHeartRateValue = 0;
unsigned long lastPeakTime = 0;
unsigned long lastHeartbeatTime = 0;
bool peakDetected = false;
int threshold = 512;
int heartRate = 0;
int heartRateArray[10];
int heartRateIndex = 0;
int heartRateSum = 0;
int previousHeartRate = 0;
const unsigned long errorTimeout = 5000;
bool errorState = false;

unsigned long lastSwitchTime = 0;
const unsigned long switchInterval = 10000;
bool showGraph = false;
int graphData[108];

bool sdCardFull = false;
unsigned long startTime;

bool initialMeasurement = true;

void setup(void) {
  Serial.begin(9600);
  pinMode(heartRatePin, INPUT);
  pinMode(batteryPin, INPUT);

  u8g2.begin();

  Serial.print("Initialisiere SD-Karte...");
  if (!SD.begin(chipSelect)) {
    Serial.println("SD-Karten Initialisierung fehlgeschlagen!");
    return;
  }
  Serial.println("SD-Karte erfolgreich initialisiert.");

  u8g2.clearBuffer();
  u8g2.setFont(u8g2_font_ncenB08_tr);

  const char* startText1 = "Messung startet,";
  const char* startText2 = "bitte Geduld.";

  int16_t x1 = (128 - u8g2.getStrWidth(startText1)) / 2;
  int16_t x2 = (128 - u8g2.getStrWidth(startText2)) / 2;

  int16_t y1 = 30;
  int16_t y2 = 45;

  u8g2.drawStr(x1, y1, startText1);
  u8g2.drawStr(x2, y2, startText2);

  u8g2.sendBuffer();

  startTime = millis();
}

void loop(void) {
  if (millis() - startTime < 5000) {
    return;
  }

  if (sdCardFull) {
    return;
  }

  heartRateValue = analogRead(heartRatePin);
  float rawBatteryValue = analogRead(batteryPin);
  float voltage = rawBatteryValue * (3.3 / 1023.0);

  Serial.print("Gemessene Spannung an A1: ");
  Serial.print(voltage);
  Serial.println(" V");

  if (heartRateValue > threshold && lastHeartRateValue <= threshold) {
    if (peakDetected == false) {
      peakDetected = true;
      unsigned long peakInterval = millis() - lastPeakTime;
      lastPeakTime = millis();
      lastHeartbeatTime = millis();
      errorState = false;
      initialMeasurement = false;
      
      if (peakInterval > 0) {
        heartRate = 60000 / peakInterval;

        heartRateSum -= heartRateArray[heartRateIndex];
        heartRateArray[heartRateIndex] = heartRate;
        heartRateSum += heartRate;
        heartRateIndex = (heartRateIndex + 1) % 10;
        int averageHeartRate = heartRateSum / 10;

        Serial.print("Herzfrequenz: ");
        Serial.println(averageHeartRate);

        if (averageHeartRate != previousHeartRate) {
          previousHeartRate = averageHeartRate;

          if (!showGraph) {
            u8g2.clearBuffer();
            u8g2.setFont(u8g2_font_ncenB08_tr);
            u8g2.drawStr(0, 20, "Herzfrequenz:");
            u8g2.setFont(u8g2_font_fub25_tr);

            char heartRateStr[10];
            sprintf(heartRateStr, "%d", averageHeartRate);
            int16_t x = (128 - u8g2.getStrWidth(heartRateStr) - u8g2.getStrWidth(" SPM")) / 2;
            u8g2.drawStr(x, 50, heartRateStr);
            u8g2.drawStr(x + u8g2.getStrWidth(heartRateStr), 50, " SPM");

            drawBattery(voltage);

            u8g2.sendBuffer();
          }
        }

        updateGraph(averageHeartRate);

        saveToSD(averageHeartRate);
      }
    }
  } else if (heartRateValue <= threshold) {
    peakDetected = false;
  }

  if (!initialMeasurement && millis() - lastHeartbeatTime > errorTimeout) {
    errorState = true;
    showGraph = false;
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(0, 20, "Herzfrequenz:");
    u8g2.setFont(u8g2_font_fub25_tr);
    u8g2.drawStr(30, 50, "Error");

    drawBattery(voltage);

    u8g2.sendBuffer();
  } else {
    errorState = false;
  }

  if (!errorState && millis() - lastSwitchTime > switchInterval) {
    showGraph = !showGraph;
    lastSwitchTime = millis();
  }

  if (showGraph) {
    drawGraph();
  }

  lastHeartRateValue = heartRateValue;
  delay(10);
}

void drawBattery(float voltage) {
  int batteryLevel = map(voltage * 10, 0, 90, 0, 4);

  u8g2.drawFrame(100, 2, 24, 10);
  u8g2.drawBox(124, 4, 2, 6);

  for (int i = 0; i < batteryLevel; i++) {
    u8g2.drawBox(102 + (i * 5), 4, 4, 6);
  }
}

void updateGraph(int heartRate) {
  for (int i = 0; i < 107; i++) {
    graphData[i] = graphData[i + 1];
  }
  if (heartRate > 40) {
    graphData[107] = heartRate;
  } else {
    graphData[107] = 0;
  }
}

void drawGraph() {
  int xOffset = 10;
  int yOffset = 10;

  u8g2.clearBuffer();
  u8g2.drawLine(xOffset, yOffset + 44, xOffset + 108, yOffset + 44);
  u8g2.drawLine(xOffset, yOffset, xOffset, yOffset + 44);

  u8g2.setFont(u8g2_font_ncenB08_tr);
  u8g2.drawStr(xOffset + 60, yOffset + 54 , "Zeit in s");
  u8g2.drawStr(xOffset - 10, yOffset - 2, "SPM");

  for (int i = 1; i < 108; i++) {
    int y1 = yOffset + 44 - map(graphData[i - 1], 40, 140, 0, 44);
    int y2 = yOffset + 44 - map(graphData[i], 40, 140, 0, 44);
    if (graphData[i - 1] > 0 && graphData[i] > 0) {
      u8g2.drawLine(xOffset + i - 1, y1, xOffset + i, y2);
    }
  }

  u8g2.sendBuffer();
}

void saveToSD(int heartRate) {
  File dataFile = SD.open("datalog.txt", FILE_WRITE);

  if (dataFile) {
    dataFile.print("Herzfrequenz: ");
    dataFile.print(heartRate);
    dataFile.println(" SPM");
    dataFile.close();
    Serial.println("Daten auf SD-Karte gespeichert.");
  } else {
    Serial.println("Fehler beim Oeffnen der Datei zum Schreiben. SD-Karte koennte voll sein.");
    sdCardFull = true;
    u8g2.clearBuffer();
    u8g2.setFont(u8g2_font_ncenB08_tr);
    u8g2.drawStr(0, 20, "SD-Karte voll!");
    u8g2.sendBuffer();
  }
}
