int heartRatePin = A0;
int heartRateValue = 0;
unsigned long previousMillis = 0;
const long interval = 500;

void setup() {
  Serial.begin(9600);
  pinMode(heartRatePin, INPUT);
}

void loop() {
  unsigned long currentMillis = millis();

  if (currentMillis - previousMillis >= interval) {
    previousMillis = currentMillis;

    heartRateValue = analogRead(heartRatePin);
    Serial.println(heartRateValue);
  }
}