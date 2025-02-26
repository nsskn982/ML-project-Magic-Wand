// Arduino code to react to a letter
void setup() {
    // Initialize the serial communication.
    Serial.begin(9600);
        
    // Define a pin as an output.
    pinMode(13, OUTPUT);
}
    
void loop() {
    // Check whether data is available.
    if (Serial.available()) {
        // Read the byte received.
        char c = Serial.read();
            
        // React differently depending on the byte.
        switch (c) {
            case 'A':
                // Switch on the LED.
                digitalWrite(13, HIGH);
                
                // Send back a confirmation.
                Serial.write("OK");

                // Wait a second.
                delay(1000);

                // Switch the LED off.
                digitalWrite(13, LOW);
            break;
            
            // Add more cases here if you wish.
        }
    }
}                
