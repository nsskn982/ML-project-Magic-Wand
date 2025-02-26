# Python code to send a letter to the Arduino
import serial
import time
        
# Adapt the port of the Arduino; e.g. 'COM3' under Windows  
ArduinoPort = 'COM3';
        
# Set baud rate; corresponds to the baud rate used in the Arduino sketch
Baudrate = 9600;
        
try:
    # Open the serial port to which the Arduino is connected
    ser = serial.Serial(ArduinoPort, Baudrate)
        
    # Wait until the connection is established.
    time.sleep(3)
        
    print(f"Connection to the Arduino established via {ArduinoPort}.")
        
    # Send data
    ser.write(b"Hello, Arduino!")
       
    # Receive data
    received_data = ser.readline().decode().strip()
    print(f"Received data: {received_data})
        
    # Close serial connection
    ser.close();
        
except serial.SerialException:
   print(f"Error when opening the serial connection to {ArduinoPort}.")
        
        
        
# Send the letter 'A' to the Arduino.
ser.write(b"A")
        
# Read the answer from the Arduino.
res = ser.read()
        
# Print the answer.
print(res)
        
# Close the port
ser.close()