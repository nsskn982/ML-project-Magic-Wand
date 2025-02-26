import serial
import os

# Delete the existing camera file if it already exists
if os.path.exists("camera"):
  os.remove("camera")
  print("deleted file camera")
else:
  print("camera does not exist") 

# Open a raw file and set it up to receive camera data
image = open('camera',mode='wb')
stopChar = bytes.fromhex('ff')

# Open a serial port that is connected to an Arduino (below is Linux, Windows and Mac would be "COM4" or similar)
# No timeout specified; program will wait until all serial data is received from Arduino
# Port description will vary according to operating system. Linux will be in the form /dev/ttyXXXX
# Windows and MAC will be COMX. Use Arduino IDE to find out name 'Tools -> Port'
ser = serial.Serial('/dev/ttyACM0')
ser.flushInput()

# Write out a single character encoded in utf-8; this is defalt encoding for Arduino serial comms
# This character tells the Arduino to start sending data
ser.write(bytes('c', 'utf-8'))

# Loop through and read data received from camera
while True:
    #Read in data from Serial a byte at a time
    ser_byte = ser.read()
    #print(ser_byte)
        
    #If Arduino has sent a byte FF, exit loop
    if (ser_byte == stopChar):
         break
    
    #Write received data to file
    image.write(ser_byte)
            
# Close port and image file to exit
ser.close()
image.close()
print("image transfer complete")
