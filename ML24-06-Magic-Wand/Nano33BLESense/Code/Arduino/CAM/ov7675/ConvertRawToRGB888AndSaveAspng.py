from PIL import Image
import os

# Delete the temp file 'imagefile' if it exists
if os.path.exists("imagefile"):
  os.remove("imagefile")
  print("Deleted temp image file")
else:
  print("Temp image file does not exist")

# Open file that contains the RGB565 raw data
image = open('camera',mode='rb')

# Open a temp file that will contain RGB888 raw data
result = open('imagefile', mode = 'wb')

# If using other picture dimensions, change here
Width = 176
Height = 144

# Binary masks used in RGB565 to RGB888 conversion 
MASK5 = 0b00011111
MASK6 = 0b00111111

print("Starting conversion")

# Loop through the raw file and convert to RGB888
# Note that the raw pixel data needs to be converted to interger
# in order for the bit shifting and masking to compute
# Result has to converted back
# This is done with commands .from_bytes and .to_bytes

for x in range (Width * Height):
  p = image.read(2) #read two bytes
  im = int.from_bytes(p, byteorder='big')
  #print(p)
  
  #Conversion of RGB565 to RGB888
  red = ((im >> 11) & MASK5) << 3
  green = ((im >> 5) & MASK6) << 2
  blue = (im >> 0 & MASK5) << 3

  r = red.to_bytes(1, 'big')
  g = green.to_bytes(1, 'big')
  b = blue.to_bytes(1, 'big')

  #print(red, green , blue)
  result.write(r)
  result.write(g)
  result.write(b)

result.close()
image.close()

rawData = open("imagefile", 'rb').read()

# the image size
imgSize = (Width,Height)

# create the image file ready for saving 
img = Image.frombytes('RGB', imgSize, rawData)

# can give any format you like .png, jpg etc.
img.save("finalpic.png")
print ("Finished conversion")
