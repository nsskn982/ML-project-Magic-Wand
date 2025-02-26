/**
*
*  @file TestMicrophone.ino
*
*  @brief Simple application using the Microphone. 
*	
*
*  The following things are missed:
* - Circuit
* - part list
* - manuel
* - used pins
* - SPL, dBV, RMS,...
*  
* There are two buttons: blue and red
* While the blue button is pressed, the measuring is going on.
* If the red button is pressed, the measuring is stopped.
*
*/

#include <PDM.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#define SCREEN_WIDTH 128 /*< wondow size width of the display */
#define SCREEN_HEIGHT 64 /*< wondow size height of the display */



Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1); /*< initialization of the display */

const int buttonPin = 11;          /*< Pin of the internal button */
const int microphoneDataPin = 7;  /*< 1. Pin for the microphone */
const int microphoneClockPin = 9; /*< 1. Pin for the microphone */

bool isMeasuring = false;    /*< flag recording */
unsigned long startTime = 0; /*< start time of the recording */
unsigned long endTime   = 0; /*< end time of the recording */

const int measureThresholdMilliseconds = 1200; /*< maximal recording time */

short sampleBuffer[256];         /*< buffer for the samples */
int maxValuesFromBuffer[9999];   /*< ? */
int lastBufferIndex = 0;         /*< ? */

volatile int samplesRead;       /*< global value for the actual sample */
volatile int maxSampleValue;    /*< global value for the actual maximal sample */
float maxdBspl = 0.0;           /*<  variable for the highest dB SPL value */
bool isDelayOver = false;       /*< Variable to check whether the start delay has expired */

// Constants for dB SPL conversion
const float Vref = 3.3;      /*< Reference voltage of the microcontroller (in volts) */
const float Vrms = 0.0036;   /*< RMS voltage corresponding to 94 dB SPL */
const float sensitivity = 42.0;  /*< Microphone sensitivity (in dBV) */

const int blueButtonPin = A6;  /*< Pin of the blue button */
const int redButtonPin = A7;   /*< Pin of the red button */
  
// Variables for averaging over 200 ms
const unsigned long averagingTime = 200;  /*< Time [ms] to moving average the values */
unsigned long averagingStartTime = 0;     /*< Start time [ms] to moving average the values */
int valueSum = 0;                         /*< sum of the values to moving average */
int valueCount = 0;                       /*< number of the values to moving average */

float absoluteSum = 0.0; /*< absolute sum of the values to moving average */
int absoluteCount = 0;   /*< absolute number of the values to moving average */

int redButtonResult = 0; /*< Button pressed? */



/**  
*  @brief the setup function runs once when you press reset or power the board
*
*  standard function of Arduino sketches
*  
*  Initialization of 
*  - the display
*  - the interrupt pin
*  - the microphone (PDM)
*
*  @param ---
*
*  @return void
*/  
void setup() {
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.clearDisplay();
  display.setTextColor(WHITE);
  display.setTextSize(2);

  pinMode(buttonPin, INPUT_PULLUP);
  Serial.begin(9600);

  PDM.onReceive(onPDMdata);
}

/**  
*  @brief Interrupt service routine for the microphone
*
*
*  Attention: as short as possible
*
*  If an interrupt is on, th function checks if there are samples, 
*  copies them into the buffer and sets a flag
*
*  @param ---
*
*  @return void
*/  
void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(sampleBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;
}

/** 
*  @brief the loop function runs over and over again forever
*  
*  standard function of Arduino sketches
*  
*  The function reads the samples and sends the samples to the display
*
*  @param ---
*
*  @return void
*/  
void loop() {
  HandleInput();
  HandleUI();
}

/** 
*  @brief the function hnadles the input
*  
*  Is the blue button pressed?
*  Is the red button pressed?
*  Start sampling
*  Stop sampling
*  call display
*
*  @param ---
*
*  @return void
*/  
void HandleInput() {
  bool blueButtonIsPressed = analogRead(blueButtonPin) == LOW;
  bool redButtonIsPressed = analogRead(redButtonPin) == LOW;

  if (blueButtonIsPressed)
    isMeasuring = true;

  if (redButtonIsPressed)
    isMeasuring = false;
    
  bool doNextStep = redButtonIsPressed || blueButtonIsPressed;
  if (doNextStep)
  {
    if (isMeasuring) 
    {
      StartSampling(); 
      absoluteSum = 0.0;
      absoluteCount = 0; 
      maxdBspl = 0.0; // Resetting the highest dB SPL value at the start of a new measurement
      isDelayOver = false; // Reset start delay
      startTime = millis(); // Record start time
    } 
    else 
    {
      StopSampling();
      if (redButtonResult == 0)
      {
        ShowMaxdBspl(); // Display of the highest dB SPL value at the end of the measurement
        redButtonResult = 1;
      }
      else
      {
        ShowAveragedBspl();
        redButtonResult = 0;
      }
    }
  }

  delay(50);
  
  // Check whether the start delay has expired
  bool isMeasureThresholdReached = (millis() - startTime) >= measureThresholdMilliseconds;
  if (isMeasuring && !isDelayOver && isMeasureThresholdReached) {
    isDelayOver = true;
  }
}


/** 
*  @brief the function gets samples
*  
*  call PDM for getting the samples
*
*  @param ---
*
*  @return void
*/  
void StartSampling() {
  if (!PDM.begin(1, 16000)) {
    Serial.println("Failed to start Measurment!");
    while (1);
  }
}

/** 
*  @brief the function ends the sampling
*  
*  call PDM for stopping
*
*  @param ---
*
*  @return void
*/  
void StopSampling() {
  PDM.end();
}

/** 
*  @brief the function shows the results
*  
*  display the result:
*  - clear the display
*  - set the cursor
*  - set the text size
*  - display the samples
*
*  @param ---
*
*  @return void
*/  
void ShowResult() {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(2);

  float dBspl = getDbValueFromPMC(maxSampleValue);
  
  display.println("dBSPL:");
  display.setTextSize(3);
  display.setCursor(20, 32);
  display.println(dBspl);
  display.display();

  if (isDelayOver && dBspl > maxdBspl) { // Check for the highest dB SPL value after the start delay
    maxdBspl = dBspl;
  }
}

/** 
*  @brief the function runs over and over again forever
*  
*  calculate the size of the maximal sample value in undefined unit
*  Use of a magic number
* 
*  @param pmcValue: maximal sample value; not used!
*
*  @return void
*/  
float getDbValueFromPMC(int pmcValue) {
  float maxSampleVoltage = maxSampleValue * Vref / 32767.0;
  float dBspl = 20.0 * log10(maxSampleVoltage / Vrms) + sensitivity;
  return dBspl;
}

/** 
*  @brief the function displays the samples
*  
* 
*  @param ---
*
*  @return void
*/  
void ShowMicrophoneValues() {
  if (samplesRead && isDelayOver) {
    maxSampleValue = 0;
    for (int i = 0; i < samplesRead; i++) {
      if (maxSampleValue < abs(sampleBuffer[i])) {
        maxSampleValue = abs(sampleBuffer[i]);
      }
    }
    samplesRead = 0;

    // Add current value to the sum
    valueSum += maxSampleValue;
    valueCount++;

    absoluteSum += getDbValueFromPMC(maxSampleValue);
    absoluteCount++;
    
    // Check if averaging time has elapsed
    unsigned long currentTime = millis();
    if (currentTime - averagingStartTime >= averagingTime) {
      // Calculate average value
      int averagedValue = valueSum / valueCount;

      if (lastBufferIndex == 9998)
        lastBufferIndex = 0;
      maxValuesFromBuffer[++lastBufferIndex] = averagedValue;

      averagingStartTime = currentTime;
      valueSum = 0;
      valueCount = 0;

      ShowResult();
    }
  }
}

/** 
*  @brief the function displays the samples
*  
* 
*  @param title: title of the graphic
*  @param dbSpl: ?
*
*  @return void
*/  
void ShowdBValue(String title, float dbSpl) {
  display.clearDisplay();
  display.setCursor(0, 0);
  display.setTextSize(2);

  if (isDelayOver)
  {
    display.println(title);
    display.println(dbSpl);
  }
  else
  {
    display.println("Messung zu   kurz");
  }

  display.display();
}

/** 
*  @brief the function displays the maximal value
*  
* 
*  @param ---
*
*  @return void
*/  
void ShowMaxdBspl() {
  ShowdBValue("Max dBSPL:    ", maxdBspl);
}

/** 
*  @brief the function displays the average value
*  
* 
*  @param ---
*
*  @return void
*/  
void ShowAveragedBspl() {
  ShowdBValue("Average   dBSPL:             ", absoluteSum / absoluteCount);
}

/** 
*  @brief the function displays the samples
*  
* 
*  @param ---
*
*  @return void
*/  
void HandleUI() {
  if (isMeasuring) {
    ShowMicrophoneValues();
  } else {
    //display.clearDisplay();
    display.display();
  }
}
