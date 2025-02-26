# Magic Wand with an Arduino Nano 33 BLE Sense

![Logo](./report/System/edgecomputer/Images/LogoUni.jpg "Our Logo")
### [author](./author.xlsx/)

Supervisor: Prof. Dr. Elmar Wings

Contributors:

- Adhiraj Walse - 7025711
- Sudeshna Nanda - 7026003
- Srikanth Nanda - 7026002

# Description of the Project

The "ML23-06 Magic Wand with an Arduino Nano 33 BLE Sense" project likely involves developing a wand-like device using the Arduino Nano 33 BLE Sense board. This wand can incorporate sensors and functionalities to simulate magical actions or control other devices via a micro-USB connection. It offers an engaging and interactive experience by integrating electronics, sensors, and programming to create a sense of magic.
 
## Table of contents

-   [Problem Description](#short-problem-description)
-   [Prerequisites](#prerequisites)
-   [Getting started](#getting-started)
-   [Ideas of the Project](#ideas-of-the-project)
-   [Directory Structure](#directory-structure)

## Short Problem Description
This project demonstrates how to detect gestures by waving a magic wand using the Arduino Nano 33 BLE Sense board. It leverages machine learning to analyze data from the accelerometer and gyroscope, enabling gesture recognition.

## Prerequisites

Before starting this project, ensure you have the following prerequisites:  

- **Arduino Nano 33 BLE Sense** board  
- **Arduino IDE** with the **ArduinoBLE** library installed  
- A **trained dataset of gestures**  
- Knowledge of **TinyML, TensorFlow, and TensorFlow Lite**

![ArduinoTop](https://github.com/Wings-hub/ML24-06-Magic-Wand-with-an-Arduino-Nano-33-BLE-Sense)


## Getting Started

To set up this project, follow these steps:  

1. **Clone or download** the project repository.  
2. **Connect** your Arduino Nano 33 BLE Sense to your computer.  
3. **Open** the Arduino IDE and load the provided sketch.  
4. **Upload** the sketch to your Arduino Nano 33 BLE Sense.  
5. **Follow** the deployment and usage instructions to train and run the gesture detection system.

## Ideas of the Project
- Our project has been tested with the [Arduino Nano 33 BLE Sense](https://store.arduino.cc/usa/nano-33-ble-sense-with-headers):  

- The Arduino Nano 33 BLE Sense comes with an **IMU (Inertial Measurement Unit)**, which is essential for gesture recognition. This sensor measures acceleration, orientation, and gyration, crucial for detecting motion and changes in orientation.  
- **Gesture recognition** involves processing sensor data in real-time and providing immediate feedback. The board uses an **RGB LED** to give visual feedback to the user upon recognizing gestures. Our board can detect three specific gestures:  
  - **WING**  
  - **RING**   
  - **SLOPE**

## Directory Structure

The project report repository is organized as follows:

[report](./report)
- Report PDF: [MagicWandPDF](./report/System/edgecomputer/MagicWand.pdf)
- Main tex file: [MagicWand](./report/System/edgecomputer/MagicWand.tex)

The report image repository is organized as follows:

[images](./report/Images)
- Hardware Description: [HardwareDescription](./report/System/edgecomputer/Images/HardwareDescription)
- Software Description: [SoftwareDescription](./report/System/edgecomputer/Images/SoftwareDescription)
- KDD: [KDD](./report/Images/KDD)
- Data Mining: [Data Mining](./report/System/edgecomputer/Images/DataMining)
- Programming: [Programming](./report/System/edgecomputer/Images/Programming)
- Bill of Materials: [Bill of Materials](./report/System/edgecomputer/Images/BillofMaterials)
- Results: [Results](./report/System/edgecomputer/Images/Results)

The Literature repository is located as follows:

[Literature](./Documents)
- MyLiterature bib file: [MyLiteratureBib](./Documents/MyLiterature.bib)
- Literature Review PDF: [MyLiteraturePDF](./Documents/LiteraturePresentation/LiteratureReview.pdf)
- Literature Review tex file: [MyLiterature](./Documents/LiteraturePresentation/LiteratureReview.tex)

Our Final Project Presentation is located as follows:

- Project Presentation PDF: [MagicWandPDF](./Presentations/MagicWand/MagicWand.pdf)
- Project Presentation Tex file: [MagicWand](./Presentations/MagicWand/MagicWand.tex)

The Poster is located as follows:

- Poster PDF: [MagicWandPosterPDF](./Poster/MagicWand.pdf)
- Poster Tex file: [MagicWandPoster](./Poster/MagicWand.tex)

The Manual is located as follows:

- Manual PDF: [MagicWandManualPDF](./manual/MagicWandManual.pdf)
- Main Tex file: [MagicWandManual](./manual/MagicWandManual.tex)

The Arduino IDE Code is located as follows:

- Magic Wand Code: [MagicWandCode](./Sourcecode/Code/Arduino/magicWandLed)
-
The Data Training Code is located as follows:

- Tests: [DataTrainingTests](./Sourcecode/Code/Datatraining/Tests)
- Modules: [DataTrainingModules](./Sourcecode/Code/Datatraining/Modules)
- ErrorHandler: [DataTrainingErrorHandler](./Sourcecode/Code/Datatraining/Errorhandler)

The requirements.txt is located as follows:

- Requirements.txt file: [Requirements](./Sourcecode/Requirements.txt)










