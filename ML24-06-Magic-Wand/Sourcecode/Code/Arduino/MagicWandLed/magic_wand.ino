/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <TensorFlowLite.h>

#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "magic_wand_model_data.h"
#include "rasterize_stroke.h"
#include "imu_provider.h"

#define BLE_SENSE_UUID(val) ("4798e0f2-" val "-4d68-af64-8a8f5258404e")

namespace {

  const int VERSION = 0x00000000;

  // Constants for image rasterization
  constexpr int raster_width = 32;
  constexpr int raster_height = 32;
  constexpr int raster_channels = 3;
  constexpr int raster_byte_count = raster_height * raster_width * raster_channels;
  int8_t raster_buffer[raster_byte_count];

  // BLE settings
  BLEService        service                       (BLE_SENSE_UUID("0000"));
  BLECharacteristic strokeCharacteristic          (BLE_SENSE_UUID("300a"), BLERead, stroke_struct_byte_count);
  
  // String to calculate the local and device name
  String name;
  
  // Create an area of memory to use for input, output, and intermediate arrays.
  constexpr int kTensorArenaSize = 30 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];
  
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  
  constexpr int label_count = 3;
  const char* labels[label_count] = {"W", "0", "L"};

  // Define LED pins
  const int redLedPin = 2;    // Pin for red LED
  const int greenLedPin = 3;  // Pin for green LED
  const int blueLedPin = 4;   // Pin for blue LED

  // Function to turn off all LEDs
  void turnOffLeds() {
    digitalWrite(redLedPin, LOW);
    digitalWrite(greenLedPin, LOW);
    digitalWrite(blueLedPin, LOW);
  }

  // Function to light up the appropriate LED based on the gesture
  void lightUpLed(const char* gesture) {
    turnOffLeds();
    if (strcmp(gesture, "W") == 0) {
      digitalWrite(redLedPin, HIGH);  // Red light for "W" gesture
    } else if (strcmp(gesture, "L") == 0) {
      digitalWrite(blueLedPin, HIGH); // Blue light for "L" gesture
    } else if (strcmp(gesture, "0") == 0) {
      digitalWrite(greenLedPin, HIGH); // Green light for "0" gesture
    }
  }

}  // namespace

void setup() {
  // Start serial
  Serial.begin(9600);
  Serial.println("Started");

  // Start IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialized IMU!");
    while (1);
  }
  SetupIMU();

  // Start BLE
  if (!BLE.begin()) {
    Serial.println("Failed to initialized BLE!");
    while (1);
  }
  String address = BLE.address();

  // Output BLE settings over Serial
  Serial.print("address = ");
  Serial.println(address);

  address.toUpperCase();

  name = "BLESense-";
  name += address[address.length() - 5];
  name += address[address.length() - 4];
  name += address[address.length() - 2];
  name += address[address.length() - 1];

  Serial.print("name = ");
  Serial.println(name);

  BLE.setLocalName(name.c_str());
  BLE.setDeviceName(name.c_str());
  BLE.setAdvertisedService(service);

  service.addCharacteristic(strokeCharacteristic);

  BLE.addService(service);
  BLE.advertise();

  // Initialize LED pins
  pinMode(redLedPin, OUTPUT);
  pinMode(greenLedPin, OUTPUT);
  pinMode(blueLedPin, OUTPUT);
  turnOffLeds(); // Ensure all LEDs are off initially

  static tflite::MicroErrorReporter micro_error_reporter;  // NOLINT
  error_reporter = &micro_error_reporter;

  model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<4> micro_op_resolver;  // NOLINT
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddMean();
  micro_op_resolver.AddFullyConnected();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  interpreter->AllocateTensors();

  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != raster_height) ||
      (model_input->dims->data[2] != raster_width) ||
      (model_input->dims->data[3] != raster_channels) ||
      (model_input->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad input tensor parameters in model");
    return;
  }

  TfLiteTensor* model_output = interpreter->output(0);
  if ((model_output->dims->size != 2) || (model_output->dims->data[0] != 1) ||
      (model_output->dims->data[1] != label_count) ||
      (model_output->type != kTfLiteInt8)) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Bad output tensor parameters in model");
    return;
  }
}

void loop() {
  BLEDevice central = BLE.central();
  static bool was_connected_last = false;  
  if (central && !was_connected_last) {
    Serial.print("Connected to central: ");
    Serial.println(central.address());
  }
  was_connected_last = central;

  const bool data_available = IMU.accelerationAvailable() || IMU.gyroscopeAvailable();
  if (!data_available) {
    return;
  }
  int accelerometer_samples_read;
  int gyroscope_samples_read;
  ReadAccelerometerAndGyroscope(&accelerometer_samples_read, &gyroscope_samples_read);

  bool done_just_triggered = false;
  if (gyroscope_samples_read > 0) {
    EstimateGyroscopeDrift(current_gyroscope_drift);
    UpdateOrientation(gyroscope_samples_read, current_gravity, current_gyroscope_drift);
    UpdateStroke(gyroscope_samples_read, &done_just_triggered);
    if (central && central.connected()) {
      strokeCharacteristic.writeValue(stroke_struct_buffer, stroke_struct_byte_count);
    }
  }
  if (accelerometer_samples_read > 0) {
    EstimateGravityDirection(current_gravity);
    UpdateVelocity(accelerometer_samples_read, current_gravity);
  }

  if (done_just_triggered) {
    RasterizeStroke(stroke_points, *stroke_transmit_length, 0.6f, 0.6f, raster_width, raster_height, raster_buffer);
    for (int y = 0; y < raster_height; ++y) {
      char line[raster_width + 1];
      for (int x = 0; x < raster_width; ++x) {
        const int8_t* pixel = &raster_buffer[(y * raster_width * raster_channels) + (x * raster_channels)];
        const int8_t red = pixel[0];
        const int8_t green = pixel[1];
        const int8_t blue = pixel[2];
        char output;
        if ((red > -128) || (green > -128) || (blue > -128)) {
          output = '#';
        } else {
          output = '.';
        }
        line[x] = output;
      }
      line[raster_width] = 0;
      Serial.println(line);
    }

    TfLiteTensor* model_input = interpreter->input(0);
    for (int i = 0; i < raster_byte_count; ++i) {
      model_input->data.int8[i] = raster_buffer[i];
    }
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed");
      return;
    }
    TfLiteTensor* output = interpreter->output(0);

    int8_t max_score;
    int max_index;
    for (int i = 0; i < label_count; ++i) {
      const int8_t score = output->data.int8[i];
      if ((i == 0) || (score > max_score)) {
        max_score = score;
        max_index = i;
      }
    }
    const char* detected_gesture = labels[max_index];
    TF_LITE_REPORT_ERROR(error_reporter, "Found %s (%d)", detected_gesture, max_score);

    // Light up the appropriate LED
    lightUpLed(detected_gesture);
  }
}
