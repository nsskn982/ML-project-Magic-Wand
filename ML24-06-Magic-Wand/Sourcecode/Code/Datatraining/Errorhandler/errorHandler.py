##
# @file errorHandler.py
#
# @brief The module contains a set of custom error functions tailored for the project.
# magicWandErrorHandler.py

def errorLoadDataset():
    raise RuntimeError("Failed to load the Magic Wand dataset. Check your data path and file formats.")

def errorProcessStrokes():
    raise RuntimeError("Error processing wand strokes. Ensure the wand gesture data is in the correct format.")

def errorCreateRaster():
    raise RuntimeError("Failed to create raster from wand strokes. Check the wand gesture data for anomalies.")

def errorTrainModel():
    raise RuntimeError("Error during training the Magic Wand model. Review the model architecture and dataset.")

def errorConvertToTFLite():
    raise RuntimeError("Error during conversion to TensorFlow Lite. Verify the model compatibility and conversion process.")

def errorDetectGesture():
    raise RuntimeError("Error during real-time gesture detection. Check the input data and model compatibility.")

def errorExportModel():
    raise RuntimeError("Error during exporting the Magic Wand model. Verify the export path and model information.")

def errorUnknown():
    raise RuntimeError("An unexpected error occurred in the Magic Wand project. Review the error messages.")

# Additional error handling functions can be added based on specific requirements.
