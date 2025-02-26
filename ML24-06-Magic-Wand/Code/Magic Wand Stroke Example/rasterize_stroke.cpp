/**
 * @file rasterize_stroke.h
 * @brief Rasterization of a stroke into a 2D color image.
 *
 * This file contains functions to convert stroke (2D coordinates of a gesture)
 * into a 2D color image using fixed-point arithmetic.
 *
 * Copyright 2020 The TensorFlow Authors. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "rasterize_stroke.h"

namespace {
constexpr int kFixedPoint = 256; ///< Fixed-point precision multiplier.

/**
 * @brief Multiplies two fixed-point numbers.
 * @param a First operand.
 * @param b Second operand.
 * @return Result of (a * b) / kFixedPoint.
 */
int32_t MulFP(int32_t a, int32_t b) {
  return (a * b) / kFixedPoint;
}

/**
 * @brief Divides two fixed-point numbers.
 * @param a Numerator.
 * @param b Denominator (defaults to 1 if zero to avoid division by zero).
 * @return Result of (a * kFixedPoint) / b.
 */
int32_t DivFP(int32_t a, int32_t b) {
  if (b == 0) {
    b = 1;
  }
  return (a * kFixedPoint) / b;
}

/**
 * @brief Converts a floating-point number to fixed-point representation.
 * @param a Floating-point value.
 * @return Fixed-point representation.
 */
int32_t FloatToFP(float a) {
  return static_cast<int32_t>(a * kFixedPoint);
}

/**
 * @brief Normalizes a coordinate from a fixed-point range to screen coordinates.
 * @param a_fp Input coordinate in fixed-point.
 * @param range_fp Range in fixed-point.
 * @param half_size_fp Half of the screen size in fixed-point.
 * @return Normalized coordinate.
 */
int32_t NormToCoordFP(int32_t a_fp, int32_t range_fp, int32_t half_size_fp) {
  const int32_t norm_fp = DivFP(a_fp, range_fp);
  return MulFP(norm_fp, half_size_fp) + half_size_fp;
}

/**
 * @brief Rounds a fixed-point value to the nearest integer.
 * @param a Fixed-point value.
 * @return Rounded integer.
 */
int32_t RoundFPToInt(int32_t a) {
  return static_cast<int32_t>((a + (kFixedPoint / 2)) / kFixedPoint);
}

/**
 * @brief Clamps a value between min and max.
 * @param a Input value.
 * @param min Minimum bound.
 * @param max Maximum bound.
 * @return Clamped value.
 */
int32_t Gate(int32_t a, int32_t min, int32_t max) {
  if (a < min) {
    return min;
  } else if (a > max) {
    return max;
  } else {
    return a;
  }
}

/**
 * @brief Computes the absolute value of an integer.
 * @param a Input value.
 * @return Absolute value of a.
 */
int32_t Abs(int32_t a) {
  return (a > 0) ? a : -a;
}

}  // namespace

/**
 * @brief Rasterizes a stroke into a 2D color image.
 *
 * Converts a stroke (sequence of 2D points) into a rasterized image with color encoding.
 * The stroke is drawn as a sequence of connected line segments, with colors shifting
 * from red to green to blue over time.
 *
 * @param stroke_points Array of stroke points (x, y pairs) in normalized coordinates.
 * @param stroke_points_count Number of points in the stroke.
 * @param x_range Range of x values in normalized space.
 * @param y_range Range of y values in normalized space.
 * @param width Width of the output image in pixels.
 * @param height Height of the output image in pixels.
 * @param out_buffer Output buffer storing the rasterized image (3 channels per pixel).
 */
void RasterizeStroke(
    int8_t* stroke_points,
    int stroke_points_count,
    float x_range, 
    float y_range, 
    int width, 
    int height,
    int8_t* out_buffer) {
  constexpr int num_channels = 3;
  const int buffer_byte_count = height * width * num_channels;

  // Initialize all pixels to black
  for (int i = 0; i < buffer_byte_count; ++i) {
    out_buffer[i] = -128;
  }

  const int32_t width_fp = width * kFixedPoint;
  const int32_t height_fp = height * kFixedPoint;
  const int32_t half_width_fp = width_fp / 2;
  const int32_t half_height_fp = height_fp / 2;
  const int32_t x_range_fp = FloatToFP(x_range);
  const int32_t y_range_fp = FloatToFP(y_range);
  const int t_inc_fp = kFixedPoint / stroke_points_count;
  const int one_half_fp = (kFixedPoint / 2);

  for (int point_index = 0; point_index < (stroke_points_count - 1); ++point_index) {
    // Get start and end points of the current line segment
    const int8_t* start_point = &stroke_points[point_index * 2];
    const int32_t start_point_x_fp = (start_point[0] * kFixedPoint) / 128;
    const int32_t start_point_y_fp = (start_point[1] * kFixedPoint) / 128;
    const int8_t* end_point = &stroke_points[(point_index + 1) * 2];
    const int32_t end_point_x_fp = (end_point[0] * kFixedPoint) / 128;
    const int32_t end_point_y_fp = (end_point[1] * kFixedPoint) / 128;
    const int32_t start_x_fp = NormToCoordFP(start_point_x_fp, x_range_fp, half_width_fp);
    const int32_t start_y_fp = NormToCoordFP(-start_point_y_fp, y_range_fp, half_height_fp);
    const int32_t end_x_fp = NormToCoordFP(end_point_x_fp, x_range_fp, half_width_fp);
    const int32_t end_y_fp = NormToCoordFP(-end_point_y_fp, y_range_fp, half_height_fp);
    const int32_t delta_x_fp = end_x_fp - start_x_fp;
    const int32_t delta_y_fp = end_y_fp - start_y_fp;
    
    // Determine line color based on stroke progression
    const int32_t t_fp = point_index * t_inc_fp;
    int8_t red_i8, green_i8, blue_i8;
    if (t_fp < one_half_fp) {
      red_i8 = Gate(RoundFPToInt((kFixedPoint - DivFP(t_fp, one_half_fp)) * 255) - 128, -128, 127);
      green_i8 = Gate(RoundFPToInt(DivFP(t_fp, one_half_fp) * 255) - 128, -128, 127);
      blue_i8 = -128;
    } else {
      green_i8 = Gate(RoundFPToInt((kFixedPoint - DivFP(t_fp - one_half_fp, one_half_fp)) * 255) - 128, -128, 127);
      blue_i8 = Gate(RoundFPToInt(DivFP(t_fp - one_half_fp, one_half_fp) * 255) - 128, -128, 127);
      red_i8 = -128;
    }
  }
}
