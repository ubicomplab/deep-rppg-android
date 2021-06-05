/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification;

import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.provider.MediaStore;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;

import org.tensorflow.lite.examples.classification.env.BorderedText;
import org.tensorflow.lite.examples.classification.env.Logger;
import org.tensorflow.lite.examples.classification.tflite.Classifier;

public class ClassifierActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final float TEXT_SIZE_DIP = 10;
  private Bitmap rgbFrameBitmap = null;
  private long lastProcessingTimeMs;
  private Integer sensorOrientation;
  private Classifier classifier;
  private BorderedText borderedText;
  /** Input image size of the model along x axis. */
  private int imageSizeX;
  /** Input image size of the model along y axis. */
  private int imageSizeY;

  // Collecting images in a buffer
  // as prototype, just perform inference every 100 frames
  private static final int CHUNK_SIZE = 20;
  private Bitmap[] bitmapBuffer = new Bitmap[CHUNK_SIZE];
  int frame_counter = 0;


  @Override
  protected int getLayoutId() {
    return R.layout.tfe_ic_camera_connection_fragment;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    recreateClassifier(getNumThreads());
    if (classifier == null) {
      LOGGER.e("No classifier on preview!");
      return;
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
  }

  protected ContentValues contentValues() {
    ContentValues values = new ContentValues();
    values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
    values.put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis() / 1000);
    values.put(MediaStore.Images.Media.DATE_TAKEN, System.currentTimeMillis());
    return values;
  }
  protected void saveImageToStream(Bitmap bitmap, OutputStream outputStream) {
    if (outputStream != null) {
      try {
        bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream);
        outputStream.close();
      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }

  @Override
  protected void processImage() {
    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
    final int cropSize = Math.min(previewWidth, previewHeight);
    int size = 400;
    int y = 370 - (size /2);
    int x = 240 - (size /2);

    Bitmap croppedBitmap = Bitmap.createBitmap(rgbFrameBitmap, y, x, size, size);
    // TESTING CODE
//    if(frame_counter == 0) {
//      try {
//        saveImage("adjusting_crop", croppedBitmap, getApplicationContext(), "Pictures");
//        System.exit(0);
//      } catch (Exception e) {
//        e.printStackTrace();
//      }
//    }

    bitmapBuffer[frame_counter] = croppedBitmap;
    Log.d("Bread", "Frame " + frame_counter + ": " + SystemClock.uptimeMillis());
    if (frame_counter == CHUNK_SIZE - 1) {
      frame_counter = 0;
      runInBackground(
              new Runnable() {
                @Override
                public void run() {
                    if (classifier != null) {
                      Log.d("Thyme", "Begin recognize: " + SystemClock.uptimeMillis());
                      Log.d("Bread", "Begin recognize: " + SystemClock.uptimeMillis());
                      final long startTime = SystemClock.uptimeMillis();
                      float[] results =
                              classifier.recognizeImage(bitmapBuffer.clone(), isRecording);
                      lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                      Log.d("Bread", "Finish recognize: " + SystemClock.uptimeMillis());
                      Log.d("Thyme", "Finish recognize: " + SystemClock.uptimeMillis());

                      runOnUiThread(
                              new Runnable() {
                                @Override
                                public void run() {
                                  Log.d("Live", Long.toString(SystemClock.uptimeMillis()));
                                  showResultsInBottomSheet(results);
                                }
                              });
                    }
                }
              });
    } else {
      frame_counter++;
    }
    readyForNextImage();
  }

  private void recreateClassifier(int numThreads) {
    if (classifier != null) {
      LOGGER.d("Closing classifier.");
      classifier.close();
      classifier = null;
    }
    try {
      classifier = Classifier.create(this, numThreads);
    } catch (IOException | IllegalArgumentException e) {
      LOGGER.e(e, "Failed to create classifier.");
      runOnUiThread(
          () -> {
            Toast.makeText(this, e.getMessage(), Toast.LENGTH_LONG).show();
          });
      return;
    }

    // Updates the input image size.
    imageSizeX = classifier.getImageSizeX();
    imageSizeY = classifier.getImageSizeY();
  }

  // Adapted from https://stackoverflow.com/questions/36624756/how-to-save-bitmap-to-android-gallery
  protected void saveImage(String fileName, Bitmap bitmap, Context context, String folderName) throws FileNotFoundException {
    // Since build version is less than 29, we're working with this, which is deprecated past 29:
    File directory = new File(Environment.getExternalStorageDirectory().toString() + File.separator + folderName);

    if (!directory.exists()) {
      directory.mkdirs();
    }
    File file = new File(directory, fileName + ".png");
    saveImageToStream(bitmap, new FileOutputStream(file));
    if (file.getAbsolutePath() != null) {
      ContentValues values = contentValues();
      values.put(MediaStore.Images.Media.DATA, file.getAbsolutePath());
      context.getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
    }
  }
}
