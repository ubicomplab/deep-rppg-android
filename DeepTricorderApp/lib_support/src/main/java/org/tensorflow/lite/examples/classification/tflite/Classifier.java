/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;
import android.content.ClipData;
import android.content.ClipboardManager;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.os.Environment;
import android.os.SystemClock;
import android.os.Trace;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintWriter;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import java.util.Map;
import java.util.TreeMap;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;
import org.tensorflow.lite.nnapi.NnApiDelegate;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.image.ops.ResizeOp.ResizeMethod;
import org.tensorflow.lite.support.image.ops.Rot90Op;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import uk.me.berndporr.iirj.*;

/** A classifier specialized to label images using TensorFlow Lite. */
public abstract class Classifier {
  public static final String TAG = "ClassifierWithSupport";

  /** The runtime device type used for executing classification. */
  public enum Device {
    CPU,
    NNAPI,
    GPU
  }

  public enum RecordingState {
    NEUTRAL,
    STARTED,
    FINISHED
  }

  /** Image size along the x axis. */
  private final int imageSizeX;

  /** Image size along the y axis. */
  private final int imageSizeY;

  /** Optional GPU delegate for accleration. */
  private GpuDelegate gpuDelegate = null;

  /** Optional NNAPI delegate for accleration. */
  private NnApiDelegate nnApiDelegate = null;

  /** An instance of the driver class to run model inference with Tensorflow Lite. */
  protected Interpreter tflite;

  /** Options for configuring the Interpreter. */
  private final Interpreter.Options tfliteOptions = new Interpreter.Options();

  /** Input image TensorBuffer. */
  private TensorImage inputImageBuffer;

  /** Input images TensorBuffer. */
  private TensorBuffer inputBuffer;

  /** Output TensorBuffer. */
  private TensorBuffer outputBuffer;

  /** Processor to apply post processing of the output probability. */
  private final TensorProcessor output1Processor;

  private Activity mActivity;

  Butterworth butterworthPulse;
  Butterworth butterworthBreath;

  private RecordingState recordingState;

  private float cumSum;
  private float[] prevSum;
  private final int MOVING_AVG_LEN = 10;
  private int numSum;

  /**
   * Creates a classifier with the provided configuration.
   *
   * @param activity The current Activity.
   * @param numThreads The number of threads to use for classification.
   * @return A classifier with the desired configuration.
   */
  public static Classifier create(Activity activity, int numThreads)
      throws IOException {
    try {
      return new ClassifierTSCAN(activity, Device.CPU, numThreads);
    } catch (Exception e) {
      throw new IOException();
    }
  }

  /** Initializes a {@code Classifier}. */
  protected Classifier(Activity activity, int numThreads) throws IOException {
    Log.d("Anand", "Starting");
    MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(activity, getModelPath());
    Log.d("Anand", "Loaded model");
    tfliteOptions.setNumThreads(numThreads);
    Log.d("Anand", "SetNumThreads");
    tflite = new Interpreter(tfliteModel, tfliteOptions);
    Log.d("Anand", "Loaded interpreter");

    // Reads type and shape of input and output tensors, respectively.
    tflite.allocateTensors();
    Log.d("Anand", "Allocated tensors");
    int[] inputShape = tflite.getInputTensor(tflite.getInputIndex("input_2")).shape();
    int[] outputShape = tflite.getOutputTensor(tflite.getOutputIndex("Identity")).shape();

    imageSizeY = inputShape[1];
    imageSizeX = inputShape[2];
    DataType imageDataType = tflite.getInputTensor(tflite.getInputIndex("input_2")).dataType();
    DataType outputDataType = tflite.getOutputTensor(tflite.getOutputIndex("Identity")).dataType();

    Log.d("Anand", "Got shapes");

    // Creates the input tensor.
    inputImageBuffer = new TensorImage(imageDataType);
    inputBuffer = TensorBuffer.createFixedSize(inputShape, imageDataType);

    // Creates the output tensors and its processors.
    outputBuffer = TensorBuffer.createFixedSize(outputShape, outputDataType);

    // Creates the post processors for the outputs.
    output1Processor = new TensorProcessor.Builder().build();

    Log.d("Anand", "Created tensors");

    mActivity = activity;

    butterworthPulse = new Butterworth();
    butterworthPulse.bandPass(1,30,1.625,1.75);

    butterworthBreath = new Butterworth();
    butterworthBreath.bandPass(1,30,0.29,0.48);

    recordingState = RecordingState.NEUTRAL;

    cumSum = 0;
    prevSum = new float[MOVING_AVG_LEN];
    for (int i = 0; i < MOVING_AVG_LEN; i++) {
      prevSum[i] = 0;
    }
    numSum = 0;
    Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
  }

  /** Runs inference and returns the classification results. */
  public float[] recognizeImage(final Bitmap[] bitmapBuffer, boolean isRecording) {
    // Logs this method so that it can be analyzed with systrace.
    Trace.beginSection("recognizeImage");
    Trace.beginSection("loadImage");
    double[][] dXsub = new double[20][36 * 36 * 3];
    Log.d("Thyme" ,"Resize: " + SystemClock.uptimeMillis());
    long startTimeForLoadImages = SystemClock.uptimeMillis();
    for(int i = 0; i < bitmapBuffer.length; i++) {
      inputImageBuffer = loadImage(bitmapBuffer[i], i);
      float[] imageArray = inputImageBuffer.getTensorBuffer().getFloatArray();
      for (int j = 0; j < 3888; j++) {
        dXsub[i][j] = (double) imageArray[j];
      }
    }

    Log.d("Thyme" ,"Normalize appearance: " + SystemClock.uptimeMillis());
    // skip normalizing appearance

    Log.d("Thyme" ,"Normalize motion: " + SystemClock.uptimeMillis());
    // normalize motion
    double[][] dXSubFrames = new double[20][36 * 36 * 3];
    for(int i = 0; i < 19; i++) {
      for (int j = 0; j < 3888; j++) {
        double divisor = (dXsub[i+1][j] + dXsub[i][j]);
        if(divisor == 0) {
          divisor = 1;
        }
        dXSubFrames[i][j] = (dXsub[i+1][j] - dXsub[i][j]) / divisor;
      }
    }

    double[] dXsubSerialized = new double[20 * 36 * 36 * 3];
    for(int i = 0; i < dXsubSerialized.length; i++) {
      dXsubSerialized[i] = dXSubFrames[i / 3888][i % 3888];
    }

    double dXmean = calcMean(dXsubSerialized);
    double dXstd = calcSTD(dXsubSerialized, dXmean);

    for(int i = 0; i < dXsubSerialized.length; i++) {
      dXsubSerialized[i] = dXsubSerialized[i] / dXstd;
    }

    Log.d("Thyme" ,"Infer vitals: " + SystemClock.uptimeMillis());

    inputBuffer.loadArray(convertDoubleToFloatArray(dXsubSerialized));

    // Prepare for inference
    long endTimeForLoadImages = SystemClock.uptimeMillis();
    Trace.endSection();
    Log.v(TAG, "Timecost preprocess images: " + (endTimeForLoadImages - startTimeForLoadImages));

    // Runs the inference call.
    Trace.beginSection("runInference");
    long startTimeForReference = SystemClock.uptimeMillis();
    tflite.run(inputBuffer.getBuffer(), outputBuffer.getBuffer().rewind());
    long endTimeForReference = SystemClock.uptimeMillis();
    Trace.endSection();
    Log.v(TAG, "Timecost to run model inference: " + (endTimeForReference - startTimeForReference));

    Trace.endSection();
    Log.d("Thyme" ,"Post-process: " + SystemClock.uptimeMillis());

    TensorBuffer pulseOutput = output1Processor.process(outputBuffer);

    float[] results = new float[20];

    float[] pulseFloats = pulseOutput.getFloatArray();

    float[] cumPulse = new float[20];

    for(int i = 0; i < 20; i++) {
      cumSum += pulseFloats[i];
      if (numSum < MOVING_AVG_LEN) {
        prevSum[numSum] = cumSum;
        numSum++;
      } else {
        // Shift values
        for(int j = 0; j < MOVING_AVG_LEN - 1; j++) {
          prevSum[j] = prevSum[j + 1];
        }
        prevSum[MOVING_AVG_LEN - 1] = cumSum;
      }

      // Calculate moving average
      float average = 0;
      for(int j = 0; j < MOVING_AVG_LEN - 1; j++) {
        average += prevSum[j] / numSum;
      }

      cumPulse[i] = (float) butterworthPulse.filter(cumSum - average);
      results[i] = cumPulse[i];
//      results[i+20] = (float) butterworthBreath.filter(breathFloats[i]);
    }

    if (recordingState == RecordingState.NEUTRAL) {
      if (isRecording) {
        // Begin recording
        beginRecording("pulseLog", pulseFloats);
        recordingState = RecordingState.STARTED;
      } else {
        // Do nothing
      }
      Log.d("Recording" ,"NEUTRAL");
    } else if (recordingState == RecordingState.STARTED) {
      if (isRecording) {
        // Continue logging
        continueRecording("pulseLog", pulseFloats);
      } else {
        recordingState = RecordingState.FINISHED;
      }
      Log.d("Recording" ,"STARTED");
    } else if (recordingState == RecordingState.FINISHED) {
      Toast.makeText(mActivity, "Recording saved and copied to clipboard!",
              Toast.LENGTH_SHORT).show();
      String recording = readRecording("pulseLog");
      ClipboardManager clipboard = (ClipboardManager) mActivity.getSystemService(Context.CLIPBOARD_SERVICE);
      ClipData clip = ClipData.newPlainText("Pulse Data", recording);
      clipboard.setPrimaryClip(clip);

      if (isRecording) {
        recordingState = RecordingState.STARTED;
      } else {
        recordingState = RecordingState.NEUTRAL;
      }
      Log.d("Recording" ,"FINISHED");
    }

    Log.d("Thyme" ,"Return: " + SystemClock.uptimeMillis());
    return results;
  }

  /** Closes the interpreter and model to release resources. */
  public void close() {
    if (tflite != null) {
      tflite.close();
      tflite = null;
    }
    if (gpuDelegate != null) {
      gpuDelegate.close();
      gpuDelegate = null;
    }
    if (nnApiDelegate != null) {
      nnApiDelegate.close();
      nnApiDelegate = null;
    }
  }

  /** Get the image size along the x axis. */
  public int getImageSizeX() {
    return imageSizeX;
  }

  /** Get the image size along the y axis. */
  public int getImageSizeY() {
    return imageSizeY;
  }

  /** Loads input image, and applies preprocessing. */
  private TensorImage loadImage(final Bitmap bitmap, int i) {
    // TESTING CODE
//    try {
//        saveImage("frame_" + i, bitmap, mActivity, "Pictures");
//    } catch (Exception e) {
//        e.printStackTrace();
//    }

    // Loads bitmap into a TensorImage.
    inputImageBuffer.load(bitmap);

    int numRotation = 1;
    ImageProcessor imageProcessor =
        new ImageProcessor.Builder()
            .add(new ResizeOp(imageSizeX, imageSizeY, ResizeMethod.NEAREST_NEIGHBOR))
            .add(new Rot90Op(numRotation))
            .build();
    return imageProcessor.process(inputImageBuffer);
  }

  /** Gets the name of the model file stored in Assets. */
  protected abstract String getModelPath();

  private double calcMean(double[] a) {
    double sum = 0;
    for(int i = 0; i < a.length; i++) {
      sum += a[i] / a.length;
    }
    return sum;
  }

  private double calcSTD(double[] a, double mean) {
    double squareDiff = 0;
    for(int i = 0; i < a.length; i++) {
      squareDiff += Math.pow(a[i] - mean, 2)/ a.length;
    }
    return Math.sqrt(squareDiff);
  }

  private float[] convertDoubleToFloatArray(double[] doubleArray) {
    float[] floatArray = new float[doubleArray.length];
    for (int i = 0 ; i < doubleArray.length; i++)
    {
      floatArray[i] = (float) doubleArray[i];
    }
    return floatArray;
  }

  // modded from https://stackoverflow.com/questions/8330276/write-a-file-in-external-storage-in-android
  private void writeToSDFile(String fileName, String data){

    // Find the root of the external storage.
    // See http://developer.android.com/guide/topics/data/data-  storage.html#filesExternal
    File root = android.os.Environment.getExternalStorageDirectory();

    // See http://stackoverflow.com/questions/3551821/android-write-to-sd-card-folder
    File dir = new File (root.getAbsolutePath() + "/download");
    dir.mkdirs();
    File file = new File(dir, fileName + ".txt");

    try {
      FileOutputStream f = new FileOutputStream(file);
      PrintWriter pw = new PrintWriter(f);
      pw.println(data);
      pw.flush();
      pw.close();
      f.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      Log.i("File", "******* File not found. Did you" +
              " add a WRITE_EXTERNAL_STORAGE permission to the   manifest?");
    } catch (IOException e) {
      e.printStackTrace();
    }
    Log.d("File", "\n\nFile written to "+file);
  }

  private void beginRecording(String filename, float[] x) {
    String xString = Arrays.toString(x);
    xString = xString.substring(1, xString.length() - 1);

    // Find the root of the external storage.
    // See http://developer.android.com/guide/topics/data/data-  storage.html#filesExternal
    File root = android.os.Environment.getExternalStorageDirectory();

    // See http://stackoverflow.com/questions/3551821/android-write-to-sd-card-folder
    File dir = new File (root.getAbsolutePath() + "/download");
    dir.mkdirs();
    File file = new File(dir, filename + ".txt");

    try {
      FileOutputStream f = new FileOutputStream(file);
      PrintWriter pw = new PrintWriter(f);
      pw.print(xString);
      pw.flush();
      pw.close();
      f.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      Log.i("File", "******* File not found. Did you" +
              " add a WRITE_EXTERNAL_STORAGE permission to the   manifest?");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private void continueRecording(String filename, float[] x) {
    String xString = Arrays.toString(x);
    xString = ", " + xString.substring(1, xString.length() - 1);

    // Find the root of the external storage.
    // See http://developer.android.com/guide/topics/data/data-  storage.html#filesExternal
    File root = android.os.Environment.getExternalStorageDirectory();

    // See http://stackoverflow.com/questions/3551821/android-write-to-sd-card-folder
    File dir = new File (root.getAbsolutePath() + "/download");
    dir.mkdirs();
    File file = new File(dir, filename + ".txt");

    try {
      FileOutputStream f = new FileOutputStream(file, true);
      PrintWriter pw = new PrintWriter(f);
      pw.print(xString);
      pw.flush();
      pw.close();
      f.close();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
      Log.i("File", "******* File not found. Did you" +
              " add a WRITE_EXTERNAL_STORAGE permission to the   manifest?");
    } catch (IOException e) {
      e.printStackTrace();
    }
  }

  private String readRecording(String filename) {
    BufferedReader reader = null;
    StringBuilder sb = new StringBuilder();
    try {
      File root = android.os.Environment.getExternalStorageDirectory();
      File dir = new File (root.getAbsolutePath() + "/download");
      dir.mkdirs();
      File file = new File(dir, filename + ".txt");

      reader = new BufferedReader(new FileReader(file));

      // do reading, usually loop until end of file reading
      String mLine;
      while ((mLine = reader.readLine()) != null) {
        //process line
        sb.append(mLine);
      }
    } catch (IOException e) {
      //log the exception
    } finally {
      if (reader != null) {
        try {
          reader.close();
        } catch (IOException e) {
          //log the exception
        }
      }
    }
    return sb.toString();
  }

  // TESTING CODE
  private void writeDoubleArrayToSDFile(String filename, double[] x) {
    String xString = Arrays.toString(x);
    xString = xString.substring(1, xString.length() - 1);
    writeToSDFile(filename, xString);
  }

  private void writeFloatArrayToSDFile(String filename, float[] x) {
    String xString = Arrays.toString(x);
    xString = xString.substring(1, xString.length() - 1);
    writeToSDFile(filename, xString);
  }

  private double[] getDoubleArrayFromString(String sbOut) {
    String[] sbSplit = sbOut.split(",");
    double[] doubleArray = new double[sbSplit.length];
    for(int i = 0; i < sbSplit.length; i++) {
      doubleArray[i] = Double.parseDouble(sbSplit[i]);
    }

    return doubleArray;
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
}



