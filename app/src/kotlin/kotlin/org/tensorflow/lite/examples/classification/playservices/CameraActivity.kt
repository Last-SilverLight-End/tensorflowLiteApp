/*
 * Copyright 2022 The TensorFlow Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.classification.playservices

import android.Manifest
import android.annotation.SuppressLint
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.View.GONE
import android.view.View.VISIBLE
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.google.android.gms.tasks.Task
import com.google.android.gms.tasks.Tasks
import com.google.android.gms.tflite.client.TfLiteInitializationOptions
import com.google.android.gms.tflite.java.TfLite
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit
import kotlin.random.Random
import org.tensorflow.lite.examples.classification.playservices.databinding.ActivityCameraBinding
import java.lang.Thread.sleep

/** Activity that displays the camera and performs object detection on the incoming frames */
class CameraActivity : AppCompatActivity() {

  private lateinit var activityCameraBinding: ActivityCameraBinding

  private lateinit var bitmapBuffer: Bitmap

  private val executor = Executors.newSingleThreadExecutor()
  private val permissions = listOf(Manifest.permission.CAMERA)
  private val permissionsRequestCode = Random.nextInt(0, 10000)

  private var lensFacing: Int = CameraSelector.LENS_FACING_BACK
  private val isFrontFacing
    get() = lensFacing == CameraSelector.LENS_FACING_FRONT

  private var pauseAnalysis = false
  private var imageRotationDegrees: Int = 0
  private var changeInfo : Int = 0
  private var useGpu = false;

  // Initialize TFLite once. Must be called before creating the classifier
  private val initializeTask: Task<Void> by lazy {
    TfLite.initialize(
      this,
      TfLiteInitializationOptions.builder()
        .setEnableGpuDelegateSupport(true)
        .build()
    ).continueWithTask { task ->
        if (task.isSuccessful) {
          useGpu = true;
          return@continueWithTask Tasks.forResult(null)
        } else {
          // Fallback to initialize interpreter without GPU
          return@continueWithTask TfLite.initialize(this)
        }
      }
      .addOnFailureListener {
        Log.e(TAG, "TFLite in Play Services failed to initialize.", it)
      }
  }
  private var classifier: ImageClassificationHelper? = null

  private fun View.visibilityChanged(action: (View) -> Unit) {
    this.viewTreeObserver.addOnGlobalLayoutListener {
      val newVis: Int = this.visibility
      if (this.tag as Int? != newVis) {
        this.tag = this.visibility

        // visibility has changed
        action(this)
      }
    }
  }

  @SuppressLint("SetTextI18n")
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    activityCameraBinding = ActivityCameraBinding.inflate(layoutInflater)
    setContentView(activityCameraBinding.root)

    // Initialize TFLite asynchronously
    initializeTask
      .addOnSuccessListener {
        Log.d(TAG, "TFLite in Play Services initialized successfully.")
        classifier = ImageClassificationHelper(this, MAX_REPORT, useGpu)
      }
// isenabled 를 true 로 , pauseAnalysis를 true 로 , view visible 를 true로

    // pauseAnalysis는 항상 true 로 이미지 나오는건 이건 뭐 수정하면 되고

    activityCameraBinding.cameraCaptureButton.visibilityChanged { view ->
      when (view.visibility) {
        VISIBLE ->  {
          activityCameraBinding.textPrediction.text = "ID : 1 Name: Model of serving"
          changeInfo = 0
        /* Do something here */ }
        GONE -> {
          activityCameraBinding.textPrediction.text = "UNKNOWN"
        /* or here */ }
      }

    }

    activityCameraBinding.cameraCaptureButton.setOnClickListener {
      // Disable all camera controls
      it.isEnabled = true

      if (changeInfo == 0 ) {
        // If image analysis is in paused state, resume it
        activityCameraBinding.textPrediction.text = "Name : Model of serving material stone Construction \n " +
                "Date : 2010.05.20 User view ranking : 9 " +
                "\n Description: The founding spirit of Sejong University is virtue, creativity, service and practice." +
                "\n To sum it up, it is creative service. \n" +
                " True service is serving all people humbly. \n" +
                " Jesus himself set the example by washing the feet of his disciples."
        changeInfo =1
        //pauseAnalysis = false
      }
      else if(changeInfo==1)
      {
        activityCameraBinding.textPrediction.text = "ID : 1 Name: Model of serving"
        changeInfo =0
      }
      // Re-enable camera controls
      it.isEnabled = true
    }
  }

  override fun onDestroy() {
    // Terminate all outstanding analyzing jobs (if there is any).
    executor.apply {
      shutdown()
      awaitTermination(1000, TimeUnit.SECONDS)
    }
    // Release TFLite resources
    classifier?.close()
    super.onDestroy()
  }

  /** Declare and bind preview and analysis use cases */
  @SuppressLint("UnsafeExperimentalUsageError")
  private fun bindCameraUseCases() =
    activityCameraBinding.viewFinder.post {
      val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
      cameraProviderFuture.addListener(
        {
          // Camera provider is now guaranteed to be available
          val cameraProvider = cameraProviderFuture.get()

          // Set up the view finder use case to display camera preview
          val preview =
            Preview.Builder()
              .setTargetAspectRatio(AspectRatio.RATIO_4_3)
              .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
              .build()

          // Set up the image analysis use case which will process frames in real time
          val imageAnalysis =
            ImageAnalysis.Builder()
              .setTargetAspectRatio(AspectRatio.RATIO_4_3)
              .setTargetRotation(activityCameraBinding.viewFinder.display.rotation)
              .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
              .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
              .build()

          var frameCounter = 0
          var lastFpsTimestamp = System.currentTimeMillis()

          imageAnalysis.setAnalyzer(
            executor,
            ImageAnalysis.Analyzer { image ->
              if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                // the analyzer has started running
                imageRotationDegrees = image.imageInfo.rotationDegrees
                bitmapBuffer =
                  Bitmap.createBitmap(image.width, image.height, Bitmap.Config.ARGB_8888)
              }

              // Early exit: image analysis is in paused state, or TFLite is not initialized
              if (pauseAnalysis || classifier == null) {
                image.close()
                return@Analyzer
              }

              // Copy out RGB bits to our shared buffer
              image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

              // Perform the image classification for the current frame
              val recognitions = classifier?.classify(bitmapBuffer, imageRotationDegrees)

              reportRecognition(recognitions)

              // Compute the FPS of the entire pipeline
              val frameCount = 1000
              if (++frameCounter % frameCount == 0) {
                frameCounter = 0
                val now = System.currentTimeMillis()
                val delta = now - lastFpsTimestamp
                val fps = 100000 * frameCount.toFloat() / delta
                Log.d(TAG, "FPS: ${"%.02f".format(fps)}")
                lastFpsTimestamp = now
              }
            }
          )

          // Create a new camera selector each time, enforcing lens facing
          val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

          // Apply declared configs to CameraX using the same lifecycle owner
          cameraProvider.unbindAll()
          cameraProvider.bindToLifecycle(
            this as LifecycleOwner,
            cameraSelector,
            preview,
            imageAnalysis
          )

          // Use the camera object to link our preview use case with the view
          preview.setSurfaceProvider(activityCameraBinding.viewFinder.surfaceProvider)
        },
        ContextCompat.getMainExecutor(this)
      )
    }

  /** Displays recognition results on screen. */
  private fun reportRecognition(
    /** 여기에서 리스트롤 뽑아내고 있지만 이걸 따로 뽑아 내야 한다.*/



    recognitions: List<ImageClassificationHelper.Recognition>?,
  ) =
    activityCameraBinding.viewFinder.post {

      // Early exit: if recognition is null, or there are not enough recognition results.
      if (recognitions == null || recognitions.size < MAX_REPORT) {
        activityCameraBinding.textPrediction.visibility = View.GONE
        activityCameraBinding.cameraCaptureButton.visibility = View.GONE
        return@post
      }

      // Update the text and UI
      activityCameraBinding.textPrediction.text =
        recognitions.subList(0, MAX_REPORT).joinToString(separator = "\n") {
          /**if(button touched)*/
          // 만약 일정 정도를 넘기면 이렇게 된다.
          if(it.confidence >=0.6f)
          {
            activityCameraBinding.cameraCaptureButton.visibility = View.VISIBLE
            //activityCameraBinding.textPrediction.text = "1"

            "%s".format(activityCameraBinding.textPrediction.text)
            //"ID : 1 Name: Model of serving"
          }
            else {
            activityCameraBinding.cameraCaptureButton.visibility = View.GONE
            "%s".format(activityCameraBinding.textPrediction.text)
            //"${"%.2f".format(it.confidence)} ${it.title}"
            //"Name : Model of serving material stone Construction Date : 2010.05.20 User view ranking : 9 Description: The founding spiriti of Sejong University is virtue, creativity, service and practice. To sum it up, it is creative service. True service is serving all people humbly. Jesus himself set the example by washing the feet of his disciples."
          }
        }

      // Make sure all UI elements are visible
      activityCameraBinding.textPrediction.visibility = View.VISIBLE


    }

  override fun onResume() {
    super.onResume()

    // Request permissions each time the app resumes, since they can be revoked at any time
    if (!hasPermissions(this)) {
      ActivityCompat.requestPermissions(this, permissions.toTypedArray(), permissionsRequestCode)
    } else {
      bindCameraUseCases()
    }
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<out String>,
    grantResults: IntArray,
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)
    if (requestCode == permissionsRequestCode && hasPermissions(this)) {
      bindCameraUseCases()
    } else {
      finish() // If we don't have the required permissions, we can't run
    }
  }

  /** Convenience method used to check if all permissions required by this app are granted */
  private fun hasPermissions(context: Context) =
    permissions.all {
      ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED
    }

  companion object {
    private val TAG = CameraActivity::class.java.simpleName
    private const val MAX_REPORT = 1
  }
}
