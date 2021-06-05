@echo ---------- Moving old TFLite file into Android app ----------
copy /y original_model.tflite %~dp0..\DeepTricorderApp\models\src\main\assets\model.tflite

@echo ---------- Building original app ----------
cd %~dp0..\DeepTricorderApp
call gradlew assembleSupportDebug

@echo ---------- Installing original app ----------
call adb install app\build\outputs\apk\support\debug\app-support-debug.apk

@echo ---------- Done! You may exit and start the app! ----------
PAUSE