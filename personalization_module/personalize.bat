@echo ---------- Collecting new data ----------
set /p comPort=Enter COM port number of pulse sensor (e.g. 3 for COM3): 
CALL conda.bat activate tf-gpu
CALL python collect_data.py %comPort%

@echo ---------- Updating model weights with new data ----------
CALL conda.bat activate tf-gpu
python train_txt.py --video_path video.npy --label_path ppg.npy

@echo ---------- Generating new TFLite file ----------
CALL conda.bat activate tf-gpu
python convert_to_tflite.py

@echo ---------- Moving new TFLite file into Android app ----------
copy /y model.tflite %~dp0..\DeepTricorderApp\models\src\main\assets\model.tflite

@echo ---------- Building updated app ----------
cd %~dp0..\DeepTricorderApp
call gradlew assembleSupportDebug

@echo ---------- Installing updated app ----------
call adb install app\build\outputs\apk\support\debug\app-support-debug.apk

@echo ---------- Done! You may exit and start the app! ----------
PAUSE