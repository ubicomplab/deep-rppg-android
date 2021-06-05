@echo ---------- Testing pulse sensor ----------
@echo OFF
set /p comPort=Enter COM port number of pulse sensor (e.g. 3 for COM3): 
CALL conda.bat activate tf-gpu
python test_sensor.py %comPort%

@echo ---------- Recorded in "pulse_test.png". ----------
PAUSE