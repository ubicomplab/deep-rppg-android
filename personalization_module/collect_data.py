import cv2, time
import serial
import matplotlib.pyplot as plt
import numpy as np
from skimage.util import img_as_float
import sys

# From https://www.youtube.com/watch?v=1XTqE7LFQjI
def webcam_demo():
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    t = 0
    while True:
        t = t + 1
        check, frame = video.read()
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    print(str(t) + "ms")
    video.release()
    cv2.destroyAllWindows()

# From https://problemsolvingwithpython.com/11-Python-and-External-Hardware/11.04-Reading-a-Sensor-with-Python/
# Adapted for use with the pulse sensor
def pulse_demo(portNum):
    # set up the serial line
    ser = serial.Serial('COM' + str(portNum), 9600)
    time.sleep(2)

    # Read and record the data
    data = []  # empty list to store the data
    for i in range(500):
        val = get_pulse_val(ser)
        print(val)
        data.append(val)  # add to the end of data list
        time.sleep(0.01)  # wait (sleep) 0.1 seconds

    ser.close()

    plt.plot(data)
    plt.xlabel('Time')
    plt.ylabel('Pulse Sensor Reading')
    plt.title('Pulse Reading vs. Time')
    plt.savefig("pulse_demo")

# Scrap code:
    # check, frame = video.read()
    # height, width, channels = frame.shape
    # print(str(height) + ", " + str(width) + ", " + str(channels))

def get_pulse_val(ser):
    b = ser.readline()  # read a byte string
    string = ''
    while string == '':
        b = ser.readline() # Make sure we get a value
        string_n = b.decode()  # decode byte string into Unicode
        string = string_n.rstrip()  # remove \n and \r
    flt = float(string)  # convert string to float
    return flt

def record_data(portNum):

    # Create empty lists
    frames = []
    ppg = []

    # Initialize capturing
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ser = serial.Serial('COM' + str(portNum), 9600)

    # Get dimensions
    check, frame = video.read()
    height, width, channels = frame.shape
    dim = 36

    time.sleep(2)
    while True:
        pulse_val = get_pulse_val(ser)
        check, frame = video.read()
        print(pulse_val)

        frames.append(frame)
        ppg.append(pulse_val)

        # Display webcam feed and exit when user is done recording
        cv2.imshow("Capturing. Press Q to quit.", frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()
    ser.close()

    for index, img in enumerate(frames):
        vidLxL = cv2.resize(img_as_float(img[:, int(width/2)-int(height/2 + 1):int(height/2)+int(width/2), :]), (dim, dim), interpolation = cv2.INTER_AREA)
        vidLxL = cv2.cvtColor(vidLxL .astype('float32'), cv2.COLOR_BGR2RGB)
        vidLxL[vidLxL > 1] = 1
        vidLxL[vidLxL < (1 / 255)] = 1 / 255
        frames[index] = vidLxL

    video_np = np.array(frames)
    ppg_np = np.array(ppg)
    np.save("video.npy", video_np)
    np.save("ppg.npy",ppg_np)

if __name__ == '__main__':
    # webcam_demo()
    # pulse_demo(sys.argv[1])
    record_data(sys.argv[1])
