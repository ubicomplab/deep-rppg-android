import time
import serial
import matplotlib.pyplot as plt
import sys

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
    plt.savefig("pulse_test")

def get_pulse_val(ser):
    b = ser.readline()  # read a byte string
    string = ''
    while string == '':
        b = ser.readline() # Make sure we get a value
        string_n = b.decode()  # decode byte string into Unicode
        string = string_n.rstrip()  # remove \n and \r
    flt = float(string)  # convert string to float
    return flt

if __name__ == '__main__':
    pulse_demo(sys.argv[1])
