import numpy as np
import serial
import time

waitTime = 0.1

# generate the waveform table
signalLength = 246
song1 = np.array([
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261], dtype=np.int16)
note1 = np.array([
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2], dtype=np.int16)
song2 = np.array([
  392, 330, 300, 349, 294, 294,
  261, 294, 330, 349, 392, 392, 392,
  392, 330, 300, 349, 294, 294,
  261, 330, 392, 392, 330,
  294, 294, 294, 294, 330, 349,
  330, 330 ,330 ,330, 349, 392,
  392, 330, 300, 349, 294, 294,
  261, 330, 392, 392, 261], dtype=np.int16)
note2 = np.array([
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 2, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 3], dtype=np.int16)
song3 = np.array([
  261, 294, 330, 261, 261, 294, 330, 261,
  330, 349, 392, 330, 349, 392,
  392, 440, 392, 349, 330, 261,
  392, 440, 392, 349, 330, 261,
  294, 196, 261, 294, 196, 261], dtype=np.int16)
note3 = np.array([
  1, 1, 1, 1, 1, 1, 1, 1,
  1, 1, 2, 1, 1, 2,
  11, 11, 11, 11, 1, 1,
  11, 11, 11, 11, 1, 1,
  1, 1, 2, 1, 1, 2,], dtype=np.int16)

signalTable = np.append(song1, song2)
signalTable = np.append(signalTable, song3)
signalTable = np.append(signalTable, note1)
signalTable = np.append(signalTable, note2)
signalTable = np.append(signalTable, note3)

# output formatter
formatter = lambda x: "%3d" % x

# send the waveform table to K66F
serdev = '/dev/ttyACM4'
s = serial.Serial(serdev)
print("Sending signal ...")
print("It may take about %d seconds ..." % (int(signalLength * waitTime)))
for data in signalTable:
  s.write(bytes(formatter(data), 'UTF-8'))
  time.sleep(waitTime)
s.close()
print("Signal sended")