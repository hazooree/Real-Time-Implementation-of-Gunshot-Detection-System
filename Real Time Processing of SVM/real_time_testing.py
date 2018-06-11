# -*- coding: utf-8 -*-
"""
Created on Thu May 10 23:37:09 2018

@author: zeeshan haider
"""
from IPython import get_ipython
get_ipython().magic('reset -sf') 
import pyaudio
import wave
import os
import librosa
import numpy as np
import cPickle

path="E:\\All Data\\study\\MS\\2\\machine learning\\Project\\background_dataset"
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 22050
RECORD_SECONDS =1
load_training = open("E:\\All Data\study\\MS\\2\\machine learning\\Project\\save_training_1.pickle",'rb')
clf = cPickle.load(load_training) # LOAD TRAINED CLASSIFIER
load_training.close()
for i in range(8000):
    WAVE_OUTPUT_FILENAME = "background"+".wav"
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(path+"\\"+WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    y, sr = librosa.load(path+"\\"+WAVE_OUTPUT_FILENAME,duration=1)
    S = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=20)
    S=np.reshape(S,np.product(S.shape))
    S=np.concatenate((S,S[840:860]))
    print clf.predict([S])    