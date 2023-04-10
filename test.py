import numpy as np
from utils.symbolic_aggregate_approximation import timeseries2symbol,convert_symbols
import matplotlib.pyplot as plt

def sax_test():

    np.random.seed(1)

    # Set the duration and frequency of the sine wave
    duration = 5  # seconds
    freq = 1  # Hz

    # Create a time vector
    t = np.linspace(0, duration, num=5000)

    # Create an amplitude vector that increases linearly with time
    amplitude = np.linspace(0, 1, num=5000)

    # Generate the sine wave
    y = amplitude * np.sin(2 * np.pi * freq * t)
    y1 = np.sin(2 * np.pi * freq * t)

    # Plot the waveform
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()


    ts1=np.random.normal(0,1,10)
    ts1=y
    ts2=np.random.normal(0,1,10)
    ts2=y1


    print(ts1,len(ts1))
    # print(ts2,len(ts2))


    symbols1,ind=timeseries2symbol(ts1,500,5,10)
    print(symbols1,ind)
    symbols12=convert_symbols(symbols1)


    symbols2,_=timeseries2symbol(ts2,1000,2,10)
    symbols22=convert_symbols(symbols2)
    # print(symbols2,ind)

    # dist=min_dist(symbols1[0],symbols2[0],10,1)

    # print(dist)

sax_test()