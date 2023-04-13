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

def pattern__test():

    word_size=5
    alphabet_size=10

    np.random.seed(1)
    # Set the duration and frequency of the sine wave
    duration = 1  # seconds
    freq = 1  # Hz

    # Create a time vector
    t = np.linspace(0, duration, num=duration*100)


    # Generate the sine wave
    y = np.sin(2 * np.pi * freq * t)

    n = np.random.normal(scale=10, size=y.size)

    y = 100 * np.sin(y) + n

    symbols,ind=timeseries2symbol(y,duration*100,word_size,alphabet_size)

    print(symbols)
    # Plot the waveform
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # specify the number of vertical bars you want to divide the plot into
    num_bars = word_size
    # calculate the positions of the bars
    bar_positions = [t[0] + (i+1)*(t[-1]-t[0])/(num_bars+1) for i in range(num_bars)]

    # add vertical lines to the plot at the specified positions
    for pos in bar_positions:
        plt.axvline(x=pos, color='k', linestyle='--')

    plt.xlim(t[0],t[-1])
    plt.show()

pattern__test()