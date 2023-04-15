import numpy as np
from utils.symbolic_aggregate_approximation import timeseries2symbol,convert_symbols,pattern_distance,min_dist
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

def simple_pattern__test():

    word_size=10
    alphabet_size=10

    np.random.seed(1)
    # Set the duration and frequency of the sine wave
    duration = 1.5  # seconds
    freq = 1  # Hz

    # Create a time vector
    t = np.linspace(0, duration, num=int(duration*100))


    # Generate the sine wave
    y = np.sin(2 * np.pi * freq * t)

    n = np.random.normal(scale=0, size=y.size)

    y = 100 * np.sin(y) + n

    symbols,ind=timeseries2symbol(y,len(y),word_size,alphabet_size)

    symbols=convert_symbols(symbols)

    # Plot the waveform
    plt.plot(t, y)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # specify the number of vertical bars you want to divide the plot into
    num_bars = word_size-1
    # calculate the positions of the bars
    bar_positions = [t[0] + (i+1)*(t[-1]-t[0])/(num_bars+1) for i in range(num_bars)]

    # add vertical lines to the plot at the specified positions
    for pos in bar_positions:
        plt.axvline(x=pos, color='k', linestyle='--')

    plt.xlim(t[0],t[-1])

    #bar_positions
    bar_width=bar_positions[1]-bar_positions[0]
    # calculate the positions of the bars
    sym_positions = [bar_width/2 + i*bar_width for i in range(word_size)]
    for i, pos in enumerate(sym_positions):
        plt.text(pos, 0.9, symbols[i], ha='center',fontweight='bold',fontsize=14)
    plt.show()

def pattern_distance_test():
    word_size=10
    alphabet_size=10

    np.random.seed(1)
    # Set the duration and frequency of the sine wave
    duration = 1.5  # seconds
    freq = 1  # Hz

    # Create a time vector
    t = np.linspace(0, duration, num=int(duration*100))
    


    # Generate the sine wave
    y = np.sin(2 * np.pi * freq * t)

    y2=np.sin(3 * np.pi * freq * t)

    symbols1,ind=timeseries2symbol(y,len(y),word_size,alphabet_size)
    symbols2,ind=timeseries2symbol(y2,len(y2),word_size,alphabet_size)

    print(pattern_distance(symbols1[0],symbols2[0]))
    print(min_dist(symbols1[0],symbols2[0],alphabet_size,len(symbols1[0])/word_size))



pattern_distance_test()