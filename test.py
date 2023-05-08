import numpy as np
from utils.symbolic_aggregate_approximation import convert_symbols,pattern_distance,min_dist,find_pattern
import matplotlib.pyplot as plt
import pandas as pd

def simple_pattern__test():

    word_size=20
    alphabet_size=20

    np.random.seed(1)
    # Set the duration and frequency of the sine wave
    duration = 1.5  # seconds
    freq = 1  # Hz
    num=int(duration*100)

    # Create a time vector
    t = np.linspace(0, duration, num=num)


    # Generate the sine wave
    y = np.cos(2 * np.pi * freq * t)

    n = np.random.normal(scale=5, size=y.size)

    y = 100 * y + n

    symbols,PAA=find_pattern(y,word_size,alphabet_size)

    # symbols=convert_symbols(symbols)

    # Plot the waveform
    plt.plot(t, (y-np.mean(y))/np.std(y),label='Original Signal')
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

    # calculate the positions of the PAA
    paa_positions = [i*len(y)//word_size for i in range(word_size)]
    sax_plot = pd.Series([np.nan for i in range(len(y))])
    for id,pos in enumerate(paa_positions):
        sax_plot[pos]=PAA[id]
    sax_plot = sax_plot.fillna(method='ffill').to_numpy()
    for i in range(len(PAA)):
        plt.plot(t,sax_plot, c='r')
    plt.show()

def pattern_distance_test():
    
    word_size=20
    alphabet_size=5

    # Set the duration and frequency of the sine wave
    duration = 1.5  # seconds
    freq = 1  # Hz
    num=int(duration*100)
    # Create a time vector
    t = np.linspace(0, duration, num=num)
    # Generate the sine wave
    y = np.cos(2 * np.pi * freq * t)
    yc=np.sin(2 * np.pi * freq * t)


    np.random.seed(1)
    n = np.random.normal(scale=5, size=y.size)
    y1 = 100 * y +n
    np.random.seed(100)
    n = np.random.normal(scale=100, size=y.size)
    y2 = n

    window_size=len(y)

    symbols1,_=find_pattern(y1,word_size,alphabet_size)
    symbols2,_=find_pattern(y2,word_size,alphabet_size)
    print(symbols1,symbols2)
    print(pattern_distance(symbols1,symbols2))
    print(min_dist(symbols1,symbols2,alphabet_size,window_size/word_size))



pattern_distance_test()