

"""Symbolic Aggregate approXimation.

References
----------
        J. Lin, E. Keogh, L. Wei, and S. Lonardi, "Experiencing SAX: a
        novel symbolic representation of time series". Data Mining and
        Knowledge Discovery, 15(2), 107-144 (2007).
"""

import numpy as np


def timeseries2symbol(data, N, n, alphabet_size):

    if alphabet_size > 20:
        print("Currently alphabet_size cannot be larger than 20. Please update the breakpoint table if you wish to do so")
        return

    win_size = int(N/n)  # number of data points on the raw time series that will be mapped to a single symbol

    pointers = []   # Initialize pointers,
    symbolic_data = np.zeros(n) # Initialize symbolic_data with a void string, it will be removed later.

    # Scan across the time series extract sub sequences, and converting them to strings.
    for i in range(len(data)-(N-1)):
        # Remove the current subsection.
        sub_section = data[i:i+N]
        
        # take care of the special case where there is no dimensionality reduction
        if N == n:
            PAA = sub_section
        
        # Convert to PAA.  
        else:
            # N is not dividable by n
            if (N/n - np.floor(N/n)):
                temp = np.zeros((n, N))
                for j in range(n):
                    temp[j, :] = sub_section
                expanded_sub_section = np.reshape(temp, (1, N*n))
                PAA = np.mean(np.reshape(expanded_sub_section, (N, n)),axis=0)
            # N is dividable by n
            else:

                PAA = np.mean(np.reshape(sub_section, (win_size, n)),axis=0)
        
        current_string = map_to_string(PAA, alphabet_size)   # Convert the PAA to a string.
        
        if not np.all(current_string == symbolic_data[-1]): # If the string differs from its leftmost neighbor...
            symbolic_data = np.vstack((symbolic_data, current_string)) # ... add it to the set...
            pointers.append(i)  # ... and add a new pointer.

    # Delete the first element, it was just used to initialize the data structure
    symbolic_data = np.delete(symbolic_data, 0, 0)
    
    return symbolic_data, pointers




def map_to_string(PAA, alphabet_size):

    string = np.zeros(len(PAA))
    
    cut_points=np.array(switch[alphabet_size])
        
    for i in range(len(PAA)):
        string[i] = np.sum(np.where(cut_points <= PAA[i],1,0))
    
    return string


def convert_symbols(symbols):
    
    # Convert the symbols to letters and join them into a string for each row
    return np.array([''.join([letter_dict[int(num)] for num in row]) for row in symbols])

# Define the cut points for the alphabet
switch = {
    2:[-np.inf,0],
    3:[-np.inf,-0.43,0.43],
    4:[-np.inf,-0.67,0,0.67],
    5:[-np.inf,-0.84,-0.25,0.25,0.84],
    6:[-np.inf,-0.97,-0.43,0,0.43,0.97],
    7:[-np.inf,-1.07,-0.57,-0.18,0.18,0.57,1.07],
    8:[-np.inf,-1.15,-0.67,-0.32,0,0.32,0.67,1.15],
    9:[-np.inf,-1.22,-0.76,-0.43,-0.14,0.14,0.43,0.76,1.22],
    10:[-np.inf,-1.28,-0.84,-0.52,-0.25,0.,0.25,0.52,0.84,1.28],
    11:[-np.inf,-1.34,-0.91,-0.6,-0.35,-0.11,0.11,0.35,0.6,0.91,1.34],
    12:[-np.inf,-1.38,-0.97,-0.67,-0.43,-0.21,0,0.21,0.43,0.67,0.97,1.38],
    13:[-np.inf,-1.43,-1.02,-0.74,-0.5,-0.29,-0.1,0.1,0.29,0.5,0.74,1.02,1.43],
    14:[-np.inf,-1.47,-1.07,-0.79,-0.57,-0.37,-0.18,0,0.18,0.37,0.57,0.79,1.07,1.47],
    15:[-np.inf,-1.5,-1.11,-0.84,-0.62,-0.43,-0.25,-0.08,0.08,0.25,0.43,0.62,0.84,1.11,1.5],
    16:[-np.inf,-1.53,-1.15,-0.89,-0.67,-0.49,-0.32,-0.16,0,0.16,0.32,0.49,0.67,0.89,1.15,1.53],
    17:[-np.inf,-1.56,-1.19,-0.93,-0.72,-0.54,-0.38,-0.22,-0.07,0.07,0.22,0.38,0.54,0.72,0.93,1.19,1.56],
    18:[-np.inf,-1.59,-1.22,-0.97,-0.76,-0.59,-0.43,-0.28,-0.14,0,0.14,0.28,0.43,0.59,0.76,0.97,1.22,1.59],
    19:[-np.inf,-1.62,-1.25,-1,-0.8,-0.63,-0.48,-0.34,-0.2,-0.07,0.07,0.2,0.34,0.48,0.63,0.8,1,1.25,1.62],
    20:[-np.inf,-1.64,-1.28,-1.04,-0.84,-0.67,-0.52,-0.39,-0.25,-0.13,0,0.13,0.25,0.39,0.52,0.67,0.84,1.04,1.28,1.64]
}

# Define the mapping of numbers to letters
letter_dict = {
    1: 'a',
    2: 'b',
    3: 'c',
    4: 'd',
    5: 'e',
    6: 'f',
    7: 'g',
    8: 'h',
    9: 'i',
    10: 'j',
    11: 'k',
    12: 'l',
    13: 'm',
    14: 'n',
    15: 'o',
    16: 'p',
    17: 'q',
    18: 'r',
    19: 's',
    20: 't'
}


