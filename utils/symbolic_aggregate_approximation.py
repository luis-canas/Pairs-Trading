import sys

"""Symbolic Aggregate approXimation.

References
----------
        J. Lin, E. Keogh, L. Wei, and S. Lonardi, "Experiencing SAX: a
        novel symbolic representation of time series". Data Mining and
        Knowledge Discovery, 15(2), 107-144 (2007).
"""

import numpy as np
from scipy.stats import norm


def pattern_distance(pattern1, pattern2):
    """
    Calculate the distance between two patterns
    """
    return np.linalg.norm(pattern1 - pattern2)




def find_pattern(data,n, alphabet_size):

    N=len(data)
    win_size = int(N/n)  # number of data points on the raw time series that will be mapped to a single symbol

    # Z normalize section.
    section = (data - np.mean(data))/np.std(data)
        
    # take care of the special case where there is no dimensionality reduction
    if N == n:
        PAA = section
    
    # Convert to PAA.  
    else:
        # N is not dividable by n
        if (N/n - np.floor(N/n)):

            #Expand the matrix
            expanded_section = np.tile(section, (n,1))
            # Reshape the matrix into a row vector
            expanded_section = expanded_section.reshape(1,n*N,order='F')

            PAA = np.mean(np.reshape(expanded_section, (N, n),order='F'),axis=0)

        # N is dividable by n
        else:
            try:
                PAA = np.mean(np.reshape(section, (win_size,n),order='F'),axis=0)
            except:
                print(section)
                print((win_size,n))
                print(sys.stderr)
                sys.exit(1)

    # norm distribution breakpoints by scipy.stats
    breakpoints = norm.ppf(np.linspace(1./alphabet_size,1-1./alphabet_size, alphabet_size-1))
    
    SAX = np.zeros(PAA.shape, dtype=int) - 1

    for id, bp in enumerate(breakpoints):
        indices = np.logical_and(SAX < 0, PAA < bp)
        SAX[indices] = id
    SAX[SAX < 0] = alphabet_size - 1


    return SAX,PAA


def convert_symbols(symbols):
    
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

    # Convert the symbols to letters and join them into a string for each row
    return [letter_dict[int(num)] for num in symbols]

def min_dist(str1, str2, alphabet_size, compression_ratio):
    if len(str1) != len(str2):
        print('error: the strings must have equal length!')
        return

    if np.any(str1 > alphabet_size) or np.any(str2 > alphabet_size):
        print('error: some symbol(s) in the string(s) exceed(s) the alphabet size!')
        return
    
    dist_matrix = build_dist_table(alphabet_size)

    dist = np.sqrt(compression_ratio * np.sum(dist_matrix[str1.astype(int)-1][:,str2.astype(int)-1]))

    return dist


def build_dist_table(alphabet_size):
    cutlines = None

    switch = {
        2: [0],
        3: [-0.43, 0.43],
        4: [-0.67, 0, 0.67],
        5: [-0.84, -0.25, 0.25, 0.84],
        6: [-0.97, -0.43, 0, 0.43, 0.97],
        7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
        8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],
        9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22],
        10:[-1.28, -0.84, -0.52, -0.25, 0, 0.25, 0.52, 0.84, 1.28],
        11:[-1.34,-0.91,-0.6,-0.35,-0.11,0.11,0.35,0.6,0.91,1.34],
        12:[-1.38,-0.97,-0.67,-0.43,-0.21,0,0.21,0.43,0.67,0.97,1.38],
        13:[-1.43,-1.02,-0.74,-0.5,-0.29,-0.1,0.1,0.29,0.5,0.74,1.02,1.43],
        14:[-1.47,-1.07,-0.79,-0.57,-0.37,-0.18,0,0.18,0.37,0.57,0.79,1.07,1.47],
        15:[-1.5,-1.11,-0.84,-0.62,-0.43,-0.25,-0.08,0.08,0.25,0.43,0.62,0.84,1.11,1.5],
        16:[-1.53,-1.15,-0.89,-0.67,-0.49,-0.32,-0.16,0,0.16,0.32,0.49,0.67,0.89,1.15,1.53],
        17:[-1.56,-1.19,-0.93,-0.72,-0.54,-0.38,-0.22,-0.07,0.07,0.22,0.38,0.54,0.72,0.93,1.19,1.56],
        18:[-1.59,-1.22,-0.97,-0.76,-0.59,-0.43,-0.28,-0.14,0,0.14,0.28,0.43,0.59,0.76,0.97,1.22,1.59],
        19:[-1.62,-1.25,-1,-0.8,-0.63,-0.48,-0.34,-0.2,-0.07,0.07,0.2,0.34,0.48,0.63,0.8,1,1.25,1.62],
        20:[-1.64,-1.28,-1.04,-0.84,-0.67,-0.52,-0.39,-0.25,-0.13,0,0.13,0.25,0.39,0.52,0.67,0.84,1.04,1.28,1.64]

    }

    cutlines = switch[alphabet_size]
    

    dist_matrix = np.zeros((alphabet_size, alphabet_size))

    for i in range(alphabet_size):
        for j in range(i+2, alphabet_size):
            dist_matrix[i,j] = (cutlines[i] - cutlines[j-1])**2
            dist_matrix[j,i] = dist_matrix[i,j]

    return dist_matrix
