
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


def find_pattern(data, n, alphabet_size):

    N = len(data)
    # number of data points on the raw time series that will be mapped to a single symbol
    win_size = int(N/n)

    # Z normalize section.
    section = (data - np.mean(data))/np.std(data)

    # take care of the special case where there is no dimensionality reduction
    if N == n:
        PAA = section

    # Convert to PAA.
    else:
        # N is not dividable by n
        if (N/n - np.floor(N/n)):

            # Expand the matrix
            expanded_section = np.tile(section, (n, 1))
            # Reshape the matrix into a row vector
            expanded_section = expanded_section.reshape(1, n*N, order='F')

            PAA = np.mean(np.reshape(expanded_section,
                          (N, n), order='F'), axis=0)

        # N is dividable by n
        else:
            PAA = np.mean(np.reshape(
                section, (win_size, n), order='F'), axis=0)

    # norm distribution breakpoints by scipy.stats
    breakpoints = norm.ppf(np.linspace(
        1./alphabet_size, 1-1./alphabet_size, alphabet_size-1))

    SAX = np.zeros(PAA.shape, dtype=int) - 1

    for id, bp in enumerate(breakpoints):
        indices = np.logical_and(SAX < 0, PAA < bp)
        SAX[indices] = id
    SAX[SAX < 0] = alphabet_size - 1

    return SAX, PAA


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

    bp = norm.ppf(np.linspace(1./alphabet_size, 1 -
                  1./alphabet_size, alphabet_size-1))

    dist_matrix = np.zeros((alphabet_size, alphabet_size))

    for i in range(alphabet_size):
        for j in range(i+2, alphabet_size):
            dist_matrix[j, i] = dist_matrix[i, j] = (bp[i] - bp[j-1])**2

    dist = np.sqrt(compression_ratio *
                   np.sum(dist_matrix[str1.astype(int), str2.astype(int)]))

    return dist


def get_best_patterns(position, spread, alphabet, word_size_long=None, window_size_long=None, word_size_exit_long=None, window_size_exit_long=None, word_size_short=None, window_size_short=None, word_size_exit_short=None, window_size_exit_short=None):

    LONG_SPREAD = 1
    SHORT_SPREAD = -1
    CLOSE_POSITION = 0

    n = len(word_size_long)
    long_sax_seq_list, short_sax_seq_list = [], []

    for idx in range(n):
        if position == CLOSE_POSITION:

            long_sax_seq, _ = find_pattern(
                spread[-window_size_long[idx]:], word_size_long[idx], alphabet)
            long_sax_seq_list.append(long_sax_seq)

            short_sax_seq, _ = find_pattern(
                spread[-window_size_short[idx]:], word_size_short[idx], alphabet)
            short_sax_seq_list.append(short_sax_seq)

        elif position == LONG_SPREAD:

            long_sax_seq, _ = find_pattern(
                spread[-window_size_exit_long[idx]:], word_size_exit_long[idx], alphabet)
            long_sax_seq_list.append(long_sax_seq)

        elif position == SHORT_SPREAD:

            short_sax_seq, _ = find_pattern(
                spread[-window_size_exit_short[idx]:], word_size_exit_short[idx], alphabet)
            short_sax_seq_list.append(short_sax_seq)

    return long_sax_seq_list, short_sax_seq_list


def get_best_distance(sax_seq_list, pattern, distances):

    n = len(pattern)

    dist = np.full(n, np.inf)

    for idx in range(n):
        d = pattern_distance(sax_seq_list[idx], pattern[idx])
        dist[idx] = d

    min_idx = np.argmin(dist)
    min_dist = np.min(dist)

    max_idx = np.argmax(dist)
    max_dist = np.max(dist)

    return min_dist, min_idx


# def get_best_distance(sax_seq_list,pattern,distances):

#     n=len(pattern)

#     dist=np.full(n, np.inf)

#     for idx in range(n):

    # d = pattern_distance(sax_seq_list[idx], pattern[idx])

    # if d<distances[idx]:
    #     dist[idx]=d


#     min_idx = np.argmin(dist)
#     min_dist = np.min(dist)


#     return min_dist,min_idx

def get_results(results, w_size):

    # Define chromossomes intervals
    x = results.X

    MAX_SIZE = w_size
    NON_PATTERN_SIZE = 1+1+1+1
    CHROMOSSOME_SIZE = NON_PATTERN_SIZE+MAX_SIZE
    ENTER_LONG = CHROMOSSOME_SIZE
    EXIT_LONG = 2*CHROMOSSOME_SIZE
    ENTER_SHORT = 3*CHROMOSSOME_SIZE
    EXIT_SHORT = 4*CHROMOSSOME_SIZE

    # create arrays of optimal patterns
    n = len(x)
    dist_long, word_size_long, window_size_long, days_long, pattern_long = np.zeros(
        n), np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int), []
    dist_exit_long, word_size_exit_long, window_size_exit_long, pattern_exit_long = np.zeros(
        n), np.zeros(n, dtype=int), np.zeros(n, dtype=int), []
    dist_short, word_size_short, window_size_short, days_short, pattern_short = np.zeros(
        n), np.zeros(n, dtype=int), np.zeros(n, dtype=int), np.zeros(n, dtype=int), []
    dist_exit_short, word_size_exit_short, window_size_exit_short, pattern_exit_short = np.zeros(
        n), np.zeros(n, dtype=int), np.zeros(n, dtype=int), []

    # extract chromossomes
    for ind, solution in enumerate(x):

        long_genes = solution[:ENTER_LONG]
        dist_long[ind], word_size_long[ind], window_size_long[ind], days_long[ind], p = long_genes[0], round(
            long_genes[1]), round(long_genes[2]), round(long_genes[3]), np.round(long_genes[4:])
        pattern_long.append(p[:word_size_long[ind]])

        exit_long_genes = solution[ENTER_LONG:EXIT_LONG]
        dist_exit_long[ind], word_size_exit_long[ind], window_size_exit_long[ind], p = exit_long_genes[0], round(
            exit_long_genes[1]), round(exit_long_genes[2]), np.round(exit_long_genes[4:])
        pattern_exit_long.append(p[:word_size_exit_long[ind]])

        short_genes = solution[EXIT_LONG:ENTER_SHORT]
        dist_short[ind], word_size_short[ind], window_size_short[ind], days_short[ind], p = short_genes[0], round(
            short_genes[1]), round(short_genes[2]), round(long_genes[3]), np.round(short_genes[4:])
        pattern_short.append(p[:word_size_short[ind]])

        exit_short_genes = solution[ENTER_SHORT:EXIT_SHORT]
        dist_exit_short[ind], word_size_exit_short[ind], window_size_exit_short[ind], p = exit_short_genes[0], round(
            exit_short_genes[1]), round(exit_short_genes[2]), np.round(exit_short_genes[4:])
        pattern_exit_short.append(p[:word_size_exit_short[ind]])

    class PatternResults(object):
        def __init__(self, dist=-1, word_size=-1, window_size=-1, days=-1, pattern=-1):
            self.dist= dist
            self.word_size= word_size
            self.window_size= window_size
            self.days= days
            self.pattern= pattern

    long=PatternResults(dist=dist_long,word_size=word_size_long,window_size=window_size_long,days=days_long,pattern=pattern_long)
    exit_long=PatternResults(dist=dist_exit_long,word_size=word_size_exit_long,window_size=window_size_exit_long,pattern=pattern_exit_long)
    short=PatternResults(dist=dist_short,word_size=word_size_short,window_size=window_size_short,days=days_short,pattern=pattern_short)
    exit_short=PatternResults(dist=dist_exit_short,word_size=word_size_exit_short,window_size=window_size_exit_short,pattern=pattern_exit_short)
    
    return long,exit_long,short,exit_short
