#/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Calculate rate constants for interconversion of states.
#
import numpy
numpy.set_printoptions(precision=4)
numpy.seterr(all='ignore')

# The number of state classifications per microsecond
LABELS_PER_US = 250000 # Output is every 10 ps

# The number of states
NSTATES = 3

def calc_rates(simlist, upto, output=True):
    '''
    Calulate the rate constants for interconversion of states, based on 
    simulations in ``simlist``.

    ----------
    Parameters
    ----------
    simlist: (list) An iteratable of paths to saved numpy arrays (.npy files)
        that specify state classifications for each simulation to include in
        the rate calculation. Each numpy array should contain integers that are
        the labels for each state (e.g., "0" might refer to the folded state and 
        "1" might refer to the unfolded state). An "unknown" state with label 
        "-1" is also allowed; this could be used for time points that are in 
        a region intermediate to the other states.

    --------
    Result
    --------
    Prints to the terminal a matrix of rate constant estimates for transitions 
    between each pair of states, along with a matrix describing the standard 
    error of the mean of each rate constant.
    '''

    count_mat = numpy.zeros((NSTATES, NSTATES))
    time_vec = numpy.zeros((NSTATES,))

    unassigned_count = 0
    for sim in simlist:
        states = most_recently_visited_state(load_states(sim, upto))
        for istate in range(NSTATES):
            for jstate in range(NSTATES):
                if istate == jstate:
                    continue
                for i in range(states.shape[0]-1):
                    if states[i] == istate and states[i+1] == jstate:
                        count_mat[istate,jstate] += 1
            time_vec[istate] += numpy.count_nonzero(states==istate)
    if output:
        print("Count matrix for i -> j transitions:")
        print(count_mat)
        print("\n")
        print("Number of time points in each state:")
        print(time_vec)
        print("\n")

    # Matrix of rate constant estimates
    rate_mat = numpy.zeros(count_mat.shape, dtype=float)

    # Matrix of standard errors
    sem_mat = numpy.zeros(count_mat.shape, dtype=float)

    for istate in range(NSTATES):
        rate_mat[istate] = count_mat[istate,:] / time_vec[istate]
        sem_mat[istate] = numpy.sqrt(rate_mat[istate,:]) / \
                          numpy.sqrt(time_vec[istate] * count_mat[istate])

    rate_mat *= LABELS_PER_US # in microseconds
    sem_mat *= LABELS_PER_US # in microseconds

    if output:
        print("Rate matrix for i -> j transitions, in 1/μs:\n"
              "  (row i, column j specifies rate constant estimate for i -> j transition)")
        print(rate_mat)
        print("\n")
        print("Matrix of standard errors of rate constants for i -> j transitions, in 1/μs:\n"
              "  (row i, column j specifies standard error of the rate constant estimate for i -> j transition)")
        print(sem_mat)
        print("\n")

    return rate_mat, sem_mat


def most_recently_visited_state(states):
    '''
    Return the most recently visited state.

    ----------
    Parameters
    ----------
    states: (numpy.ndarray) An array of integers, where states[i] indicates the
        state index of the simulation at time ``i``.

    -------
    Returns
    -------
    most_recent: (numpy.ndarray) An array of integers, where most_recent[i]
        indicates the state that the simulation at time ``i`` has most recently
        visited. That is, if states[i] is a defined state (not -1), then 
        most_recent[i] == states[i]; otherwise most_recent[i] = states[j] for
        the largest j < i such that states[j] >= 0.
    '''
    most_recent = numpy.copy(states)
    for i, state in enumerate(most_recent):
        if i > 0 and state == -1:
            most_recent[i] = most_recent[i-1]
    return most_recent

def load_states(statefilepath, upto):
    '''
    Load the state classifications.

    ----------
    Parameters
    ----------
    statefilepath: (str) The file path to a .npy file containing an array of
        integers, which label the states.

    -------
    Returns
    -------
    states: (numpy.ndarray) A shape (n,) array, where n is the length of the 
        simulation and states[i] is an integer denoting the state classification
        of the simulation at time ``i``.
    '''
    states = numpy.load(statefilepath)[:upto]
    return states


if __name__ == "__main__":
#    simlist =['01-alpha/states.npy',
             #'02-alpha/states.npy',
             #'03-alpha/states.npy']
#    simlist =['01-c7eq/states.npy',
#              '02-c7eq/states.npy',
#              '03-c7eq/states.npy',]
    simlist =['01/states.npy',
              '02/states.npy',
              '03/states.npy',
              '04/states.npy',
              '05/states.npy',]
    #looking_at_state = (0, 2) # Looking at Alpha Helix --> C7ax
    looking_at_state = (1,2) # Looking at C7eq --> C7ax
    rate, se = [], []
    intervals = numpy.asarray([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000])#, 110000, 120000, 130000, 140000, 150000])

    for j in intervals:
        rate_mat, sem_mat = calc_rates(simlist, j, output=True)
        rate.append(rate_mat[looking_at_state])
        se.append(sem_mat[looking_at_state])

    if looking_at_state == (0,2): # alpha
        numpy.save('bf_rate_intervals_alpha.npy', numpy.asarray([intervals/LABELS_PER_US,rate,se]))
    elif looking_at_state == (1,2): # c7eq
        numpy.save('bf_rate_intervals_c7eq.npy', numpy.asarray([intervals/LABELS_PER_US,rate,se]))
    print(intervals/LABELS_PER_US)
    print(rate)
    print(se)
