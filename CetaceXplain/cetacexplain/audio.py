import numpy as np
from scipy.io.wavfile import write
from scipy import signal
from scipy.signal import chirp

#TODO: Think about adding amplitude data to wav file generation
#TODO: Fix rough transition "clicking" between frequency values passed to generate_wavfile
#      see https://stackoverflow.com/questions/68471877/tone-sweep-from-array-of-frequencies-with-python/68472399#68472399
#      for the start to a possible solutiuon
def generate_wavfile(wavfile_name, frequencies, total_duration = 1, num_of_intervals = 1000):
    """
    Generate a wav file from a given array of frequency values. The wav file is saved in the local directory.

    Parameters
    ----------
    wavfile_name : string
        Name of wavfile to be generated.
    frequencies : numpy array
        Numpy array of frequncy values (in Hz).
    total_duration : float, optional
        Desired total duration of wav file (in seconds). The default is 1 s.
    num_of_intervals : int, optional
        num_of_intervals. The default is 1000.

    Returns
    -------
    None.

    """

    # inspired by useful stackexchange comment and scipy docs: 
    # https://stackoverflow.com/questions/68471877/tone-sweep-from-array-of-frequencies-with-python/68472399#68472399
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.chirp.html

    # initialize array of amplitudes varying with respect to time
    # (see return value of chirp() in scipy doc above for more info)
    wav_array = np.array([])

    # specify time step between each frequency value given
    step_duration = total_duration/(len(frequencies) - 1) # there are (n-1) step durations for n frequencies
    step_time_array = np.linspace(0, step_duration, num_of_intervals)
    
    # iteratively add values for each step between frequencies
    for index, frequency in np.ndenumerate(frequencies):
        i = index[0]
        if i < len(frequencies) - 1:
            wav_array = np.append(wav_array, 
                                  chirp(step_time_array, f0=frequencies[i],
                                        f1= frequencies[i+1], t1=step_duration, method="linear"))
    
    # write wav_array to a wavfile with the sample rate being the 
    # length of wav array divided by the desired total duration of the audio clip
    write(wavfile_name, rate=int(len(wav_array)/total_duration), data=wav_array.astype(np.float32)) 