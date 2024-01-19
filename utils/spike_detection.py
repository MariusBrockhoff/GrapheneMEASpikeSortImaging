import numpy as np
import pickle
import math


def spike_detection(filtered_signal, electrode_stream, fsample, data_preprocessing_config):
    """
    Detect spikes in a filtered signal and save the results.

    This function processes a filtered signal to detect spikes based on thresholding and
    records the spike shapes and timings.

    Arguments:
        filtered_signal: An array of the filtered signal. Shape = (recorded data points, number of electrodes).
        electrode_stream: Original data file containing all raw data and channel infos.
        fsample: Sampling frequency in Hz.
        data_preprocessing_config: Configuration for data preprocessing

    Outputs:
        A dictionary containing all detected spikes, file name, sampling frequency, recording length,
        and spike shapes. Spikes are stored under the key 'Raw_spikes', with shape (number of spikes x 2 +
        dat_points_pre_min + dat_points_post_min). The first column contains the electrode number, the second
        column contains the spike time, and the rest contain the spike shape.

    Returns:
        results: A dictionary with the detected spike information and metadata.
    """

    length_of_chunck = data_preprocessing_config.INTERVAL_LENGTH
    points_pre = data_preprocessing_config.DAT_POINTS_PRE_MIN
    points_post = data_preprocessing_config.DAT_POINTS_POST_MIN


    overlap = 20  # 10 data point overlap between chunks to check whether it is the local minima.
    t = 0  # rejects the 1st second to avoid ripples in filter
    no_chunks = math.ceil(filtered_signal.shape[0]/fsample/length_of_chunck)
    spk = list()
    spike_times = [[0] for x in range(filtered_signal.shape[1])]

    while t < no_chunks:  # rejects the incomplete chunk at the end to avoid filter ripple
        if (t+1)*fsample*length_of_chunck <= len(filtered_signal):
            chunk = filtered_signal[t*fsample*length_of_chunck:((t+1)*fsample*length_of_chunck + overlap - 1)]
        else:
            chunk = filtered_signal[t*fsample*length_of_chunck:]

        med = np.median(np.absolute(chunk)/0.6745, axis=0)
    
        for index in range(len(chunk)):
            if points_pre < index < fsample*length_of_chunck-points_post:
                threshold_cross = chunk[index, :] < data_preprocessing_config.MIN_TH*med
                threshold_arti = chunk[index, :] > data_preprocessing_config.MAX_TH*med

                threshold = threshold_cross*threshold_arti
                probable_spike = threshold

                if np.sum(probable_spike > 0):
                    for e in range(filtered_signal.shape[1]):
                        ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
                        channel_id = ids[e]
                        channel_info = electrode_stream.channel_infos[channel_id]
                        ch = int(channel_info.info['Label'][-2:])
                        # whether threshold exceeded at an electrode and if it is rejected
                        if probable_spike[e] == 1 and not (ch in data_preprocessing_config.REJECT_CHANNELS):
                            t_diff = (fsample*t*length_of_chunck + index) - spike_times[e][-1]
                            # whether the spike is 2ms apart and whether it is the true minimum and not just any point
                            # below -5*SD
                            if t_diff > data_preprocessing_config.REFREC_PERIOD*fsample and chunk[index, e] == np.min(chunk[(index-points_pre):(index+points_post), e]):
                                spike_times[e].append(fsample*t*length_of_chunck + index)
                                # making sure that the whole spike waveform is within the limits of the filtered signal
                                # array
                                if (fsample*t*length_of_chunck + index + points_post) < filtered_signal.shape[0]:
                                    # selecting 1.6ms around the spike time from the whole filtered signal array
                                    spk_wave = list(filtered_signal[(fsample*t*length_of_chunck + index - points_pre):(fsample*t*length_of_chunck + index + points_post), e])
                                    spk_wave.insert(0, (fsample*t*length_of_chunck + index))
                                    spk_wave.insert(0, ch)
                                    spk.append(spk_wave)

        t = t+1

    print("Total number of detected spikes:", len(spk))    
    dat_arr = np.array(spk)
    results = {"Filename": data_preprocessing_config.FILE_NAME, "Sampling rate": fsample, "Recording len": filtered_signal.shape[0] / fsample,
               "Raw_spikes": dat_arr[dat_arr[:, 1].argsort()]}

    #Save resulting spike file at save_path
    with open(data_preprocessing_config.SAVE_PATH, 'wb+') as f:
        pickle.dump(results, f, -1)
    return results
    
