"""
Module for opening and processing raw recording data from .h5 files.

This module contains a function to open and process raw recording data, particularly from
files generated by Multi Channel Systems (MCS). It handles large files by processing data
in chunks to avoid memory limitations.

Functions:
    file_opener_raw_recording_data(file_name, path, is_big_file=False): Opens a .h5 file
    and extracts recording data, electrode stream, and sampling frequency.
"""
# If you want to use / import Multi channel Systems data, easiest way is to
# import their custom functions to open the standard generated .h5 files:
# MCS PyData tools


# pip install McsPyDataTools
import McsPy
import McsPy.McsData
from McsPy import ureg, Q_


def file_opener_raw_recording_data(file_name, path, is_big_file=False):
    """
        Opens a .h5 file and extracts raw recording data, electrode stream, and sampling frequency.

        This function opens a .h5 file, typically from Multi Channel Systems (MCS), and extracts
        the raw recording data, the electrode stream, and the sampling frequency. It can handle
        large files by reading in the data in segments to avoid RAM limitations.

        Args:
            file_name (str): The name of the file to be opened.
            path (str): The path of the file location.
            is_big_file (bool, optional): Flag to indicate if the file is too big to be opened at once.
                                          Defaults to False.

        Returns:
            tuple: A tuple containing:
                - recording_data (numpy.ndarray): Array containing the raw recording data.
                                                 Shape = (recorded data points, number of electrodes)
                - electrode_stream (McsPy.McsData.AnalogStream): The original data file containing all raw data and channel infos.
                - fsample (int): Sampling frequency in Hz.

        Note:
            - Requires McsPyDataTools for processing .h5 files generated by Multi Channel Systems.
            - Handles large files by processing in chunks if 'is_big_file' is True.
        """
    file_path = path + file_name + '.h5'
    
    big_file = is_big_file
    
    file = McsPy.McsData.RawData(file_path)
    electrode_stream = file.recordings[0].analog_streams[0]

    # extract basic information
    electrode_ids = [c.channel_id for c in electrode_stream.channel_infos.values()]
    fsample = int(electrode_stream.channel_infos[0].sampling_frequency.magnitude)
    
    # Get signal
    if big_file:
        step_size = 10
        min_step = 0
        max_step = 60
        for i in range(min_step, max_step, step_size):
        #signal = get_channel_data(electrode_stream, channel_ids = [j for j in range(i,i+step_size)])
        scale_factor_for_uV = Q_(1,'volt').to(ureg.uV).magnitude
        if i == min_step:
          recording_data = (get_channel_data(electrode_stream, channel_ids = [j for j in range(i,i+step_size)]) * scale_factor_for_uV).T
        else:
          recording_data = np.concatenate((recording_data, (get_channel_data(electrode_stream, channel_ids = [j for j in range(i,i+step_size)]) * scale_factor_for_uV).T), axis=1)
        print("iteration", i+step_size, "completed")
        
    
    else:
      signal = get_channel_data(electrode_stream, channel_ids = [])
      scale_factor_for_uV = Q_(1,'volt').to(ureg.uV).magnitude
      recording_data = (get_channel_data(electrode_stream, channel_ids = []) * scale_factor_for_uV).T
    
    print("recording.shape:", recording_data.shape)
    
    return recording_data, electrode_stream, fsample