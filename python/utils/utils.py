import h5py
import numpy as np
from scipy.signal import decimate
from scipy.signal import butter
from scipy.signal import filtfilt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import datetime
import os

def decim_to_100(directory_to_file,name_of_file,decim_factor=50):
    ''' Function to decimate the raw files into 100 Hz, and convert phase data to microstrain data
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)
        decim_factor: factor to decimate

    Returns:
        Saves a decimated .h5 file, with time and strain

    Raises:
    '''
    # Open the HDF5 file
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r')
    # Read the dataset
    raw_data = file['/Acquisition/Raw[0]/RawData']
    raw_data = np.double(raw_data) # Convert to double
    # Transpose data
    raw_data = raw_data.T
    nch, _ = raw_data.shape
    # Import time
    raw_time = np.double(file['/Acquisition/Raw[0]/RawDataTime'])

    # Decimate
    # 100,000 Hz -> 100 Hz, decim_factor = 1000
    # Get length of decimated vector
    decim_length = len(decimate(raw_data[0,:], decim_factor))
    # Initialize phase_decim array
    # Use empty to make sure that we don't impute values
    raw_decim = np.empty((nch, decim_length))
    time_decim = np.empty((decim_length))

    # decim
    # Decimate each channel time series 
    for i in range(nch): # range(nch)
        raw_decim[i, :] = decimate(raw_data[i,:], decim_factor)

    # Decimate TIME data DON'T use decimate function! The decimate function downsamples the signal
    # after applying an anti-aliasing filter! Just take every decim_factor (1000th) entry
    time_decim = raw_time[0::decim_factor]

    # Check if it is actually 100 Hz
    time_difference = time_decim[1] - time_decim[0]
    if (time_difference - 10000) < 1e-3: # datetime format here, 10000 is 0.01 seconds, unix format
        pass
    else:
        print(time_difference -10000)
        raise ValueError

    # Convert raw_data to phase_data
    phase_decim = raw_decim / 10430.378350470453

    # Convert to strain
    ### Does this change with different channel readouts? CHECK!
    Lambd = 1550e-9 # wavelength for Rayleigh incident light, 1550 nm
    # if nch == 102:
    #     print(8)
    #     Lgauge = 8.167619
    # else:
    #     print(1)
    #     Lgauge = 1.0209523
    Lgauge = 8.167619
    n_FRI = 1.468200 # fiber refractive index
    PSF = 0.78 # photoelastic scaling factor xi

    strain_decim = (Lambd / (4*np.pi*n_FRI*Lgauge*PSF)) * phase_decim * 1e6 # microstrain
    
    # Save decimated strain data
    with h5py.File(directory_to_file+'/'+name_of_file+'_decimated100hz'+'.h5', 'w') as hf:
        hf.create_dataset('strain',  data=strain_decim)
        hf.create_dataset('time',data=time_decim)
        hf.close()

def batch_decim_to_100(directory):
    ''' Decimate all .h5 files in the specified directory.
    Args:
        directory: The path to the directory containing the .h5 files.

    Returns:
        None. Decimated files are saved in the same directory with the '_decimated100hz' suffix.
    '''
    # List all files in the directory
    files = os.listdir(directory)

    # Filter only .h5 files which do not contain 'decimated' in their filename
    h5_files = [f for f in files if f.endswith('.h5') and 'decimated' not in f]

    # Decimate each of these files
    for file in h5_files:
        print(f"Decimating {file}")
        # Extract the file name without the extension
        name_without_ext = os.path.splitext(file)[0]
        decim_to_100(directory, name_without_ext)
        print(f"{file} decimated")


def load_decim_data(directory_to_file,name_of_file):
    ''' Function to load decimated 100 Hz and process the datetimes
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)
        decim_factor: factor to decimate

    Returns:
        strain_data: numpy double of strain data
        time: list of datetimes

    Raises:
    '''
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r+')
    data = file['strain']
    time = file['time']
    # Convert decimated time data to datetime
    time_datetime = [datetime.datetime.fromtimestamp(i/1000000) for i in time]
    # convert h5 group to double
    return np.double(data),time_datetime

def sort_filenames_by_time(filenames):
    ''' Helper function to sort filenames
    '''
    return sorted(filenames, key=lambda x: datetime.datetime.strptime(x.split('_')[1], "%Y-%m-%dT%H%M%S%z"))

def load_decim_data_helper(directory_to_file,name_of_file):
    ''' Helper function to load decimated 100 Hz
    Args:
        directory_to_file: full directory to file (string)
        name_of_file: .h5 file that you want to decimate (string)
        decim_factor: factor to decimate

    Returns:
        strain_data: 
        time: 

    Raises:
    '''
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r+')
    data = file['strain']
    time = file['time']
    return data,time


def concatenate_and_save_h5(directory_to_file, output_filename):
    ''' Function to compile the decimated files
    '''
    files = [f for f in os.listdir(directory_to_file) if f.endswith('_decimated100hz.h5')]
    sorted_files = sort_filenames_by_time(files)

    # Read the first file to determine the shape of the data
    first_strain, _ = load_decim_data_helper(directory_to_file, sorted_files[0].replace('.h5', ''))
    num_spatial_points, _ = first_strain.shape

    # Initialize the arrays based on the shape of the first file
    all_strain_data = np.empty((num_spatial_points, 0))
    all_time_data = np.empty((0,))  # Assuming time data is 1D

    for file in sorted_files:
        print(file)
        strain_data, time_data = load_decim_data_helper(directory_to_file, file.replace('.h5', ''))
        all_strain_data = np.append(all_strain_data, strain_data, axis=1)
        all_time_data = np.append(all_time_data, time_data)  # Assuming time data is 1D

    # print(np.shape(all_strain_data))
    # print(np.shape(all_time_data))

    # Get correct file name to save, get the first datetime of the filenames
    filename = '_'.join(sorted_files[0].split('_')[:2])

    # Save data into a new h5 file
    with h5py.File(directory_to_file+'/'+filename+output_filename+'.h5', 'w') as f:
        f.create_dataset('strain', data=all_strain_data)
        f.create_dataset('time', data=all_time_data)


### Classes

class filter_plot_single:
    def __init__(self,strain_data):
        self.strain_data = strain_data.T
        ## Get properties of strain_data
        nch,nt = np.shape(self.strain_data)
        self.nch = nch
        self.nt = nt
        self.time = np.linspace(0,self.nt/1000,self.nt)
        self.dt = self.time[1] - self.time[0]
        self.psd = None
        self.freqs = None

    def butterworth(self,channel,order=2,cutoff_freq=0.05):
        nyquist_freq = 0.5 * 100
        normal_cutoff = cutoff_freq / nyquist_freq

        b,a = butter(order, normal_cutoff, btype='high')
        self.filtered_data = filtfilt(b,a,self.strain_data[channel,:])
        self.channel = channel
        return self.filtered_data
    
    def psd_freqs(self):
        # Compute the FFT
        fft = np.fft.rfft(self.filtered_data)
        # Compute the PSD
        self.psd = np.abs(fft)**2
        # Get frequencies corresponding to values of PSD
        self.freqs = np.fft.rfftfreq(self.nt,self.dt)
        return self.psd, self.freqs

    def plot_side_by_side(self):
        plt.figure(figsize=(10,6))
        plt.plot(self.time,self.strain_data[self.channel,:]-np.mean(self.strain_data[self.channel,0:150]),label='Raw Data')
        plt.plot(self.time,self.filtered_data,label='Filtered Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Strain (microstrain)')
        plt.title('Butterworth Band Filter')
        plt.grid()
        plt.legend()
        plt.show

    def plot_filtered(self,time_start,time_end,freq=5000):
        plt.figure(figsize=(10,6))
        plt.plot(self.time[time_start:time_end],self.filtered_data[time_start:time_end],label='Filtered Data')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Strain (microstrain)')
        plt.title('Butterworth Band Filtered')
        plt.grid()
        plt.legend()
        plt.show

    def plot_psd(self):
        plt.figure(figsize=(10,6))
        plt.plot(self.freqs,self.psd)
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Power Sepctral Density')
        plt.xlim([0,10])
        plt.title('Power Spectrum Density')
        plt.grid()
        plt.show


    def plot_psd_filtered(self,max_freq):
        fig, axs = plt.subplots(1,2,figsize=(12,4))
        # Times Series
        axs[0].plot(self.time,self.filtered_data)
        axs[0].set_title('Time Series Data')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Microstrain')
        axs[0].grid(visible=True, which='major', axis='both')
        axs[0].set_xlim([0,1.05*np.max(self.time)])
        axs[0].set_ylim([-1.05*np.max(np.abs(self.filtered_data)),1.05*np.max(np.abs(self.filtered_data))])
        # PSD
        axs[1].plot(self.freqs,self.psd)
        axs[1].set_title('PSD')
        axs[1].set_xlabel('Frequency (Hz)')
        axs[1].set_ylabel('PSD')
        axs[1].grid(visible=True, which='major', axis='both')
        axs[1].set_xlim([0,max_freq])

        plt.tight_layout()
        plt.show()

    def butter_all(self,channels,order=2,cutoff_freq=0.05,freq = 5000):
        nyquist_freq = 0.5 * freq
        normal_cutoff = cutoff_freq / nyquist_freq

        b,a = butter(order, normal_cutoff, btype='high')
        self.filtered_data_all = np.zeros_like(self.strain_data)
        for channel in channels:
            self.filtered_data_all[channel] = filtfilt(b,a,self.strain_data[channel,:])
        self.channels = channels
        return self.filtered_data_all

    def iso_view_plot(self,channels,time_start,time_end):
        # "
        # % Time is in 
        # "
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111,projection='3d')

        # Plot each spatial point time series
        for i in channels:
            zs = np.ones_like(self.time)*i*8.167619
            ax.plot(self.time[time_start:time_end],zs[time_start:time_end],self.filtered_data_all[i][time_start:time_end],color=u'#1f77b4',alpha=0.8)


        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')
        ax.set_zlabel('Data Value')
        ax.set_title('3D Line Plots for Time Series at Each Spatial Point')
        ax.set_box_aspect([6, 4, 1])
        ax.view_init(25, 35)  # (elevation, azimuth)
        plt.show()

    def iso_view_scatter(self,channels,time_start,time_end):
        # "
        # % Time is in 
        # "
        cmap = plt.get_cmap('RdYlGn')
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111,projection='3d')
        

        # Plot each spatial point time series
        for i in channels:
            zs = np.ones_like(self.time)*i*8.167619
            sc = ax.scatter(self.time[time_start:time_end],zs[time_start:time_end],self.filtered_data_all[i][time_start:time_end],c=self.filtered_data_all[i][time_start:time_end],cmap=cmap,alpha=0.8)

        fig.colorbar(sc, ax=ax)
        ax.set_xlabel('Time')
        ax.set_ylabel('Channel')
        ax.set_zlabel('Data Value')
        ax.set_title('3D Line Plots for Time Series at Each Spatial Point')
        ax.set_box_aspect([6, 4, 1])
        ax.view_init(20, 35)  # (elevation, azimuth)
        plt.show()

class filter_plot_comparison:
    def __init__(self,strain_P1,strain_P2,strain_P3,strain_P4):
        self.channels = None
        self.strain_P1 = strain_P1
        self.strain_P2 = strain_P2
        self.strain_P3 = strain_P3
        self.strain_P4 = strain_P4
        ## Get properties of strain_data
        # Phase 1
        nch_P1,nt_P1 = np.shape(self.strain_P1)
        self.nch_P1 = nch_P1
        self.nt_P1 = nt_P1
        self.time_P1 = np.linspace(0,self.nt_P1/1000,self.nt_P1)
        self.dt_P1 = self.time_P1[1] - self.time_P1[0]
        self.psd_P1 = None
        self.freqs_P1 = None
        # Phase 2
        nch_P2,nt_P2 = np.shape(self.strain_P2)
        self.nch_P2 = nch_P2
        self.nt_P2 = nt_P2
        self.time_P2 = np.linspace(0,self.nt_P2/1000,self.nt_P2)
        self.dt_P2 = self.time_P2[1] - self.time_P2[0]
        self.psd_P2 = None
        self.freqs_P2 = None
        # Phase 3
        nch_P3,nt_P3 = np.shape(self.strain_P3)
        self.nch_P3 = nch_P3
        self.nt_P3 = nt_P3
        self.time_P3 = np.linspace(0,self.nt_P3/1000,self.nt_P3)
        self.dt_P3 = self.time_P3[1] - self.time_P3[0]
        self.psd_P3 = None
        self.freqs_P3 = None
        # Phase 4
        nch_P4,nt_P4 = np.shape(self.strain_P4)
        self.nch_P4 = nch_P4
        self.nt_P4 = nt_P4
        self.time_P4 = np.linspace(0,self.nt_P4/1000,self.nt_P4)
        self.dt_P4 = self.time_P4[1] - self.time_P4[0]
        self.psd_P4 = None
        self.freqs_P4 = None

    def butterworth(self,channels,order=2,cutoff_freq= 0.1):
        nyquist_freq = 0.5 * 100
        normal_cutoff = cutoff_freq / nyquist_freq

        b,a = butter(order, normal_cutoff, btype = 'high')
        
        # Initialize filtered matrices
        self.filtered_data_P1 = np.zeros_like(self.strain_P1)
        self.filtered_data_P2 = np.zeros_like(self.strain_P2)
        self.filtered_data_P3 = np.zeros_like(self.strain_P3)
        self.filtered_data_P4 = np.zeros_like(self.strain_P4)
        

        # Filter for all channels
        for i in channels:
            self.filtered_data_P1[i] = filtfilt(b,a,self.strain_P1[i,:])
            self.filtered_data_P2[i] = filtfilt(b,a,self.strain_P2[i,:])
            self.filtered_data_P3[i] = filtfilt(b,a,self.strain_P3[i,:])
            self.filtered_data_P4[i] = filtfilt(b,a,self.strain_P4[i,:])

        self.channels = channels
        return self.filtered_data_P1,self.filtered_data_P2,self.filtered_data_P3,self.filtered_data_P4
    
    def plotting_all_phases(self,channel):
        fig, axs = plt.subplots(1,2,figsize=(16,4))
        # Times Series Raw
        axs[0].plot(self.time_P1,self.strain_P1[channel,:]-np.mean(self.strain_P1[channel,:150]),label='Phase 1')
        axs[0].plot(self.time_P2,self.strain_P2[channel,:]-np.mean(self.strain_P2[channel,:150]),label='Phase 2')
        axs[0].plot(self.time_P3,self.strain_P3[channel,:]-np.mean(self.strain_P3[channel,:150]),label='Phase 3')
        axs[0].plot(self.time_P4,self.strain_P4[channel,:]-np.mean(self.strain_P4[channel,:150]),label='Phase 4')

        axs[0].set_title('Time Series Data')
        axs[0].set_xlabel('Time (s)')
        axs[0].set_ylabel('Microstrain')
        axs[0].grid(visible=True, which='major', axis='both')
        axs[0].legend()
        # axs[0].set_xlim([0,1.05*np.max(self.time_P1,self.time_P2,self.time_P3,self.time_P4)])

        # Times Series Filtered
        axs[1].plot(self.time_P1,self.filtered_data_P1[channel],label='Phase 1')
        axs[1].plot(self.time_P2,self.filtered_data_P2[channel],label='Phase 2')
        axs[1].plot(self.time_P3,self.filtered_data_P3[channel],label='Phase 3')
        axs[1].plot(self.time_P4,self.filtered_data_P4[channel],label='Phase 4')

        axs[1].set_title('Time Series Data')
        axs[1].set_xlabel('Time (s)')
        axs[1].set_ylabel('Microstrain')
        axs[1].grid(visible=True, which='major', axis='both')
        axs[1].legend()
        # axs[1].set_xlim([0,1.05*np.max(self.time_P1,self.time_P2,self.time_P3,self.time_P4)])
        # axs.set_ylim([-1.05*np.max(np.abs(self.filtered_data)),1.05*np.max(np.abs(self.filtered_data))])
        # PSD


    
    def psd_freqs(self):
        # Compute the FFT
        fft = np.fft.rfft(self.filtered_data)
        # Compute the PSD
        self.psd = np.abs(fft)**2
        # Get frequencies corresponding to values of PSD
        self.freqs = np.fft.rfftfreq(self.nt,self.dt)
        return self.psd, self.freqs
    




### For Luna and DAS comparisons
class comparison_plots:
    def __init__(self,das_data,luna_data):
        self.das_data = das_data
        self.luna_data = luna_data

    def test(self):
        self.das_data
        self.luna_data

    