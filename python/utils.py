import h5py
import numpy as np
from scipy.signal import decimate
from scipy.signal import butter
from scipy.signal import filtfilt
import matplotlib.pyplot as plt







def decim_to_100(directory_to_file,name_of_file):

    # Open the HDF5 file
    file = h5py.File(directory_to_file+'/'+name_of_file+'.h5', 'r')
    # Read the dataset
    raw_data = file['/Acquisition/Raw[0]/RawData']
    raw_data = np.double(raw_data) # Convert to double
    # Transpose data
    raw_data = raw_data.T
    nch, _ = raw_data.shape

    # Decimate
    decim_factor = 50  # 5,000 Hz -> 100 Hz
    # Get length of decimated vector
    decim_length = len(decimate(raw_data[0,:], decim_factor))
    # Initialize phase_decim array
    raw_decim = np.zeros((nch, decim_length))

    # Decimate each channel time series 
    for i in range(nch): # range(nch)
        raw_decim[i, :] = decimate(raw_data[i,:], decim_factor)

    # Convert raw_data to phase_data
    phase_decim = raw_decim / 10430.378350470453

    # Convert to strain
    Lambd = 1550e-9
    Lgauge = 8.167619
    n_FRI = 1.468200
    PSF = 0.78

    strain_decim = (Lambd / (4*np.pi*n_FRI*Lgauge*PSF)) * phase_decim * 1e6 # microstrain

    # Save decimated strain data
    with h5py.File(directory_to_file+'/'+name_of_file+'_decimated100hz'+'.h5', 'w') as hf:
        hf.create_dataset(directory_to_file+'/'+name_of_file+'_decimated100hz',  data=strain_decim)

def load_decim_data(directory_of_file):
    file = h5py.File(directory_of_file+'.h5', 'r')
    # data = file['/Acquisition/Raw[0]/RawData']
    data = file[directory_of_file]
    # convert h5 group to double
    return np.double(data)



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

    