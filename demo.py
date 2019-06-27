
long='''  ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ 
|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|
 ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ 
|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|
                                                                                                                                             
                                                                                                                                             
                                                                                                                                             
                  ________  ___ _____  ___  ___                                    _____                 _                                   
  ______ ______  |  _  |  \/  |/  ___| |  \/  |                                   /  ___|               (_)           ______ ______          
 |______|______| | | | | .  . |\ `--.  | .  . | __ _ _ __   __ _  __ _  ___ _ __  \ `--.  ___ _ ____   ___  ___ ___  |______|______|         
  ______ ______  | | | | |\/| | `--. \ | |\/| |/ _` | '_ \ / _` |/ _` |/ _ \ '__|  `--. \/ _ \ '__\ \ / / |/ __/ _ \  ______ ______          
 |______|______| \ \_/ / |  | |/\__/ / | |  | | (_| | | | | (_| | (_| |  __/ |    /\__/ /  __/ |   \ V /| | (_|  __/ |______|______|         
                  \___/\_|  |_/\____/  \_|  |_/\__,_|_| |_|\__,_|\__, |\___|_|    \____/ \___|_|    \_/ |_|\___\___|                         
                                                                  __/ |                                                                      
                                                                 |___/                                                                       
                                                                                                                                             
 ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ 
|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|
 ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ ______ 
|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|______|'''


print(long)

import terminalplot as plt
import numpy as np
import time
import cv2

spaces= '''\


'''
print(spaces)

print('Loading new PYNQ Overlay: xv2Filter2DDilate')

# Load filter2D + dilate overlay
from pynq import Overlay
bareHDMI = Overlay("/usr/local/lib/python3.6/dist-packages/"
               "pynq_cv/overlays/xv2Filter2DDilate.bit")
import pynq_cv.overlays.xv2Filter2DDilate as xv2

# Load xlnk memory mangager
from pynq import Xlnk
Xlnk.set_allocator_library("/usr/local/lib/python3.6/dist-packages/"
                           "pynq_cv/overlays/xv2Filter2DDilate.so")
mem_manager = Xlnk()

print('Loaded 2D Filter Overlay')

print('Requesting video input')

hdmi_in = bareHDMI.video.hdmi_in
hdmi_out = bareHDMI.video.hdmi_out

from pynq.lib.video import *
hdmi_in.configure(PIXEL_GRAY)
hdmi_out.configure(hdmi_in.mode)

hdmi_in.cacheable_frames = False
hdmi_out.cacheable_frames = False

hdmi_in.start()
hdmi_out.start()

print('Video input successful')

mymode = hdmi_in.mode
print(str(mymode))

height = hdmi_in.mode.height
width = hdmi_in.mode.width
bpp = hdmi_in.mode.bits_per_pixel


#filters
gaussian = np.array([[0.0625,0.125,0.0625],[0.125,0.25,0.125],[0.0625,0.125,0.0625]],np.float32)
sobelV = np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]],np.float32)
sobelH = np.array([[1.0,2.0,1.0],[0.0,0.0,0.0],[-1.0,-2.0,-1.0]],np.float32)
avgB = np.ones((3,3),np.float32)/9.0
laplacianH = np.array([[0.0,1.0,0.0],[1.0,-4.0,1.0],[0.0,1.0,0.0]],np.float32)
gaussianH = np.array([[-0.0625,-0.125,-0.0625],[-0.125,0.75,-0.125],[-0.0625,-0.125,-0.0625]],np.float32)


print('Starting hardware accelerated demo')

def hardwareDemo(kernel_g):
    numframes = 1000 # used to calculate the FPS
    fps = 0 # placehold
    fpsData = []
    start = time.time()
    for _ in range(numframes):
        inframe = hdmi_in.readframe()
        outframe = hdmi_out.newframe()
        xv2.filter2D(inframe, -1, kernel_g, dst=outframe, borderType=cv2.BORDER_CONSTANT)
        cv2.putText(outframe, '{:.2f}'.format(fps), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255)) # FPS on the frame       
        inframe.freebuffer()
              
        hdmi_out.writeframe(outframe)
    end = time.time()
    fps = numframes/(end-start)
    print('Frames per second: {:.2f} FPS'.format(fps))
    fpsData.append(fps)
    
def softwareDemo(kernel_g):
    numframes = 80 # used to calculate the FPS
    fps = 0 # placehold
    fpsData = []
    start = time.time()
    for _ in range(numframes):
        inframe = hdmi_in.readframe()
        outframe = hdmi_out.newframe()
        cv2.filter2D(inframe, -1, kernel_g, dst=outframe, borderType=cv2.BORDER_CONSTANT)
        cv2.putText(outframe, '{:.2f}'.format(fps), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255)) # FPS on the frame
        inframe.freebuffer()
        hdmi_out.writeframe(outframe)
    end = time.time()
    fps = numframes/(end-start)
    print('Frames per second: {:.2f} FPS'.format(fps))
    fpsData.append(fps)

    
print('Software driven Gaussian blur')
softwareDemo(gaussian)

print('Software driven Sobel Vertical')
softwareDemo(sobelV)

print('Gaussian blur')
hardwareDemo(gaussian)

print('Horizontal Sobel')
hardwareDemo(sobelH)

print('Vertical Sobel')
hardwareDemo(sobelV)




print('Imaging demo complete. Closing video input')

hdmi_out.close()
hdmi_in.close()

print(spaces)

print('Loading new PYNQ Overlay: FIR Filter')

import asciiplotlib as apl

def plot_to_notebook(time_sec,in_signal,n_samples,out_signal=None):
    fig = apl.figure()
    fig.plot(time_sec[:n_samples]*1e6,in_signal[:n_samples],label='Input signal', width=150, height=45)
    if out_signal is not None:
        fig.plot(time_sec[:n_samples]*1e6,out_signal[:n_samples],label='FIR output', width=150, height=45)
    fig.show()


# Total time
T = 0.002
# Sampling frequency
fs = 100e6
# Number of samples
n = int(T * fs)
# Time vector in seconds
t = np.linspace(0, T, n, endpoint=False)
# Samples of the signal
samples = 10000*np.sin(0.2e6*2*np.pi*t) + 1500*np.cos(46e6*2*np.pi*t) + 2000*np.sin(12e6*2*np.pi*t)
# Convert samples to 32-bit integers
samples = samples.astype(np.int32)

from scipy.signal import lfilter

coeffs = [-255,-260,-312,-288,-144,153,616,1233,1963,2739,3474,4081,4481,4620,4481,4081,3474,2739,1963,1233,616,153,-144,-288,-312,-260,-255]

import time
start_time = time.time()
sw_fir_output = lfilter(coeffs,70e3,samples)
stop_time = time.time()
sw_exec_time = stop_time - start_time
print('Software FIR execution time: ',sw_exec_time)

plot_to_notebook(t, samples, 1000)
time.sleep(5)


print(spaces)
print('Hardware Accelerated FIR Filter')


from pynq import Overlay
import pynq.lib.dma

# Load the overlay
overlay = Overlay('../fir_filter/fir_accel.bit')

# Load the FIR DMA
dma = overlay.filter.fir_dma

from pynq import Xlnk
import numpy as np

# Allocate buffers for the input and output signals
xlnk = Xlnk()
in_buffer = xlnk.cma_array(shape=(n,), dtype=np.int32)
out_buffer = xlnk.cma_array(shape=(n,), dtype=np.int32)

# Copy the samples to the in_buffer
np.copyto(in_buffer,samples)

# Trigger the DMA transfer and wait for the result
import time
start_time = time.time()
dma.sendchannel.transfer(in_buffer)
dma.recvchannel.transfer(out_buffer)
dma.sendchannel.wait()
dma.recvchannel.wait()
stop_time = time.time()
hw_exec_time = stop_time-start_time

plot_to_notebook(t, samples, 1000, out_signal=out_buffer)

print('Hardware FIR execution time: ',hw_exec_time)
print('Hardware acceleration factor: ',sw_exec_time / hw_exec_time)

# Free the buffers
in_buffer.close()
out_buffer.close()