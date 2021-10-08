"""
Plot signals from superdums .eis.dump files
Command line interface or direct python integration
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-
try: from ispw.libgpeis.tools.EisDump import DumpReader, DumpWriter
except: from EisDump import DumpReader, DumpWriter
import matplotlib.pyplot as plt
import cv2, numpy, optparse
import sys
from scipy import signal as sg
import json
import math
import os
import numpy as np

try:
    from tqdm import tqdm
except: #Fake tqdm for an alond py27 environment...
    class tqdm:
        def __init__(self, *args, **kwargs): pass
        def update(self, *args, **kwargs): pass
        def close(self, *args, **kwargs): pass



Camera_configs = {
    'basic': {
        'flip'        :  []
    ,   'axis2invert' : []
    }
,   'boilers':{
        'flip'       : [[1,2],]
    ,   'axis2invert': [2,]
    }
}


GYRO, FGYR, ACCL, MAGN, FRAME, STABORIENT = 'gyro', 'fgyr', 'accl', 'magn', 'frame', 'stabOrient'
TIMESTAMP, FRAMENO , CENTERTIME = 'timestamp', 'frameNo', 'centerTime'
GRAVITY , GRAVDEBUG = 'gravity', 'gravDebug'
X, Y, Z, W = 'x', 'y', 'z', "w"
ALLDIMENSIONS = [GYRO, FGYR, ACCL, MAGN, FRAME, STABORIENT, GRAVITY, FRAME, GRAVDEBUG]
def continuify(aa, ref):
    """
    Continuifies an axis-angle vector @aa with respect to a reference vector @ref
    """
    from math import pi, sqrt
    n1s = numpy.dot(ref, ref)
    n2s = numpy.dot(aa, aa)
    dot = numpy.dot(aa, ref)
    diffn = n1s + n2s - 2 * dot
    if diffn < 0.25 * pi ** 2 or (dot > 0 and diffn < pi ** 2):
        return aa

    if n2s == 0:
        if n1s > pi ** 2:
            n1 = sqrt(n1s)
            scale = round(n1 / (2 * pi)) * (2 * pi) / n1
            return scale * aa
        return aa

    n2 = sqrt(n2s)
    k = np.round((dot / n2 - n2) / (2 * pi))
    scale = 1 + 2 * pi * k / n2
    return scale * aa


def mulq(a, b):
    """
    Quaternion multiplication
    """
    return (
	a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
	a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
	a[0] * b[2] + a[2] * b[0] + a[3] * b[1] - a[1] * b[3],
	a[0] * b[3] + a[3] * b[0] + a[1] * b[2] - a[2] * b[1]
    )


def quatToAA(q):
    """
    Convert quaternion to Axis Angle
    """
    conv = 1
    _w_ = -1 if q[0] < -1 else (1 if q[0] > 1 else q[0])
    n = math.sqrt(1 - _w_ * _w_)
    if n == 0:
        return (2 * q[1] * conv, 2 * q[2] * conv, 2 * q[3] * conv)
    s = 2 * math.acos(_w_)
    if s > math.pi:
        s -= 2 * math.pi
    s *= conv
    return (q[1] * s / n,  q[2] * s / n,  q[3] * s / n)


def findFrame(frames, frameNo):
    i = min(frameNo, len(frames)-1)
    while frames[i][FRAMENO] > frameNo:
        i -= 1
    while frames[i][FRAMENO] < frameNo:
        i += 1
    assert(frames[i][FRAMENO] == frameNo)
    return frames[i]


class SuperDump:
    def __init__(self, filename, flips = [], axis2invert=[], endFrame = sys.maxsize):
        self.filesize = os.path.getsize(filename)
        self.data = DumpReader(filename)
        self.gyroTid  = self.data.findTypeId(GYRO)
        self.fgyrTid  = self.data.findTypeId(FGYR)
        self.acclTid  = self.data.findTypeId(ACCL)
        self.magnTid  = self.data.findTypeId(MAGN)
        self.frameTid = self.data.findTypeId(FRAME)
        self.storiTid = self.data.findTypeId(STABORIENT)
        self.gyro = []
        self.fgyr = []
        self.accl = []
        self.magn = []
        self.filtered_gyro = []

        self.gravity = []    # gravity from frame
        self.hforient = []   # orientation from gyro (high freq)
        self.lforient = []   # orientation from fgyr (low freq)
        self.storient = []   # stabilized orientation

        self.frames = []
        self.endFrame = endFrame
        #read data
        self.hfpose = numpy.eye(3)
        self.lfpose = numpy.eye(3)

        endFrameReached = False
        # heuristic on the eis.dump length to show a realistic progress bar
        pbar = tqdm(total=int(self.filesize*0.00118*0.71), desc="Loading .eis.dump %s"%os.path.basename(filename))

        while not self.data.eof() and not endFrameReached:
            tid, data = self.data.read()
            samples = self.data.parse([tid, data])
            name = self.data.types[tid]['name']
            # tsAcc = samples[0][TIMESTAMP] if tid == self.acclTid else 0 #OLD CODE ... UNKNOW REASON
            if tid == self.frameTid:
                self.frames += samples
            for sample in samples:
                # gyroscope or filtered gyroscope
                if tid == self.gyroTid or tid == self.fgyrTid:
                    gyr = eval("self."+name)
                    gyr.append(numpy.array([sample[TIMESTAMP] / 1e6, sample[X], sample[Y], sample[Z]]))
                    # simple gyro integration to get to an orientation estimate
                    freq = 6400 if name == "gyro" else 200
                    dt = (1.0 / freq) if len(gyr) <= 1 else (gyr[-1][0] - gyr[-2][0])
                    mat, _ = cv2.Rodrigues(-gyr[-1][1:] * dt)
                    pose = self.hfpose if name == "gyro" else self.lfpose
                    pose = numpy.dot(mat, pose)
                    rot1, _ = cv2.Rodrigues(pose)
                    if name==GYRO: self.hfpose = pose
                    else: self.lfpose=pose
                    orient = self.hforient if name == "gyro" else self.lforient
                    orient.append(numpy.array([sample[TIMESTAMP] / 1e6, rot1[0], rot1[1], rot1[2]]))
                    if len(gyr) > 1:
                        orient[-1][1:] = continuify(orient[-1][1:], orient[-2][1:])

                # accelerometer
                elif tid == self.acclTid:
                    self.accl.append(numpy.array([sample[TIMESTAMP] / 1e6, sample[X], sample[Y], sample[Z]]))
                    # self.accl.append(numpy.array([tsAcc / 1e6, sample[X], sample[Y], sample[Z]])) #OLD CODE ... UNKNOW REASON
                    # tsAcc += 1. / 200. #OLD CODE ... UNKNOW REASON

                # magnetometer
                elif tid == self.magnTid:
                    self.magn.append(numpy.array([sample[TIMESTAMP] / 1e6, sample[X], sample[Y], sample[Z], sample['rhall']]))

                # frame
                elif tid == self.frameTid:
                    if sample["frameNo"] > self.endFrame:
                        endFrameReached = True
                    if GRAVITY in sample:
                        self.gravity.append([sample[CENTERTIME] / 1e6, sample[GRAVITY][0], sample[GRAVITY][1], sample[GRAVITY][2]])

                # stabilized orientation
                elif tid == self.storiTid:
                    n = len(self.storient)
                    q = (sample[W], -sample[X], -sample[Y], -sample[Z])
                    if len(self.storient) == 0:
                        stbasis = [q[0], -q[1], -q[2], -q[3]]
                    q = mulq(q, stbasis)
                    aa = quatToAA(q)
                    self.storient.append(numpy.array([findFrame(self.frames, sample[FRAMENO])[CENTERTIME] / 1e6, aa[0], aa[1], aa[2]]))
                    if len(self.storient) > 1:
                        self.storient[-1][1:] = continuify(self.storient[-1][1:], self.storient[-2][1:])
            pbar.update(1)
        pbar.close()

        self.gyro = numpy.array(self.gyro).T
        self.fgyr = numpy.array(self.fgyr).T
        self.accl = numpy.array(self.accl).T
        self.magn = numpy.array(self.magn).T        
        self.gravity = numpy.asarray(self.gravity).T
        self.hforient = numpy.array(self.hforient).T
        self.lforient = numpy.array(self.lforient).T
        self.storient = numpy.array(self.storient).T

        self.flipLowFreqColumns(flips)
        self.invertLowFreqAxis(axis2invert)


    def flipLowFreqColumns(self, flips):
        for flip in flips:
            try: flipped = numpy.flip(flip)
            except:  flipped  = flip[::-1]
            self.lforient[flip, :] = self.lforient[flipped, :]
            self.fgyr[flip, :]     = self.fgyr[flipped, :]
            self.accl[flip, :]     = self.accl[flipped, :]

    def invertLowFreqAxis(self, axis2invert):
        for idx in axis2invert:
            self.lforient[idx, :] *= -1.
            self.fgyr[idx, :]     *= -1.
            self.accl[idx, :]     *= -1.


    def plotInfo(self, figIdx, suptitle, nChannel, x, y, color, label='', ylim=None):
        if figIdx not in self.figIdxList:
            fig = plt.figure(figIdx)
            splist = []
            for i in range(nChannel):
                splist.append(fig.add_subplot(100*nChannel + 10 + i + 1, sharex=self.sharedAxisHandle))
                if self.sharedAxisHandle is None: self.sharedAxisHandle  = splist[0]
            self.figList.append((fig, splist))
            self.figIdxList.append(figIdx)
        else: #retrieve existing figure and subplots
            fig, splist = self.figList[self.figIdxList.index(figIdx)]

        fig.suptitle(suptitle)
        for i in range(nChannel):
            splist[i].plot(x, y[i, :], color, label=label)
            splist[i].grid(True)
            if ylim:
                splist[i].set_ylim(ylim)
        if label != '': splist[i].legend()
        splist[i].set_xlabel('Time, s')



    def save_orientation(self,time, orientation, freq):

        dictData = {"orientation": []}
        for i in range(len(time)):
            dictData["orientation"].append(  
                                            {"x": float(orientation[0, i]),
                                            "y":  float(orientation[1, i]),
                                            "z":  float(orientation[2, i]),
                                            "timestamps": float( time[i])})
        with open(filename + "_orientation_" +str(freq) +".json", 'w') as json_file:
            json.dump(dictData, json_file)


    def save_gravity(self,time, gravity):

        dictData = {"gravity": []}
        for i in range(len(time)):
            dictData["gravity"].append(  
                                            {"x": float(gravity[0, i]),
                                            "y":  float(gravity[1, i]),
                                            "z":  float(gravity[2, i]),
                                            "timestamps": float( time[i])})
        with open(filename + "_gravity.json", 'w') as json_file:
            json.dump(dictData, json_file)

    


    def viewer(self, dimensionsToPlot = ALLDIMENSIONS[:-2]):
        """
        Plot multiple figures constrained by the absciss axis.
        Allows to zoom simulateneouslyon the same trim of the signal
        Use dimensionsToPlot list to select which dimension you want to plot
        choose between [GYRO, FGYR, ACCL, MAGN, FRAME, STABORIENT, GRAVITY, FRAME, GRAVDEBUG]
        If the superdump does not contain the signal track, the key you ask in dimensionToPlot will be ignored
        """
        self.figList, self.figIdxList, self.sharedAxisHandle = [], [], None
        if self.frameTid:
            self.frGraph=numpy.array([(x[CENTERTIME]/1000000.,0) for x in self.frames]).transpose()
            # self.plotInfo( dimensionsToPlot.index(FRAME), u'Frame sampling', 1, self.frGraph[0,:], self.frGraph[1:,:], '+', label="Frame sampling")
            linindex = numpy.arange(len(self.frGraph[1, :]))
            if FRAME in dimensionsToPlot:
                self.plotInfo(dimensionsToPlot.index(FRAME), u'Frame sampling', 1, self.frGraph[0, :], numpy.array([linindex]) ,'-o', label="Frame sampling")
        if STABORIENT in dimensionsToPlot:
            figorient = dimensionsToPlot.index(STABORIENT)
            if self.fgyrTid and len(self.lforient) > 0:
                self.plotInfo(figorient, u'Orientation [°]', 3, self.lforient[0,:], 180./numpy.pi * self.lforient[1:,:], 'b:', label='200 Hz')
                #self.save_orientation(self.lforient[0,:], 180./numpy.pi * self.lforient[1:,:], 200)
            if self.gyroTid and len(self.hforient) > 0:
                self.plotInfo(figorient, u'Orientation [°]', 3, self.hforient[0,:], 180./numpy.pi*self.hforient[1:,:], 'b-', label='6.4 KHz')
                #self.save_orientation(self.hforient[0,:], 180./numpy.pi * self.hforient[1:,:], 6400)
            if self.storiTid and len(self.storient) > 0:
                self.plotInfo(figorient, u'Orientation [°]', 3, self.storient[0,:], 180./numpy.pi * self.storient[1:,:], 'g-.', label='HS signal')
            if FRAME in dimensionsToPlot and (self.frameTid):
                self.plotInfo(figorient, u'Orientation [°]', 1, self.frGraph[0, :], self.frGraph[1:, :], '.', label="Frame sampling")


        if GYRO in dimensionsToPlot and self.gyroTid and len(self.gyro) > 0:
            self.plotInfo(dimensionsToPlot.index(GYRO), u'Gyroscope angular rate [°/s]', 3, self.gyro[0,:], numpy.rad2deg(self.gyro[1:,:]), 'm', label='6.4 KHz')

        if FGYR in dimensionsToPlot and self.fgyrTid and len(self.fgyr) > 0:
            self.plotInfo(dimensionsToPlot.index(GYRO), u'Gyroscope angular rate [°/s]', 3, self.fgyr[0,:], numpy.rad2deg(self.fgyr[1:,:]), 'm:', label='200 Hz')

        if ACCL in dimensionsToPlot and self.acclTid and len(self.accl) > 0:
            self.plotInfo(dimensionsToPlot.index(ACCL),'Accelerometer signal', 3, self.accl[0,:], self.accl[1:,:], 'r')

        if MAGN in dimensionsToPlot and self.magnTid and len(self.magn) > 0:
            print("magneto")
            self.plotInfo(dimensionsToPlot.index(MAGN),'Magnetometer signal', 4, self.magn[0,:], self.magn[1:,:], 'g')

        if GRAVITY in dimensionsToPlot and  len(self.gravity)>0:
            self.plotInfo(dimensionsToPlot.index(GRAVITY),'Gravity', 3, self.gravity[0,:], self.gravity[1:,:], 'k', ylim=(-1, 1))
            #self.save_gravity(self.gravity[0,:],self.gravity[1:,:])

        if GRAVDEBUG in dimensionsToPlot and self.acclTid and self.gyroTid:
            from scipy.interpolate import interp1d
            gravity = numpy.zeros(self.accl.shape)
            idxinterpol = interp1d(self.hforient[0,:], range(len(self.hforient[0,:])), bounds_error=False)
            n = 0
            for i in range(self.accl.shape[1]):
                from math import isnan
                t = idxinterpol(self.accl[0,i])
                if isnan(t): continue
                t = int(t)
                pose, _ = cv2.Rodrigues(self.hforient[1:,t].astype(numpy.float32))
                gravity[0,n] = self.accl[0,i]
                gravity[1:,n] = numpy.dot(pose.T, self.accl[1:,i])

                n += 1

            if True: #low pass on rotated accelerometer
                sizeFilt = 100
                halfSize = sizeFilt // 2
                gravity[1, halfSize:-halfSize] = numpy.convolve(gravity[1, :], numpy.ones((sizeFilt, )) / sizeFilt, mode="same")[halfSize:-halfSize]
                gravity[2, halfSize:-halfSize] = numpy.convolve(gravity[2, :], numpy.ones((sizeFilt, )) / sizeFilt, mode="same")[halfSize:-halfSize]
                gravity[3, halfSize:-halfSize] = numpy.convolve(gravity[3, :], numpy.ones((sizeFilt, )) / sizeFilt, mode="same")[halfSize:-halfSize]
            self.plotInfo(dimensionsToPlot.index(GRAVDEBUG), 'Gravity debug estimation', 3, gravity[0, :n], gravity[1:, :n], 'c')


            try:gnorm = numpy.linalg.norm(gravity[1:,:n],  axis=0)
            except: gnorm = numpy.apply_along_axis(numpy.linalg.norm, 0, gravity[1:,:n])
            self.plotInfo(dimensionsToPlot.index(GRAVDEBUG)+1, 'Gravity norm', 1, gravity[0,:n], numpy.array([gnorm,]), 'g')

        plt.show()


    def save_important_features(self):

        dictData = {"filtered_gyro": [], "gravity": [], "orientation[°]": [], "accl": []}

        #filtered_gyro
        for i in range(len(self.filtered_gyro)):
            dictData["filtered_gyro"].append({"x":  float(self.filtered_gyro[i, 0]),
                                        "y":  float(self.filtered_gyro[i, 1]),
                                        "z":  float(self.filtered_gyro[i, 2])})

        time = self.gravity[0,:]
        data = self.gravity[1:,:]
        for i in range(len(time)):
            dictData["gravity"].append({"x": float(data[0, i]),
                                        "y":  float(data[1, i]),
                                        "z":  float(data[2, i]),
                                        "timestamps": float(time[i])})

        time = self.hforient[0,:]
        data = 180./numpy.pi * self.hforient[1:,:]
        for i in range(len(time)):
            dictData["orientation[°]"].append({"x":  float(data[0, i]),
                                        "y":  float(data[1, i]),
                                        "z":  float(data[2, i]),
                                        "timestamps": float(time[i])})

        time = self.accl[0,:]
        data = self.accl[1:,:]
        for i in range(len(time)):
            dictData["accl"].append({"x":  float(data[0, i]),
                                    "y":  float(data[1, i]),
                                    "z":  float(data[2, i]),
                                    "timestamps": float(time[i])})

        with open(filename + "_features.json", 'w') as json_file:
            json.dump(dictData, json_file)





    def freqAnalysis(self, subSamplingFactor, filterSignal, samplingFreq=6400, dimension = X):
        """
        Analyze gyroscope
        """
        print(samplingFreq)
        indexToStudy = [None, X, Y, Z].index(dimension)
        cutFreq = samplingFreq // (2 * subSamplingFactor)
        nyquistFreq = samplingFreq // 2
        subSampledFreq = samplingFreq // subSamplingFactor

        w = sg.firwin(65, cutFreq, fs=samplingFreq)

        if (filterSignal):
            fullFilteredData = [numpy.convolve(self.gyro[1], w, mode='same'),
                                numpy.convolve(self.gyro[2], w, mode='same'),
                                numpy.convolve(self.gyro[3], w, mode='same')]

            fullFilteredData = numpy.array(fullFilteredData, numpy.float32).T
            self.filtered_gyro = fullFilteredData
            #dictData = {"data": []}
            #for i in range(len(fullFilteredData)):
                #dictData["data"].append({"x": float(fullFilteredData[i, 0]),
                                         #"y": float(fullFilteredData[i, 1]),
                                         #"z": float(fullFilteredData[i, 2])})
           # with open(filename + "_fgyr_" + str(samplingFreq)+ ".json", 'w') as json_file:
                #json.dump(dictData, json_file)

        

        # Analyse 1s of the signal


        t = 10
        n = samplingFreq * 10
        nSub = subSampledFreq * 10

        gyroFilt = numpy.convolve(self.gyro[indexToStudy], w, mode='same')
        fft_filtgyro = numpy.fft.fft(gyroFilt[n:n + samplingFreq] * numpy.hanning(samplingFreq))

        gyroFiltDecimated = gyroFilt[::subSamplingFactor]

        fft_hfgyro = numpy.fft.fft(self.gyro[indexToStudy, n:n + samplingFreq] * numpy.hanning(samplingFreq))
        fft_lfgyro = numpy.fft.fft(gyroFiltDecimated[nSub: nSub + subSampledFreq] * subSamplingFactor * numpy.hanning(subSampledFreq))

        print("Residual energy: %.3f" % (float(100 * numpy.sum(fft_hfgyro.real[cutFreq:nyquistFreq]**2 + fft_hfgyro.imag[cutFreq:nyquistFreq]**2) / numpy.sum(fft_hfgyro.real[:nyquistFreq]**2 + fft_hfgyro.imag[:nyquistFreq]**2))) + str('%'))

        #fig1, ax1 = plt.subplots()
        #ax1.plot(numpy.arange(0, 1000, 1000. / samplingFreq), self.gyro[indexToStudy, n:n + samplingFreq], label="Raw signal")
        #ax1.plot(numpy.arange(0, 1000, 1000. / samplingFreq), gyroFilt[n:n + samplingFreq], label="Filtered signal")

        #ax1.set_title("Raw Gyro signal")
        #ax1.set_xlabel("Time (ms)")
        #ax1.set_ylabel("Gyro (%s-axis)"%dimension)
        #plt.legend()

        #fig2, ax2 = plt.subplots()
        #ax2.plot(numpy.sqrt(fft_hfgyro.real[:nyquistFreq]**2 + fft_hfgyro.imag[:nyquistFreq]**2), label="High frequency spectrum")
        #ax2.plot(numpy.sqrt(fft_filtgyro.real[:nyquistFreq]**2 + fft_filtgyro.imag[:nyquistFreq]**2), label="Filtered High frequency spectrum")
        #ax2.plot(numpy.sqrt(fft_lfgyro.real[:cutFreq]**2 + fft_lfgyro.imag[:cutFreq]**2), label="Low frequency spectrum")
        #ax2.set_xlabel("Frequency (Hz)")
        #ax2.set_ylabel("Energy")
        #ax2.set_title("Spectrum of the gyroscope signal")
        #plt.grid(True)
        #ax2.set_yscale('log')
        #ax2.legend()
        #plt.show()

if __name__ == "__main__":
    # parse arguments
    parser = optparse.OptionParser(description  = "Plots dumped IMU signals.", usage = "%prog [options] eis_dump.bin")
    parser.add_option("-e", "--endFrame", type=int, default=sys.maxsize)
    parser.add_option("-c", "--cameraConfig", default="boilers", choices=["basic", "boilers"])
    parser.add_option("-f", "--freqAnalysis", action="store_const", default=False, const=True, help="Analyse the spectrum of the signal.")
    parser.add_option("-F", "--filterSignal", action="store_const", default=False, const=True, help="Filter the signal with a FIR and outputs a .json with the 6400Hz filtered track.")
    parser.add_option("-s", "--subsampling", type=int, default=4, help="Subsampling factor. Needed to design the FIR.")
    parser.add_option("-o", "--gyroOrientation", action="store_const", default=False, const=True, help="Store the stabilised gyro orientation vectors")
    (Options, args) = parser.parse_args()
    if len(args)>0: filename = args[0]
    else:
        # filename="/Exchange/gladysmarc/HandHeldCameraMotion/GX010582.MP4.eis_dump.bin"
        filename = r"/Public/Image/Jira/Vercors/STAB-400_twitch/superdumps/GX010154.MP4.eis_dump.bin"
        import warnings
        warnings.warn("Using a default superdump, please provide the path to the .eis.dump you want to visualize")
    assert os.path.exists(filename), "%s does not exist"%filename
    dump = SuperDump(filename, Camera_configs[Options.cameraConfig]['flip'], Camera_configs[Options.cameraConfig]['axis2invert'], Options.endFrame)

    if (Options.freqAnalysis):
        dump.freqAnalysis(Options.subsampling, Options.filterSignal)
    dump.save_important_features()
    #dump.viewer()