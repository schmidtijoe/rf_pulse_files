"""
Script for converting .txt or.pta objects into serializable rfpf pulse class
"""
import pathlib as plib
import rfpf


path = plib.Path("./pulses/semc_pulse.txt").absolute()

rf = rfpf.RF.load_from_txt(path, bandwidth_in_Hz=1192.1693985, duration_in_us=1801, time_bandwidth=2.1470970867057)
rf.save("./pulses/semc_pulse_rfpf.pkl")

