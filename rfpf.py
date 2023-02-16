import dataclasses as dc
import logging

import numpy as np
import pathlib as plib
import typing
import pickle
import pandas as pd

logModule = logging.getLogger(__name__)


@dc.dataclass
class GlobalSystem:
    gamma_Hz = 42577478.518     # Hz/T
    max_grad_mT = 40.0          # mT/m

    def check_bandwidth_for_slice(self, slice_thickness_in_mm):
        max_bandwidth = self.gamma_Hz * self.max_grad_mT * 1e-3 * slice_thickness_in_mm * 1e-3
        return max_bandwidth
        

@dc.dataclass
class RF:
    filename: str = ""
    bandwidth_in_Hz: float = 1000.0
    duration_in_us: int = 2000
    time_bandwidth: float = bandwidth_in_Hz * duration_in_us * 1e-6
    num_samples: int = duration_in_us

    amplitude: np.ndarray = np.zeros(num_samples)
    phase: np.ndarray = np.zeros(num_samples)

    def __post_init__(self):
        # check array sizes - for some reason this is not working properly when creating class with input args
        if self.amplitude.shape[0] != self.num_samples:
            self.amplitude = np.zeros(self.num_samples)
        if self.phase.shape[0] != self.num_samples:
            self.phase = np.zeros(self.num_samples)

    def display(self):
        columns = {
            "Bandwidth": ["Hz", self.bandwidth_in_Hz],
            "Duration": ["us", self.duration_in_us],
            "Time-Bandwidth": ["1", self.time_bandwidth],
            "Number of Samples": ["1", self.num_samples]
        }
        display = pd.DataFrame(columns, index=["units", "value"])
        print(display)

    def set_shape_on_raster(self, raster_time):
        # interpolate shape to duration raster
        N = int(self.duration_in_us * 1e-6 / raster_time)
        self.amplitude = np.interp(
            np.linspace(0, self.amplitude.shape[0], N),
            np.arange(self.amplitude.shape[0]),
            self.amplitude
        )
        self.phase = np.interp(
            np.linspace(0, self.phase.shape[0], N),
            np.arange(self.phase.shape[0]),
            self.phase
        )
        self.num_samples = N

    def save(self, f_name: typing.Union[str, plib.Path]):
        p_name = plib.Path(f_name).absolute()
        # check existing
        if p_name.suffixes:
            p_name.parent.mkdir(parents=True, exist_ok=True)
        else:
            p_name.mkdir(parents=True, exist_ok=True)
        if p_name.is_file():
            p_name.unlink()
        with open(p_name, "wb") as p_file:
            pickle.dump(self, p_file)

    @classmethod
    def load(cls, f_name):
        p_name = plib.Path(f_name).absolute()
        if p_name.is_file():
            with open(p_name, "rb") as p_file:
                rf_cls = pickle.load(p_file)
        return rf_cls

    @classmethod
    def load_from_txt(cls, f_name: typing.Union[str, plib.Path],
                      bandwidth_in_Hz: float = None,
                      duration_in_us: int = None,
                      time_bandwidth: float = None,
                      num_samples: int = None):
        """
        read file from .txt or .pta. Need to fill the additional specs.
        :param f_name: file name
        :param bandwidth_in_Hz: Bandwidth in Hertz (optional if duration and tbw provided)
        :param duration_in_us:  Duration in microseconds (optional if bandwidth and tbw provided)
        :param time_bandwidth: Time Bandwidth product unitless (optional if bandwidth and duration provided)
        :param num_samples: number of samples of pulse optional, if None pulse sampled per microsecond
        :return:
        """
        if bandwidth_in_Hz is None:
            if duration_in_us is None:
                err = f"No bandwidth provided: provide Duration and time-bandwidth product"
                logModule.error(err)
                raise ValueError
            bandwidth_in_Hz = time_bandwidth / duration_in_us * 1e6
            rf_cls = cls(bandwidth_in_Hz=bandwidth_in_Hz, duration_in_us=duration_in_us)
        elif duration_in_us is None:
            if time_bandwidth is None:
                err = f"No duration provided: provide Bandwidth and time-bandwidth product"
                logModule.error(err)
                raise ValueError
            duration_in_us = int(1e6 * time_bandwidth / bandwidth_in_Hz)
            rf_cls = cls(bandwidth_in_Hz=bandwidth_in_Hz, duration_in_us=duration_in_us)
        else:
            rf_cls = cls(bandwidth_in_Hz=bandwidth_in_Hz, duration_in_us=duration_in_us)
        if num_samples is None:
            num_samples = duration_in_us
        t_name = plib.Path(f_name).absolute()
        if t_name.is_file():
            # load file content
            ext_file = plib.Path(t_name)
            with open(ext_file, "r") as f:
                content = f.readlines()

            # find line where actual data starts
            start_count = -1
            while True:
                start_count += 1
                line = content[start_count]
                start_line = line.strip().split('\t')[0]
                if start_line.replace('.', '', 1).isdigit():
                    break

            # read to array
            if content.__len__() != num_samples:
                logModule.info(f"file content not matching number of samples given {num_samples}")
                num_samples = content.__len__() - start_count
                logModule.info(f"adjusting number of samples to {num_samples}")
                rf_cls.num_samples = num_samples
            content = content[start_count:start_count+num_samples]
            temp_amp = np.array([line.strip().split('\t')[0] for line in content])
            temp_phase = np.array([line.strip().split('\t')[1] for line in content])

            rf_cls.amplitude = temp_amp
            rf_cls.phase = temp_phase
        if rf_cls.amplitude.shape[0] != rf_cls.phase.shape[0]:
            err = "shape of amplitude and phase do not match"
            logModule.error(err)
            raise AttributeError(err)
        return rf_cls

    def resample_to_duration(self, duration_in_us: int):
        # want to use pulse with different duration,
        # ! in general time bandwidth properties do not have to go linearly with duration
        # we have a pulse with given tb prod at given duration,
        # if we change the duration the bandwidth is expected to change accordingly
        self.set_bandwidth_Hz(self.time_bandwidth / duration_in_us * 1e6)
        self.set_duration_us(duration=duration_in_us)

    def set_bandwidth_Hz(self, bw: float):
        self.bandwidth_in_Hz = bw

    def set_duration_us(self, duration: int):
        self.duration_in_us = duration

    def get_num_samples(self) -> int:
        return self.num_samples

    def get_dt_sampling_in_s(self) -> float:
        return 1e-6 * self.duration_in_us / self.num_samples

    def get_dt_sampling_in_us(self) -> float:
        return self.duration_in_us / self.num_samples


if __name__ == '__main__':
    rf = RF()
    s_path = plib.Path("pulses/test.pkl").absolute()
    rf.save(s_path)

    rf_l = RF.load(s_path)

    rf_t = RF.load_from_txt("pulses/semc_pulse.txt", bandwidth_in_Hz=1192.1694, duration_in_us=1801)
    a = rf
