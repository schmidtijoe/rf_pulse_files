import dataclasses as dc
import logging

import numpy as np
import pathlib as plib
import inspect
import typing
import pickle

logModule = logging.getLogger(__name__)


def load_external_rf(rf_file) -> np.ndarray:
    """
        if pulse profile is provided, read in
        :param filename: name of file (txt or pta) of pulse
        :return: pulse array, pulse length
        """
    # load file content
    ext_file = plib.Path(rf_file)
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
    content = content[start_count:]
    temp = [line.strip().split('\t')[0] for line in content]

    pulseShape = np.array(temp, dtype=float)
    return pulseShape


@dc.dataclass
class ExtRf:
    filename: str = ""
    bandwidth_in_Hz: float = 1000.0
    duration_in_us: int = 2000
    time_bandwidth: float = bandwidth_in_Hz * duration_in_us * 1e-6

    amplitude: np.ndarray = np.zeros(duration_in_us)
    phase: np.ndarray = np.zeros(duration_in_us)

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

    def _att_to_dict(self):
        atts = inspect.getmembers(self)
        d = {}
        for a in atts:
            if not inspect.isroutine(a[1]):
                if not a[0].startswith('__') and not a[0].endswith('__') and not a[0].startswith('_'):
                    val = a[1]
                    if isinstance(val, np.ndarray):
                        val = a[1].tolist()
                    d.__setitem__(a[0], val)
        return d

    def save(self, f_name: typing.Union[str, plib.Path]):
        p_name = plib.Path(f_name).absolute()
        if p_name.parent.is_dir():
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
            content = content[start_count:start_count+num_samples]
            temp_amp = np.array([line.strip().split('\t')[0] for line in content])
            temp_phase = np.array([line.strip().split('\t')[1] for line in content])

            rf_cls.amplitude = temp_amp
            rf_cls.phase = temp_phase
        return rf_cls


if __name__ == '__main__':
    rf = ExtRf()
    s_path = plib.Path("pulses/test.pkl").absolute()
    rf.save(s_path)

    rf_l = ExtRf.load(s_path)

    rf_t = ExtRf.load_from_txt("pulses/semc_pulse.txt", bandwidth_in_Hz=1192.1694, duration_in_us=1801)
    a = rf
