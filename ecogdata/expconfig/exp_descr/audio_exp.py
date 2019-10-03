"A module to describe auditory experiments."

import numpy as np
from matplotlib.cm import jet

from .base_exp import StimulatedExperiment

__all__ = ['TonotopyExperiment']


class TonotopyExperiment(StimulatedExperiment):
    """
    An experiment with multiple tone presentations that vary in
    pitch (labeled "tones") and amplitude (labeled "amps")
    """

    # These patterns were used in one set of exps..
    fixed_tones_pattern = \
        (2831, 8009, 5663, 8009, 2831, 1000, 1000, 16018, 1415, 4004,
         4004, 22651, 708, 1415, 500, 5663, 2000, 1000, 5663, 22651,
         16018, 2000, 500, 32036, 2831, 708, 500, 11326, 11326, 708,
         1415, 2000, 16018, 32036, 32036, 8009, 4004, 11326, 22651)

    fixed_amps_pattern = (30, 50, 30, 30, 70, 70, 50, 70, 30, 30, 70, 30, 30,
                          50, 50, 70, 70, 30, 50, 50, 30, 30, 30, 50, 50, 50,
                          70, 30, 50, 70, 70, 50, 50, 30, 70, 70, 50, 70, 70)

    def __init__(self, **kwargs):
        kwargs.setdefault('condition_order', ('tones', 'amps'))
        super(TonotopyExperiment, self).__init__(**kwargs)

    def stim_str(self, n, mpl_text=False):
        tone = self.tones[n]
        amp = self.amps[n]

        tone_khz = tone // 1000
        tone_dec = tone - tone_khz * 1000
        tone_dec = int(float(tone_dec) / 100 + 0.5)

        s = '%d.%d KHz' % (tone_khz, tone_dec)
        s = s + ' (%d)' % amp
        if mpl_text:
            import matplotlib as mpl
            u_amps = np.unique(self.amps)
            ctab = jet(np.linspace(0.1, 0.9, len(u_amps)))
            cidx = u_amps.searchsorted(amp)
            return mpl.text.Text(text=s, color=ctab[cidx])
        else:
            return s

    @classmethod
    def from_repeating_sequences(
            cls, time_stamps, sequences, condition_order=(), **kwargs
    ):

        # do this for historical consistency?
        if not sequences:
            sequences = dict(tones=cls.fixed_tones_pattern,
                             amps=cls.fixed_amps_pattern)

        # also enforce default condition order here
        if not len(condition_order):
            condition_order = 'tones', 'amps'

        return super(TonotopyExperiment, cls).from_repeating_sequences(
            time_stamps, sequences, condition_order=condition_order, **kwargs
        )
