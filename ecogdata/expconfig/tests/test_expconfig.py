import random
import string
from ..config_tools import *


def randomword(length):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))


def test_sessions_to_delta():
    sessions = ['2016-07-01', '2016-07-02', '2016-07-17', '2016-07-05']
    days = [0, 1, 16, 4]
    day_names = ['Day {0:d}'.format(d) for d in days]

    # test basic
    deltas = sessions_to_delta(sessions)
    assert day_names == deltas

    deltas = sessions_to_delta(sessions, sortable=True)
    nums = [int(s.split(' ')[-1]) for s in sorted(deltas)]
    assert sorted(days) == sorted(nums), 'Day names sorted'

    deltas = sessions_to_delta(sessions, num=True)
    assert deltas == days, 'Day numbers correct'

    # test more complicated strings
    r_sessions = [randomword(5) + s + randomword(5) for s in sessions]
    deltas = sessions_to_delta(r_sessions)
    assert day_names == deltas

    deltas = sessions_to_delta(r_sessions, sortable=True)
    nums = [int(s.split(' ')[-1]) for s in sorted(deltas)]
    assert sorted(days) == sorted(nums), 'Day names sorted'

    deltas = sessions_to_delta(r_sessions, num=True)
    assert deltas == days, 'Day numbers correct'

    # test arbitrary reference
    ref = randomword(5) + '2016-06-15' + randomword(5)
    deltas = sessions_to_delta(sessions, reference=ref, num=True)
    assert deltas == [d + 16 for d in days]
