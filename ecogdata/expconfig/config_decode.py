import os
from ecogdata.util import Bunch


__all__ = ['Parameter', 'TypedParam', 'BoolOrNum', 'NSequence', 'NoneOrStr', 'Path', 'parse_param',
           'uniform_bunch_case']


class Parameter:
    "A pass-thru parameter whose value is the command (a string)"

    def __init__(self, command, default=''):
        self.command = command
        self.default = default

    def value(self):
        if self.command:
            return self.command
        return self.default

    @classmethod
    def with_default(cls, value, *args):
        def _gen_param(command):
            return cls(command, *args, default=value)
        return _gen_param


class TypedParam(Parameter):
    "A simply typed parameter that can be evaluated by a 'type'"

    def __init__(self, command, ptype, default=''):
        super(TypedParam, self).__init__(command, default=default)
        self.ptype = ptype

    @staticmethod
    def from_type(ptype, default=''):
        def _gen_param(command):
            return TypedParam(command, ptype, default=default)

        return _gen_param

    def value(self):
        if self.command:
            return self.ptype(self.command)
        return self.ptype(self.default)


class BoolOrNum(Parameter):
    "A value that is a boolean (True, False) or a number"

    def value(self):
        cmd = super(BoolOrNum, self).value().lower()
        if cmd in ('true', 'false'):
            return cmd == 'true'
        return float(self.command)


class NSequence(Parameter):
    "A sequence of numbers (integers if possible, else floats)"

    def value(self):
        cmd = super(NSequence, self).value()
        cmd = cmd.strip('(').strip(')').strip('[').strip(']').strip(',')
        cmd = cmd.replace(' ', '')
        if len(cmd):
            try:
                return list(map(int, cmd.split(',')))
            except ValueError:
                return list(map(float, cmd.split(',')))
        return ()


class NoneOrStr(Parameter):
    """
    A single value that is None (null) or something not null.
    Will return a string here.
    """

    def value(self):
        cmd = super(NoneOrStr, self).value()
        return None if cmd.lower() == 'none' else cmd


class Path(NoneOrStr):
    """
    Speific string that may include ~
    """

    def value(self):
        val = super(Path, self).value()
        if val is not None:
            # catch one pernicious corner case
            if len(val) > 1 and val[0] == os.path.sep and val[1] == '~':
                val = val[1:]
            val = os.path.expanduser(val)
        return val


def parse_param(name, command, table):
    p = table.get(name.lower(), Parameter)(command)
    return p.value()


def uniform_bunch_case(b):
    b_lower = Bunch()
    for k, v in b.items():
        if isinstance(k, str):
            b_lower[k.lower()] = v
        else:
            b_lower[k] = v
    return b_lower
