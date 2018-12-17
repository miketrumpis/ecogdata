import os
from ecogdata.util import Bunch


__all__ = ['Parameter', 'TypedParam', 'BoolOrNum', 'NSequence', 'NoneOrStr', 'Path', 'parse_param',
           'uniform_bunch_case']


class Parameter(object):
    "A pass-thru parameter whose value is the command (a string)"

    def __init__(self, command):
        self.command = command

    def value(self):
        return self.command


class TypedParam(Parameter):
    "A simply typed parameter that can be evaluated by a 'type'"

    def __init__(self, command, ptype):
        super(TypedParam, self).__init__(command)
        self.ptype = ptype

    @staticmethod
    def from_type(ptype):
        def _gen_param(command):
            return TypedParam(command, ptype)

        return _gen_param

    def value(self):
        return self.ptype(self.command)


class BoolOrNum(Parameter):
    "A value that is a boolean (True, False) or a number"

    def value(self):
        cmd = self.command.lower()
        if cmd in ('true', 'false'):
            return cmd == 'true'
        return float(self.command)


class NSequence(Parameter):
    "A sequence of numbers (integers if possible, else floats)"

    def value(self):
        cmd = self.command.strip('(').strip(')').strip('[').strip(']').strip(',')
        cmd = cmd.replace(' ', '')
        if len(cmd):
            try:
                return map(int, cmd.split(','))
            except ValueError:
                return map(float, cmd.split(','))
        return ()


class NoneOrStr(Parameter):
    """
    A single value that is None (null) or something not null.
    Will return a string here.
    """

    def value(self):
        return None if self.command.lower() == 'none' else self.command


class Path(NoneOrStr):
    """
    Speific string that may include ~
    """

    def value(self):
        val = super(Path, self).value()
        if val is not None:
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

