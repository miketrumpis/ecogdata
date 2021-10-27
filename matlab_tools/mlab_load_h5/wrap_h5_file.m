function s = wrap_h5_file(f, xpose)

s = struct;
s.OSR = h5read(f, '/OSR');
s.sampRate = h5read(f, '/sampRate');
s.numRow = h5read(f, '/numRow');
s.numChan = h5read(f, '/numChan');
s.Fs = h5read(f, '/Fs');

data = h5read(f, '/data');

if (nargin > 1) & (xpose > 0) & size(data,2) < size(data,1)
    s.data = data';
else
    s.data = data;
end

