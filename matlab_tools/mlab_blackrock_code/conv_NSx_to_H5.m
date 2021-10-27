function conv_NSx_to_H5(nsfile, h5file, chans, all_stamps)

[a b c] = fileparts(nsfile);
nev_file = [a '/' b '.nev'];
nev = openNEV(nev_file, 'nomat', '8bits', 'nosave');
Fs = double(nev.MetaTags.SampleRes);

% Movshon data session technically had triggers on every frame, but
% trial initiation was marked with a different code. The Froemke
% data only had triggers marked for trial initiation (so we use all
% of them)
if (nargin > 3) & all_stamps
    pos_edge = round(nev.Data.SerialDigitalIO.TimeStamp);
else
    % look for special code
    stim_idx = find(nev.Data.SerialDigitalIO.UnparsedData==2);
    pos_edge = round(nev.Data.SerialDigitalIO.TimeStamp(stim_idx));
end

if nargin > 2
    cmin = max(chans(1), 1);
    cmax = min(chans(2), 96);
    c_str = sprintf('c:%d:%d', cmin, cmax);
    ns = openNSx(nsfile, c_str);
else
    ns = openNSx(nsfile);
end

sz = size(ns.Data);
disp(sprintf('loaded file of size (%d, %d) sampled at %1.1f', ...
             sz(1), sz(2), Fs))
% make a quick struct with .data and .trig_idx and then
% use save_h5_struct utility

s = struct;
s.data = ns.Data;
s.trig_idx = double(pos_edge);
s.Fs = Fs;
s.nev_path = nev_file;

save_h5_struct(h5file, s, 0);