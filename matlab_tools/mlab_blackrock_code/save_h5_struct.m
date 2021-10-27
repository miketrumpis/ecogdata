function save_h5_struct(fname, s, z)

% save an experiment in HDF5
if nargin < 3
    z = 5;
end

fcpl = H5P.create('H5P_FILE_CREATE');
fapl = H5P.create('H5P_FILE_ACCESS');
fid = H5F.create(fname,'H5F_ACC_TRUNC',fcpl,fapl);
H5F.close(fid);

write_struct(fname, '/', s, z);

function write_struct(fn, pth, s, z)
% workhorse function
% 
% Some conversion rules 
% * boolean type --> int8
% * "data" block --> row-by-row table
% * empty sets --> 1x1 table

snames = fieldnames(s);

for n = 1:length(snames)
    name = snames{n};
    f = getfield(s, name);
    %isdata = ~isstr(f);
    disp([pth name])
    if isstruct(f)
        disp(['writing substruct ' pth name])
        fid = H5F.open(fn, 'H5F_ACC_RDWR', ...
                       'H5P_DEFAULT');
        gid = H5G.create(fid, [pth name], 'H5P_DEFAULT', ...
                         'H5P_DEFAULT', 'H5P_DEFAULT');
        H5F.close(fid);
        
        subpth = [pth name '/'];        
        write_struct(fn, subpth, f, z);
    elseif isstr(f)
        h5writeatt(fn, pth, name, f)
    elseif iscell(f)
        %% xxx! only supporting cell strings
        write_str_cell(fn, [pth name], f);
    else
        % handle as an data block if the memory footprint is > 100 MB
        if prod(size(f)) * 8 > 100e6
        %if strcmp(name, 'data')
            % set up data a little different
            h5create(fn, [pth name], size(f), ...
                     'ChunkSize', [1 min(2^15, size(f,2))], ...
                     'Datatype', class(f), 'Deflate', z); 
        else
            if isempty(f)
                h5create(fn, [pth name], [1 1])
            else
                h5create(fn, [pth name], size(f));
            end
        end
        if strcmp(class(f), 'logical')
        %if strcmp(name, 'trig')
            % write this data a little different
            f = int8(f);
        end
        if ~isempty(f)
            h5write(fn, [pth name], f)
        end
    end
end


function write_str_cell(fn, pth, val)

%% dear The Mathworks, why is this not built in? 
fid = H5F.open(fn,'H5F_ACC_RDWR','H5P_DEFAULT');

% Set variable length string type
VLstr_type = H5T.copy('H5T_C_S1');
H5T.set_size(VLstr_type,'H5T_VARIABLE');

% Create a dataspace for cellstr
H5S_UNLIMITED = H5ML.get_constant_value('H5S_UNLIMITED');
dspace = H5S.create_simple(1,numel(val),H5S_UNLIMITED);

% Create a dataset plist for chunking
plist = H5P.create('H5P_DATASET_CREATE');
H5P.set_chunk(plist,2); % 2 strings per chunk

% Create dataset
%dset = H5D.create(fid,pth,VLstr_type,dspace,plist);
dset = H5D.create(fid,pth,VLstr_type,dspace,plist);

% Write data
H5D.write(dset,VLstr_type,'H5S_ALL','H5S_ALL','H5P_DEFAULT',val);

% Close file & resources
H5P.close(plist);
H5T.close(VLstr_type);
H5S.close(dspace);
H5D.close(dset);
H5F.close(fid);