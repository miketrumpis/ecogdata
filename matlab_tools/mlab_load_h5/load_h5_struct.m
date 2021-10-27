function h5struct = load_h5_struct(fname)

h5struct = group_to_struct(fname, '/');

function s = group_to_struct(fname, group)

s = struct;

h5s = h5info(fname, group);

% search for all datasets and attributes
for n = 1:length(h5s.Datasets)
    name = h5s.Datasets(n).Name;
    val = h5read(fname, [group name]);
    s = setfield(s, name, val);
end

for n = 1:length(h5s.Attributes)
    s = setfield(s, h5s.Attributes(n).Name, ...
                    h5s.Attributes(n).Value);
end

for n = 1:length(h5s.Groups)
    name = h5s.Groups(n).Name;
    subgroup = [name '/'];
    path_levels = strsplit(name, '/');
    s = setfield(s, path_levels{end}, ...
                    group_to_struct(fname, subgroup));
end

