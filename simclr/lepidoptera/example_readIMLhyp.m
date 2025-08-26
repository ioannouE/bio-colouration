% here is a simple routine to test the function readIMLhyp.m

% ===== name/address of a hyperspectral scan =====
name = 'Arctia caja_1D_030717.bil';
address = pwd;

% ===== read scan =====
if exist([address , '\', name], 'file')
    [im, wvls, scan, gain] = readIMLhyp( [address, '\'], name);
else
    error(['No scan named ' name ' in folder ' address])
end


