% Specify the file paths
pdmFilePath = 'pdm1a.pdm';
% wavFilePath = 'path/to/your/output.wav';

% Parameters
bitDepth = 16; % Output bit depth
fs = 44100;  % Sampling frequency in Hz (adjust according to your file)

% Read PDM file
fid = fopen(pdmFilePath, 'rb');
pdmData = fread(fid, 'uint16');
fclose(fid);
pdmData=pdmData(1:3842616);
% Decode PDM to PCM (8-bit)
pcmData = pdm2pcm(pdmData, bitDepth);

% Create a WAV file
% audiowrite(wavFilePath, pcmData, fs);

disp('Decoding complete and WAV file saved.');