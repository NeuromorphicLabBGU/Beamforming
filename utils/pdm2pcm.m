function pcmData = pdm2pcm(pdmData, bitDepth)
    % Reshape PDM data to 8 bits per sample
    pdmData = reshape(pdmData, 16, []).';

    % Convert PDM to PCM
    pcmData = sum(double(pdmData) .* 2.^(7:-1:0), 2);

    % Normalize to 8-bit range
    pcmData = pcmData / max(pcmData) * (2^(bitDepth - 1) - 1);

    % Convert to 8-bit integer
    pcmData = uint8(pcmData);
end