
function smoothedVector=ksmooth(vector,windowWidth) 
% smooth a vector using gaussian kernel with a width equal to windowWidth
% useful for plotting gene expressions over pseudotime

gaussFilter = pre_post_process.gausswin(windowWidth);
gaussFilter = gaussFilter / sum(gaussFilter); % Normalize.

% Do the blur.
smoothedVector = conv(vector, gaussFilter);
smoothedVector=smoothedVector(windowWidth:end-windowWidth);
end
