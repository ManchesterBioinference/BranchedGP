% n-point gaussian window of with w
% a: window narrowness, n: curve smoothness

function x = gausswin(n, w)
  if nargin == 1, w = 2.5; end
  x = exp( -0.5 * (w/n * (-(n-1) : 2 : n-1)') .^ 2 );
end
