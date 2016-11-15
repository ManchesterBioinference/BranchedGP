function spdists_und = spdists_undirected( spdists )
% spdists_und = spdists_undirected( spdists )
%
% make spdists undirected (by duplicating directional edges).

[i, j, vals] = find( spdists );

% create new sparse matrix, copying transposed spdists on top
spdists_und = accumarray( [i j; j i], [vals; vals], [], @max, [], true );

% in case new matrix is smaller than original, set the bottom right cell to reset size
if( numel( spdists_und ) < numel( spdists ) )
	[m, n] = size( spdists );
	spdists_und( m, n ) = 0;
end


