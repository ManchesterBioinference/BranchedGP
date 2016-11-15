function spdists = parfor_spdists_knngraph( x, k, varargin )
% spdists = spdists_knngraph( x, k, distance, chunk_size, verbose )
%
% finds the nearest neighbor graph for data matrix x. the graph is represented as a sparse matrix. for each point in x,
% the k nearest neighbors are found. distance is the distance metric used by knnsearch (such as euclidean, cosine,
% etc.).
%
% if chunk_size is specified, x will be split to chunk_size-by-D submatrices, and the calculation will be run
% separately on each submatrix (in order to conserve memory).

    distance = 'euclidean';
    chunk_size = size( x, 1 );
    verbose = 1;
        
	for i=1:length(varargin)-1
        if(strcmpi(varargin{i},'distance'))
            distance = varargin{i+1};
        elseif(strcmpi(varargin{i},'chunk_size'))
            chunk_size = varargin{i+1};
        elseif(strcmpi(varargin{i},'verbose'))
            verbose = varargin{i+1};
        end
    end

	n = size( x, 1 );

	spdists = sparse([],[],[], n, n, n*k);

	total_chunks = ceil( n / chunk_size );
    all_chunks   = cell(total_chunks,2);
    
    tic
    % iterate over submatrices
    %parfor iter = 1:total_chunks
     for iter = 1:total_chunks

        from = 1+chunk_size*(iter-1);
		to = min( from + chunk_size - 1, n );
		rx = from:to;
		
		[ idx, d ] = knnsearch( x, x( rx, : ), 'k', k + 1, 'distance', distance );

		idx( :, 1 ) = []; d( :, 1 ) = []; % remove self neighbor

		% update spdists
		js = repmat( rx', 1, k );
		indices = sub2ind( size( spdists ), idx(:), js(:) );
        d(d==0) = eps;
        all_chunks (iter, :) = {indices(:), d(:)};

		if( verbose )
			% report progress if required by user
			if( mod( iter, 10 ) == 0 )
				fprintf( 1, '.' );
			elseif( mod( iter, 100 ) == 0 )
				fprintf( 1, '%3.2f%%', iter / total_chunks * 100 );
			end
		end
    end
    for iter = 1:length(all_chunks)
        spdists(all_chunks{iter, 1}) = all_chunks{iter, 2}; %ignore warning, the matrix was pre allocated for efficientcy 
    end
 
    sprintf('distributed sparse knn graph computed: %g', toc)
end
