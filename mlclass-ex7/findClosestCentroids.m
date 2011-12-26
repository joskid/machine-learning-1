function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
m = size(X,1);d = size(X,2);
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%



for i =1:m
	mu = centroids(1,:);
	point = X(i,:);
	summ = sum((point-mu).^2);
	idx(i)=1;
	for j=2:K
		mu = centroids(j,:);
		temp = sum((point-mu).^2);
		if summ > temp
			idx(i)=j;summ = temp;
		end
	end
end




% =============================================================

end

