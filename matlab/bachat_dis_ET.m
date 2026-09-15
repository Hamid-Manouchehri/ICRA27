function Z = bachat_dis_ET(X,Y)

m1 = mean(X); 
m2 = mean(Y);

% Define covariance matrices
cov1 = var(X); 
cov2 = var(Y);

% Calculate differences and averages
meanDiff = m1 - m2;
covAvg = (cov1 + cov2) / 2;

% Compute Bhattacharyya Distance
Z = 0.125 * (meanDiff' / covAvg * meanDiff) + ...
    0.5 * log(det(covAvg) / sqrt(det(cov1) * det(cov2)));
