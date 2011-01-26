function [nrv, weights] = rvmreg(X,Y,ker,p1)
%RVM Regression
%Chris Bates
% This function calculates weights and number of relevance vectors for a
% given training set and training targets.  Instead of calling Gunn's
% svkernel file, a neural network distance function was called
% Below is commented code for the cross data set
% clear 
% crossdata = load('C:\crossnn.txt');
% X = crossdata(:,1:2);
% Y = crossdata(:,3)-2;
% Y(101:end,1) = 1;

% X = [trainC1; trainC2];
% Y = [tgtC1; tgtC2];
% p1 = .18;

% INITALIZE
NUMPTS        = size(X,1);
t             = Y;
M             = 20;
%mu            = X(1:10:size(X,1), :);      %Gaussian function means
mu            = X;                          %Every data point will be a center
s             = p1;                         %RBF spread
b_old          = .5;                        %set beta
a_old         = 2.0*ones(size(mu,1),1);     %set alpha  
A             = diag(a_old);
CHANGETHRESH  = .01;
MAXITERS      = 100;
ker = 'rbf';



% CALCULATE PHI
PairWiseDists = dist(X,mu');
PairWiseSqDists = PairWiseDists.*PairWiseDists;
PHI = exp(-PairWiseSqDists./(2*s.^2));

%First iteration of mean and sigma....eq. 7.83
sigma = pinv(A + b_old*PHI'*PHI);
mean  = b_old*sigma*PHI'*t;


%-----Log of Marginal Liklihood (eq 7.85)---------------------%
% C = pinv(Beta)*eye(NUMPTS)+PHI*pinv(A)*PHI';
% RVLogLiklihood = -.5*NUMPTS*log(2*pi)+log(det(C))+ t'*pinv(C)*t;

%----Find Alpha-----------------------------------------------%
delta_alpha = inf;


for i=1:MAXITERS
    
%while ((delta_alpha > CHANGETHRESH) && (iter < MAXITERS))
    gamma             = 1-(a_old.*diag(sigma));
    %are these alpha's supposed to be replaced at every iteration or just
    %added to the matrix.
    a_new             = gamma./(mean.^2);

    % CHECK CONVERGENCE
    for iter = 1:size(mu,1)
        if a_new(iter) < 1E9
            a_old(iter) = a_new(iter);
        else
            a_old(iter) = a_old(iter);
        end
    end



    termb             = PHI*mean;
    for k=1:NUMPTS
        b_numerator(k)=t(k)-termb(k);
    end
    b_numerator       = norm(b_numerator);
    b_numeratorsq     = b_numerator.*b_numerator;

    %this could potentially be a problem
    b_new             = pinv(b_numeratorsq ./ (NUMPTS - sum(gamma)));

    A                 = diag(a_old);
    sigma             = pinv(A + b_new*PHI'*PHI);
    mean              = b_new*sigma*PHI'*t;

    
end

 A_new                = diag(a_new);
 in_sigma = A_new + b_new*PHI'*PHI;
 sigma_new            = pinv(in_sigma);
 mean_new             = b_new*sigma_new*PHI'*t;
 nrv = NUMPTS - sum(a_new > 1E9);
 weights = mean_new;
 
 
 
  


