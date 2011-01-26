function [] = FitTrendingModel(testdata,traindata,normalize,order,Lambda)

% FitTrendingModel.m
% Author:  Chris Bates
% Date: 8/25/10
% Description:  Function takes a training set, a testing set, a normalization set, a polynomial order specification, and a lambda (for regularization).  This algorithm solves the least squares normal equations case and plots the polynomial that best fits the data.  Test data can be supplied to predict according to model


% INITALIZE
x         = testdata(:,1);
y         = testdata(:,2)./normalize;
xn        = traindata(:,1);                       %inputs of training set
tn        = traindata(:,2)                       %desired outpts training set
NumPts    = size(traindata,1);                    %number of data points
NumPts_test = size(testdata,1);
exparr    = repmat(0:order, [NumPts 1]);           %expon pre-compute pow of xn
exparr_test = repmat(0:order,[NumPts_test 1]);
Aall      = repmat(xn, [1 order+1]).^exparr;     %pre-compute powers of xn
Aall_test = repmat(x,[1 order+1]).^exparr_test;
exparr    = repmat(0:order, size(x));             %expon pre-compute powe of x
powerall  = repmat(x, [1 order+1]).^exparr;       %pre-computed powers of x

M=order;
    
    %PLOT TRAINING DATA
    figure(M);
     scatter(x,y,'o','linewidth',3);

    
    %SOLVE NORMAL EQUATIONS FOR THE NON-REGUALRIZED CASE
    A = Aall(:,1:M+1);
    A_test = Aall_test(:,1:M+1);
    ata = A'*A
    w = pinv(A'*A)*A'*tn;
    
    %BUILD THE NON-REGULARIZED APPROXIMATION POLYNOMIAL
    %COULD PROCESS BE MORE EFFICIENT
    powers = powerall(:, 1:M+1);
    approx = powers*w;
    plot(x,approx,'k','linewidth', 3);
    %PLOT THE NON-REGULARIZED POLYNOMIAL APPROXIMATION IN RED
    figure(M);
    hold on
%      plot(x,approx,'r','linewidth',3)
    
    %SOLVE DIAGONALLY LOADED NORMAL EQUATIONS FOR THE REGULARIZED CASE
    nsize = size(A,2);
    nsize_test = size(A_test,2);
    AL = A'*A+Lambda*eye(size(nsize));
    AL_test = A_test'*A_test+Lambda*eye(size(nsize_test));
    w = pinv(AL)*A'*tn;
    w_test = pinv(AL_test)*A_test'*y;
    
    %BUILD THE REGULARIZED APPROXIMATION POLYNOMIAL
    approx = powers*w;
    approx_test = powers*w_test;
    
    %PLOT REGULARIZED POLYNOMIAL APPROXIMATION TO THE SINE WAVE IN BLACK
    figure(M);
    hold on
      plot(x,approx_test,'r','linewidth',3);
    

