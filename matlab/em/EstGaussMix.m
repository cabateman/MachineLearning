function [MixProps, Means, CoVars,NumComps] = EstGaussMix(X,EstAlg,K)

%Gaussian Mixture Component Estimation
%
%  Usage: [MixProps, Means, CoVars,NumComps] = EstGaussMix(X,EstAlg,K)
%
%  Parameters: X           - D x N input data where D is the dimensionality and
%                            N is the number of samples.
%
%              EstAlg      - 'E' for EM and 'V' for Variational learning
%              K           - Optional argument indicating the number of
%                            components
%              MixProps    - NumComps estimated mixture proportions
%              Means       - D x NumComps estimated means
%              Covars      - D x D x NumComps estimated Covariance
%                            Matricies
%              NumComps    - 1 x 1 Number of Mixture components as
%                            discussed above.
%
%
%  Author: Chris Bates (cbates@ufl.edu)
%
%  References: C Bishop. Pattern
%  Recognition and Machine Learning. New York: Springer, 2006
%  MixProps  = 0;
%  Means     = 0;
%  CoVars    = 0;
%  NumComps  = 0;

% Check correct number of arguments
if (nargin <2)
    fprintf('You gave %f arguments...please enter X and EstAlg. \n',nargin);

else

    % If K not assigned, assign an intial guess of 10 components
    if (nargin<3)
        K=10;
    end
    % Sample 2D Gaussian Mixture for Testing
%         mean1              = [1.5 15];
%         sigma1             = [1 1.5; 1.5 3];
%         mean2              = [8 12];
%         sigma2             = [1 1; 1 8];
%         Mixture(1:100,:)   = mvnrnd(mean1,sigma1,100);
%         Mixture(101:200,:) = mvnrnd(mean2,sigma2,100);
%         Comp1              = Mixture(:,1);
%         Comp2              = Mixture(:,2);
%         X                  = [Comp1(:) Comp2(:)];
%   
%         100D Gaussian Dataset for Testing
%         mean1              = 1:1:100;
%         sigma1             = eye(100)*rand + rand;
%         mean2              = 1:2:200;
%         sigma2             = eye(100)*rand + rand;
%         Mixture(1:100,:)   = mvnrnd(mean1,sigma1,100);
%         Mixture(101:200,:) = mvnrnd(mean2,sigma2,100);
%         X                  = Mixture;

    % INITIALIZE GLOBAL PARAMETERS
    N                    = size(X,1);
    D                    = size(X,2);
    CHANGETHRESH         = .1;
    MEANPRUNE            = .1;
    MIXPRUNE             = .05;
    MAXITERS             = 50;

    % Use kmeans to intially guess Means
    [IDX,C]              = kmeans(X,K);

    % Assemble cell array of estimated Means
    Means                = cell(zeros(1,size(C,1)));
    for iterC = 1: size(C,1)
        Means{iterC}     = C(iterC,:);
    end

    % Assemble cell array of estimated Covariances
    CoVars               = cell(zeros(1,K));
    for iterK = 1:K
        CoVars{iterK}    = eye(D);
    end

    % Assign datapoints to a specific cluster to calculate MixProps
    % Note--This could probably be optimized
    clear iterK;
    cluster              = zeros(size(IDX,1),K);
    for iterIDX = 1:size(IDX,1)
        for iterK = 1:K
            if IDX(iterIDX) == iterK
                cluster(iterIDX,iterK) = 1;
            else
                cluster(iterIDX,iterK) = 0;
            end
        end
    end

    % Randomly guess that data will be evenly distributed among K clusters.
    % Assemble cell array of cluster data points
    MixProps            = cell(zeros(1,size(C,1)));
    Nk               = cell(zeros(1,K));
    clusterSum       = sum(cluster);
    for iterNk = 1:K
        Nk{iterNk}    = clusterSum(iterNk);
        MixProps{iterNk} = clusterSum(iterNk)/length(cluster);
    end
    
    loglikelihood        = 1E-9;

    switch lower(EstAlg)
        % Execute EM specific algorithm
        case 'e'
            for iterEM = 1:MAXITERS

                %                     %To Print figures for 2D Case, Uncomment--WILL SLOW
                %                     figure(iterEM)
                %                     hold on

                clear iterK
                for iterK = 1:K

                    %     %To Print figures for 2D Case, Uncomment--WILL
                    %     SLOW
                    %     x2D = -2:.1:100;
                    %     y2D = -2:.1:100;
                    %     [PlotX,PlotY] = meshgrid(x2D,y2D);
                    %     clusterdata = [PlotX(:) PlotY(:)];
                    %     p = mvnpdf(clusterdata,Means{iterK},CoVars{iterK});
                    %
                    %     scatter(Mixture(1:100,1),Mixture(1:100,2),15,'r')
                    %     scatter(Mixture(101:200,1),Mixture(101:200,2),15,'r')
                    %     hold on
                    %     meansgraph = Means{iterK};
                    %     plot(meansgraph(1,1),meansgraph(1,2),'ob','LineWidth',3)
                    %     hold on
                    %
                    %     contour(PlotX,PlotY,reshape(p,size(x2D,2),size(y2D,2)),3,'g');
                    %

                    %---E (Expectation) Step -----
                    %9.23
                    normal                      = mvnpdf(X,Means{iterK},CoVars{iterK});
                    numer(:,iterK)              = MixProps{iterK}*normal;
                         end
                    denom                       = sum(numer,2);

                    %This step perhaps should be iterated through another loop as the
                    %responsibilities need to be calculated with a denominator that is
                    %constant through the whole calculation--that is, the sum over all the
                    %components chosen.
                         clear iterK;
                         for iterK = 1:K
                    rsb                         = numer(:,iterK)./denom;
                    Nk{iterK}                   = sum(rsb);

                    %---M (Maximization) Step -----

                    %9.24  Evaluate new means
                    Means_new{iterK}            = (sum(repmat(rsb,1,D).*X)./Nk{iterK});



                    %9.25 Evaluate new sigmas
                    dev                         = X-repmat(Means_new{iterK},N,1);
                    for iterN =1:N
                        rsbx(iterN,:)           = rsb(iterN)*(dev(iterN,:)*dev(iterN,:)');
                    end
                    CoVars_new{iterK}           = (sum(rsbx)/Nk{iterK})*eye(size(X,2));


                    %9.26 Evaluate new mixing coefficients
                    MixProps_new{iterK}         = Nk{iterK}/N;


                    %FOR DIAGONAL CASE--Not sure if this works
                    %     error_diag            = rsb'*(X-repmat(Means_new{iterK},size(X,1),1)).^2;
                    %     CoVars_new            = error_diag' * error_diag;
                end

                % Keep track of components that need to be pruned
                componentTrace = 1;
                clear toprune;

                % PRUNE COMPONENTS VIA MEANS
                % We compare all the means against each other.
                %
                % Keep track of the means to be pruned, but do the pruning
                % at the end of the loop so the remaining calculations can
                % be carried out.
                %
                % Make sure means are not compared with themselves.  When
                % two means are close enough, select one of the two to be
                % pruned.

                for iterMean1 = 1:K
                    for iterMean2 = 1:K
                        clear means_new1;
                        clear means_new2;
                        means_new1 = Means_new{iterMean1};
                        means_new2 = Means_new{iterMean2};
                        if (means_new1(1,1) ~= means_new2(1,1))
                            if(~sum(abs(means_new1(1,1)-means_new2(1,1)) > MEANPRUNE))
                                if(~sum(abs(means_new1(1,2)-means_new2(1,2)) > MEANPRUNE))
                                    % PRINT CONVERGE CONDITIONS SATISFIED
%                                     fprintf('ConvergeMean1 : %3.0f\n',iterMean1);
%                                     fprintf('ConvergeMean2 : %3.0f\n',iterMean2);
%                                     fprintf('\n');
                                    toprune(componentTrace) = iterMean2;
                                    componentTrace = componentTrace+1;

                                end
                            end
                        end
                    end

                end

                %9.28 Evaluate log likelihood
                loglikelihood_new = sum(log(sum(numer,2)));

                %PRINT EM STATS
%                 fprintf('EM Statistics\n');
%                 fprintf('Iteration Number : %3.0f\n',iterEM);
%                 fprintf('LogLikelihood Change : %3.0f\n',abs(loglikelihood_new-loglikelihood));
%                 fprintf('\n');


                % CHECK LOGLIKELIHOOD CONVERGENCE
                if (~sum(abs(loglikelihood_new-loglikelihood) > CHANGETHRESH))
                    break;
                end

                if (exist('toprune') ~= 0)
                    for del = 1:size(toprune)

                        % PRINT COMPONENTS TO BE PRUNED
%                         fprintf('To Prune: %3.0f\n',toprune(del));
%                         fprintf('\n');

                        % Prune Mean and Covariances that meet convergence
                        % criteria
                        Means_new(toprune(del))  = [];
                        CoVars_new(toprune(del)) = [];
                        MixProps_new(toprune(del)) = [];
                       
                    end

                    K                            = K - 1;

                end

                % Set new parameters as current parameters to be used upon
                % next iteration.
                loglikelihood                    = loglikelihood_new;
                Means                            = Means_new;
                MixProps                         = MixProps_new;
                CoVars                           = CoVars_new;
                NumComps                         = K;

            end


        case 'v'
            % INITIALIZE VARIATIONAL-SPECIFIC PARAMETERS
            munot              = ones(1,D);
            alphanot           = 1;
            betanot            = 1;
            vnot               = D;
            Wnot               = .1*eye(D);
            clear iterK
            for iterK = 1:K
            %Generate precisions Ak for each component
                Ak{iterK}     = eye(size(X,2));
            
            %Generate means uk for each component
                uk{iterK}     = Means{iterK};

            %Generate means muk for each component
                muk{iterK}    = Means{iterK};

            %Generate sigmas Sk for each component
                Sk{iterK}     = eye(size(X,2));

            %Generate wisharts Wk for each component
                Wk{iterK}     = eye(size(X,2));

            %Generate means xbark for each component
                xbark{iterK} = Means{iterK};

            %Guess initial mixing alphas
            alphak(iterK)          = alphanot + Nk{iterK};

            %Guess initial betas
            beta(iterK)           = betanot + Nk{iterK};

            %Guess initial v
            v(iterK)              = vnot + Nk{iterK};
            end

            for iterV = 1:MAXITERS

                %                     %To Print figures for 2D Case, Uncomment--WILL SLOW
                %                     figure(iterV)
                %                     hold on

                clear iterK
                clear numer
                for iterK = 1:K

                    %     %To Print figures for 2D Case, Uncomment--WILL
                    %     SLOW
                    %     x2D = -2:.1:100;
                    %     y2D = -2:.1:100;
                    %     [PlotX,PlotY] = meshgrid(x2D,y2D);
                    %     clusterdata = [PlotX(:) PlotY(:)];
                    %     p = mvnpdf(clusterdata,Means{iterK},CoVars{iterK});
                    %
                    %     scatter(Mixture(1:100,1),Mixture(1:100,2),15,'r')
                    %     scatter(Mixture(101:200,1),Mixture(101:200,2),15,'r')
                    %     hold on
                    %     meansgraph = Means{iterK};
                    %     plot(meansgraph(1,1),meansgraph(1,2),'ob','LineWidth',3)
                    %     hold on
                    %
                    %     contour(PlotX,PlotY,reshape(p,size(x2D,2),size(y2D,2)),3,'g');
                    %

                    %---E (Expectation) Step -----

                    % CALCULATE RESPONSIBILTY (10.49)

                    % First calculate ln(pnk)

                    %10.64--Expect[(xn-uk)'Ak(xn-uk)]
                    clear iterN;
                    clear devmuk;
                    clear cross;
                    for iterN =1:N
                        devmuk                 = X(iterN,:)-muk{iterK};
                        cross(iterN,:)         = (devmuk*Wk{iterK}*devmuk');
                    end
                    clear e_ukAk;
                    e_ukAk             = repmat(D/beta(iterK),N,1)+v(iterK)*cross;

                    %10.65--Expect[lnAk]
                    psisum             = zeros(D,1);
                    clear dim;
                    for dim = 1:D
                        psisum(dim)      = psi((v(iterK)+1-dim)/2);
                    end
                    clear e_lnAk;
                    %Log Determinant becomes unstable at high dimensions
                    %Implement Cholesky factorization
                    [L,p] = chol(Wk{iterK},'lower');
                    option1 = log(det(Wk{iterK}));
                    option2 = trace(log(L));
                    e_lnAk             = sum(psisum)+D*log(2)+option2;

                    %10.66--Expect[lnpik]
                    clear e_lnpik;
                    e_lnpik            = psi(alphak(iterK))-psi(sum(alphak));


                    %10.46
                    clear lnpnk;
                    lnpnk              = e_lnpik + .5.*e_lnAk- D/2.*log(2*pi)-.5*e_ukAk;


                    numer(:,iterK)     = exp(lnpnk);
                end
                clear denom;
                denom                  = sum(numer,2);

                % Keep track of components that need to be pruned
                componentTrace = 1;
                clear toprune;
                
                clear iterK;
                for iterK = 1:K
                    clear rsb;
                    rsb                = numer(:,iterK)./denom;

                    Nk{iterK}         = sum(rsb);
                    
                    MixProps_new{iterK}        = Nk{iterK}/N;
                    
                    alphak(iterK)     = alphanot + Nk{iterK};


                    %---M (Maximization) Step -----

                    %10.52  Evaluate new means xbar
                    xbark{iterK}      = (sum(repmat(rsb,1,D).*X)./Nk{iterK});

                    %10.53 Evaluate new precision Sk
                    clear iterN;
                    clear dev;
                    clear rsbx;
                    for iterN =1:N
                        dev            = X(iterN,:)-xbark{iterK};
                        rsbx(iterN,:)  = rsb(iterN)*(dev*dev');
                    end
                    Sk{iterK}         = (sum(rsbx)/Nk{iterK})*eye(D);
                    
                    % PRUNE COMPONENTS VIA MIXPROPS

                    % For variational, it seems the posterior has trouble
                    % converging more than overlapping means.  Therefore,
                    % the mixture proportions will be monitored and when it
                    % goes below a threshold (ie 5%), that component will no longer contribute so it should be pruned.
                    %
                    % Keep track of MixProps to be pruned, but do the pruning
                    % at the end of the loop so the remaining calculations can
                    % be carried out.

                    
                    clear mp;
                    clear sk;
                    mp = MixProps_new{iterK};
                    sk = Sk{iterK};
                    if (mp <= MIXPRUNE)
                        toprune(componentTrace) = iterK;
                        componentTrace = componentTrace+1;
                    else

                        if (cond(sk) > 1E9)
                            Sk{iterK} = eye(D);
                            toprune(componentTrace) = iterK;
                            componentTrace= componentTrace+1;
                        end
                    end

                    
                    %Evaluate terms for the Gaussian-Wishart Posterior 10.59
                    %or to carry out updates upon iterations.

                    %10.60
                    betak(iterK)      = betanot + Nk{iterK};
                    %10.61
                    muk{iterK}        = (1/betak(iterK)).*(betanot.*munot+Nk{iterK}.*xbark{iterK});
                    %10.62
                    clear devmu;
                    devmu              = xbark{iterK}-munot;
                    invWk{iterK}      = inv(Wnot)+Nk{iterK}*Sk{iterK}+((betanot*Nk{iterK})/(betanot+Nk{iterK})).*devmu*devmu';
                    %10.63
                    vk(iterK)         = vnot + Nk{iterK};

                    % Now we want to evaluate the performance of the VI
                    % technique by creating a posterior distribution given
                    % by the parameters we calculated.
                    
                    %**Note** This is the first point of failure for VI if
                    %Sk becomes singular, which is often caused by no data
                    %points in the mixture, or zero responsibilities being
                    %given to the data points.
                    
                    p   = mvnpdf(X,muk{iterK},Sk{iterK});

                    posterior(:,iterK)         = MixProps_new{iterK}.*p;

                end

                %9.28 Evaluate max posterior (which I'll call
                %loglikelihood_new just to keep things consistent between
                %EM and VI.
                loglikelihood_new = sum(log(sum(posterior,2)));

                %PRINT VI STATS
%                 fprintf('VI Statistics\n');
%                 fprintf('Iteration Number : %3.0f\n',iterV);
%                 fprintf('LogLikelihood Change : %3.0f\n',abs(loglikelihood_new-loglikelihood));
%                 fprintf('\n');

                % CHECK CONVERGENCE
                if (~sum(abs(loglikelihood_new-loglikelihood) > CHANGETHRESH))
                    break;
                end

                if (exist('toprune') ~= 0)
                    for del = 1:size(toprune)

                        % PRINT COMPONENTS TO BE PRUNED
%                         fprintf('To Prune: %3.0f\n',toprune(del));
%                         fprintf('\n');

                        % Prune Mean and Covariances that meet convergence
                        % criteria
                        muk(toprune(del))  = [];
                        Sk(toprune(del)) = [];
                        Nk(toprune(del)) = [];
                        MixProps_new(toprune(del)) = [];
                    end

                    K                            = K - 1;

                end
                loglikelihood = loglikelihood_new;
                MixProps      = MixProps_new;
                Means         = muk;
                CoVars        = Sk;
                Means_new     = muk;
                CoVars_new    = Sk;
            end




        otherwise
            fprintf('Please enter E (Expectation Maximization) or V (Variational Inference) for EstAlg\n');
    end
                Means                            = Means_new;
                MixProps                         = MixProps_new;
                CoVars                           = CoVars_new;
                NumComps                         = K;
end
