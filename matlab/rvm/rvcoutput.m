function predictedY = rvcoutput(trnX,trnY,tstX,ker,alpha,bias,p1,actfunc)
%RVCOUTPUT Calculate RVC Output
%
%  Usage: predictedY = rvcoutput(trnX,trnY,tstX,ker,alpha,bias,actfunc)
%
%  Parameters: trnX   - Training inputs
%              trnY   - Training targets
%              tstX   - Test inputs
%              ker    - kernel function
%              beta   - Lagrange Multipliers
%              bias   - bias
%              actfunc- activation function (0(default) hard | 1 soft) 
%
%  Author: Chris Bates based off of Steve Gunn's code

  if (nargin < 7 | nargin > 8) % check correct number of arguments
    help rvcoutput
  else

    if (nargin == 7)
      actfunc = 0;
    end
    n = size(trnX,1);
    m = size(tstX,1);
    H = zeros(m,n);  
    for i=1:m
      for j=1:n
        H(i,j) = trnY(j)*svkernel(ker,tstX(i,:),trnX(j,:),p1);
      end
    end
%     if (actfunc)
%       predictedY = softmargin(H*alpha + bias);
%     else
      predictedY = sign(H*alpha + bias);
   % end
  end
