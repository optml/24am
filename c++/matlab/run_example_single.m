clc
clear all
 
compile

m=15;
n=30;
B=rand(m,n)
B_single=randn(m,n, 'single');  
B_single=B_single*0+B;

 
params=[]
params(1)=0; % algorithm
%   L0_constrained_L2_PCA = 0,
%	L0_constrained_L1_PCA = 1,
%	L1_constrained_L2_PCA = 2,
%	L1_constrained_L1_PCA = 3,
%		 = 4,
%	L0_penalized_L1_PCA = 5,
%	L1_penalized_L2_PCA = 6,
%	L1_penalized_L1_PCA = 7

    
params(2)=n; % penalty/constraint
params(3)=0.0; % toll for algorithm to stop
params(4)=60; % total iterations
params(5)=1024;% total starting point
params(6)=64;  % batch-size

[U Sigma V] = svd(B);

 
[x] = dense_multicore_24am_wrapper(B,params);
   
disp(sprintf('error for  PV with double is %e', norm(abs(x)-abs(V(:,1)))))
 
 
[y] = dense_multicore_24am_wrapper(B_single,params);
   
disp(sprintf('error for  PV with float is %e', norm(abs(y)-abs(V(:,1)))))
 