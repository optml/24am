clc
clear all
 s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
s=10;
n=100;
 
xx=rand(n,1);
sel = randperm(n); 
sel = sel(1:s);
            % add outliers
x_s= zeros(n,1);
x_s(sel)=rand(s,1)
 
x_s=x_s/norm(x_s)
%%
 
 
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

    
params(2)=s; % penalty/constraint
params(3)=0.0; % toll for algorithm to stop
params(4)=60; % total iterations
params(5)=64;% total starting point
params(6)=64;  % batch-size

noiseLevel=0.001:0.025:0.301
errors=[];
supports=[];
errors1=[];
supports1=[];
exp=1;
for no=noiseLevel
    
    for tr=1:20
        
        
        B=[];
        for j=1:10
            % ratio of outliers
            % number of outliers
            p=5
            % noisy base signal
            noise = x_s+randn(n,1)*no;
            sel = randperm(n); sel = sel(1:p);
            noise(sel)=0;
            sel = randperm(n); sel = sel(1:p);
            noise(sel)=ones(p,1); 
        
           B=[B noise   ];
        end
         
        
         
        
          
        
        
        m=mean(B);
        for j=1:length(m)
            B(:,j)=B(:,j)-m(j)*ones(n,1);
        end 
        
        
        params(1)=0;
        [xS] = dense_multicore_24am_wrapper(B',params);
        er=norm(xS-(x_s));
        err=norm(xS+(x_s));
        er=min(er,err);
        sp=sum(xS.*x_s~=0);
        errors(exp,tr)=er;
        supports(exp,tr)=sp;

        params(1)=1;
        [xS1] = dense_multicore_24am_wrapper(B',params);

         er=norm(xS1-(x_s));
        err=norm(xS1+(x_s));
        er=min(er,err)
        sp=sum(xS1.*x_s~=0);

        errors1(exp,tr)=er;
        supports1(exp,tr)=sp;
        end
    exp=exp+1;
end
%% 


doPlot