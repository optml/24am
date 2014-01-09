clc
clear all
 s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
s=10;
n=100;
x_s=rand(n,1);
vals=sort(abs(x_s))
tr=vals(n-s)
x_s(abs(x_s)<=tr)=0
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
params(5)=1024;% total starting point
params(6)=64;  % batch-size

noiseLevel=0.001:0.2:1.901
errors=[];
supports=[];
errors1=[];
supports1=[];
exp=1;
for no=noiseLevel
    
    for tr=1:5
        
        p=3;
        B=[];
        for j=1:5   
            % ratio of outliers
            % number of outliers
            
            % noisy base signal
            noise = randn(n,1)*0.001;
            % positions of the outliers
            sel = randperm(n); sel = sel(1:p);
            % add outliers
            noise(sel) = noise(sel) + rand(p,1); 
            noise=noise/norm(noise)*no;
        
           B=[B x_s+noise   ];
        end
         for j=1:3   
            % ratio of outliers
            % number of outliers
           
            % noisy base signal
            noise = randn(n,1)*0.001;
            % positions of the outliers
            sel = randperm(n); sel = sel(1:p);
            % add outliers
            noise(sel) = noise(sel) + rand(p,1); 
        noise=noise/norm(noise)*no;
          % B=[B  noise   ];
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
close all

 h=fig('units','inches','width',7,'height',2,'font','Helvetica','fontsize',16)

plot(noiseLevel, mean(errors'),'rs-')
hold on
plot(noiseLevel, mean(errors1'),'bd-')

xlabel('Noise Level \eta')
ylabel('||x_s - x||_2')
legend('L2 PCA','L1 PCA')

subplot(2,1,2)
plot(noiseLevel, mean(supports'),'rs-')
hold on
plot(noiseLevel, mean(supports1'),'bd-')

xlabel('Noise Level \eta')
ylabel('||x_s .* x||_0')
legend('L2 PCA','L1 PCA')


