 clc
 clear all
 


s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
 
load('orl_faces2.mat')


s=112*92/10
PC=10

params=[]
params(1)=0; % algorithm
    
params(2)=s; % penalty/constraint
params(3)=0.0; % toll for algorithm to stop
params(4)=20; % total iterations
params(5)=64;% total starting point
params(6)=32;  % batch-size


%
[r,c]=size(B);
%avg=mean(B')*0;
%for i=1:r
  % B(i,:)=B(i,:)-avg(i); 
%end

avg=mean(B)*1;
B=B-ones(r,1)*avg;



BB=B;
%%
L2=zeros(length(avg),PC);
for i=1:PC
    tic
[x] = dense_multicore_24am_wrapper(BB,params);
x=x/norm(x);
toc
BB=BB-(BB*x)*x';
L2(:,i)=x;

%figure(i)
%I = reshape(x,m,n)
%spy(I)

end
 
BB=B;

L1=[];
params(1)=1;
for i=1:PC
    tic
[x] = dense_multicore_24am_wrapper(BB,params);
    toc
x=x/norm(x);
BB=BB-(BB*x)*x';


L1=[L1 x];
 

end


%%
%figure(i)
%close all
 

