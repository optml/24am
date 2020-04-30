clc
  


s = RandStream('mt19937ar','Seed',1);
RandStream.setGlobalStream(s);
 
 imshow(reshape(avg,m,n));
 

%%
%figure(i)
close all
clf
PC=10
for i=1:PC
    subplot(5,PC/5,i)
I = reshape(L1(:,i),m,n);
spy(I)
end



%%

%avg=mean(B);
%B=B-ones(r,1)*avg;

%%
%[U,S,V]=svd(B);
%%
close all
PC=40
sample=13
clf
img=A(sample, :);
I = reshape(img,m,n);
%h=fig('units','inches','width',1,'height',1,'font','Helvetica','fontsize',16)
imwrite(I,sprintf('%d_orig.png',sample))
 
imshow(I)
 
img=B(sample,:)+avg
I = reshape(img,m,n);
imshow(I) 
 imwrite(I,sprintf('%d_occ.png',sample))
ff=2
for k=0:1
    
 if k==0
     MET=L1;
 else
     MET=L2;
 end
    ff=ff+1

ff=ff+1
img=B(sample,:)
imNew=zeros(size(img));
for i=1:PC
    imNew = imNew + ( MET(:,i)*(MET(:,i)'*img'))';
   
end
imNew=imNew+avg;
I = reshape(imNew,m,n);
figure
imshow(I)
imwrite(I,sprintf('%d_rec_%d_l%d.png',sample, PC,1+k))
 

end

%%
TPC=40
l1s=[]
l2s=[]
[rr cc]=size(A);
MET=L1;
for PC=1:TPC
    PC
  val=0;
  for sample=1:rr
    oim=A(sample,:);
    recon=A(sample,:);
    val=val+norm(oim'-MET(:,1:PC)*(MET(:,1:PC)'*recon')+avg')^2;
  end
  val=val/r;  
  l1s(PC)=val;
end
MET=L2;
for PC=1:TPC
    PC
  val=0;
  for sample=1:rr
    oim=A(sample,:);
    recon=A(sample,:);
    val=val+norm(oim'-MET(:,1:PC)*(MET(:,1:PC)'*recon')+avg')^2;
  end
  val=val/r;  
  l2s(PC)=val;
end
%%
close all
h=fig('units','inches','width',8,'height',8,'font','Helvetica','fontsize',28)
lw=3
ms=10
hold off
plot(l1s,'bs-','LineWidth',lw,'MarkerSize',ms)
hold on
plot(l2s,'rx--','LineWidth',lw,'MarkerSize',ms)
legend('L1-PCA','L2-PCA','Location','NorthEast')
xlabel('m')
ylabel('E(m)')
grid on
ylim([5800 10000])

