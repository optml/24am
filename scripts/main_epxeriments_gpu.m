  clc
clear all
 
 

Data = importdata('gpu_test2.log'); 
% printf("%d,%d,%1.5f,%d,%f,%f,%d,%d,%d,%d,%d\n", optimizationSettings->formulation, GPU,
%			optimizationStatistics->fval, nnz, mt->getElapsedCPUTime(),
%			mt->getElapsedWallClockTime(), optimizationStatistics->it, n, m,
%			optimizationSettings->totalStartingPoints,sizeofvariable);
 
 
 
DOUBLE=(Data(:,11)==8);

ALGS=[2,6];
 
lw=2
SIZES=unique(Data(:,8));
totalStartingPoints=unique(Data(:,10))
 totalStartingPoints=totalStartingPoints(1:2:end)
 totalStartingPoints=totalStartingPoints(1:2:end)
Schema=['r<-';'bh-';'ks-'];
for alg=2:4:2
   
    
   figure(1)
    set(gca,'FontSize',25)
    hold off
     figure(2)
     set(gca,'FontSize',25)
     hold off
    for spI=1:length(totalStartingPoints)
        sp=totalStartingPoints(spI);
   
     
       
         
      
   %   DD=Data(DOUBLE & (Data(:,10)==sp) & (Data(:,1)==alg) ...
   %     & (Data(:,2)==1),[ 8,6]);    
     
   % loglog(DD(:,1),DD(:,2),'<--' );
  % hold on
  %  CC=Data(DOUBLE & (Data(:,10)==sp) & (Data(:,1)==alg) ...
  %      & (Data(:,2)==0),[ 8,6]);    
    
  %    loglog(CC(:,1),CC(:,2),'r<--' );        
    
    
       
     
  %  loglog(CC(:,1),CC(:,2)./DD(:,2),'k<--' );    
     
    
    
     figure(1)
     
    DD=Data(~DOUBLE & (Data(:,10)==sp) & (Data(:,1)==alg) ...
        & (Data(:,2)==1),[ 8,6]);    
    loglog(DD(:,1),DD(:,2),'s-','LineWidth',lw,'MarkerSize',10 );
    hold on
 
    CC=Data(~DOUBLE & (Data(:,10)==sp) & (Data(:,1)==alg) ...
        & (Data(:,2)==0),[ 8,6]);    
    loglog(CC(:,1),CC(:,2),'r<-','LineWidth',lw,'MarkerSize',10  );        
    
      figure(2)
      
     
      
    loglog(CC(:,1),CC(:,2)./DD(:,2),Schema(spI,:),'LineWidth',lw ,'MarkerSize',10 );    
    hold on
 
    
    end
    
    
 %%
    figure(1)
grid on
xlabel('p')
ylabel('Computation Time')
legend( 'GPU','CPU', 'Location','NorthWest')
xlim([min(SIZES) max(SIZES)])
ylim([0.003,20000])
 
%%
     figure(2)
grid on     
ylabel('Speedup')  

 

 legend('GPU1','GPU16','GPU256', 'Location','NorthWest')  



xlabel('p')
xlim([min(SIZES) max(SIZES)])
ylim([0.1,350])

%%

if (alg==2)
  figure(1)
   print('-depsc','-tiff','-r800','l1_pen_l1PCA_gpu.eps')   
  figure(2)
   print('-depsc','-tiff','-r800','l1_pen_l1PCA_gpu_sp.eps')     
   
   
end
if (alg==6)

    
    print('-depsc','-tiff','-r800','l1_con_l1PCA_gpu.eps')   
end

  
end