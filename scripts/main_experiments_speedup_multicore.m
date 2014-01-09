clc
clear all

 
 
Data = importdata('../results/paper_experiment_multicore_speedup.txt'); 
Data=Data.data;

sizes = unique(Data(:,3))
cores = unique(Data(:,8))

FINAL_DATA=[];
for i=[1 2 4 8]

   for ci=1:length(cores)
       for siz=1:length(sizes)
       idx=(Data(:,3)==sizes(siz))&(Data(:,8)==cores(ci));
       FINAL_DATA(ci,siz)=Data(idx,4);
    
       end 
   end 
end
 


%%
TP=['x-';'<-';'h-']
CP=['r','b','g','k']
Schema=['r<-';'bh-';'ks-';'ro-';'bs-'];
figure(1)
hold off
%PLOT 
  set(gca,'FontSize',25)
  Y=[];
  for core=1:length(cores)
     y=FINAL_DATA(core,:);
     TM=y(:);
     Y=[Y;TM];
     loglog(sizes(1:end),TM(1:end), Schema(core,:),'MarkerSize',10,'LineWidth',2);
      hold on
  end 
  legend('1 CORE','2 CORES','4 CORES','8 CORES','16 CORES','Location','NorthWest')
  ylabel('\fontsize{25}Computation Time') 
  grid on
  xlim([min(sizes),max(sizes(1:end))])
  ylim([min(Y),max(Y)])
  xlabel('\fontsize{25}p')
  ax1 = gca;
  grid on
print('-depsc','-tiff','-r800', 'multicore_times.eps')   
 
 

%%
figure(2)
%PLOT 
  set(gca,'FontSize',25)
  Y=[];
  FINAL_DATA=FINAL_DATA*diag(FINAL_DATA(1,:).^(-1));
  FINAL_DATA=FINAL_DATA.^(-1);
  hold off
  for core=1:length(cores)
     y=FINAL_DATA(core,:);
     TM=y(:);
     Y=[Y;TM];
     loglog(sizes(1:end),TM(1:end), Schema(core,:),'MarkerSize',10,'LineWidth',2);
      hold on
  end 
 set(gca,'FontSize',25)
 legend('\fontsize{25}1 CORE',...
       '\fontsize{25}2 CORES','\fontsize{25}4 CORES','\fontsize{25}8 CORES','16 CORES','Orientation','vertical',...
      'Location','NorthWest')
       ylabel('\fontsize{25}Speedup') 
  xlim([min(p)-0.1,max(p(1:end-1))])
  ylim([min(Y),max(Y)])  
  xlabel('\fontsize{25}p')
  grid on
print('-depsc','-tiff','-r800','multicore_speedup.eps')  


