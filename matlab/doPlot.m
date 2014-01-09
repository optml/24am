 
close all

 h=fig('units','inches','width',8,'height',8,'font','Helvetica','fontsize',20)

 lw=5
plot(noiseLevel, mean(errors'),'rs-','LineWidth',lw)
hold on
plot(noiseLevel, mean(errors1'),'bd-','LineWidth',lw)

xlabel('Noise Level \eta')
ylabel('||x_{Signal} - x_{PCA}||_2')
legend('L2 PCA','L1 PCA','Location','SouthEast')
xlim([0,max(noiseLevel)])
h=fig('units','inches','width',8,'height',8,'font','Helvetica','fontsize',20)
plot(noiseLevel, mean(supports'),'rs-','LineWidth',lw)
hold on
plot(noiseLevel, mean(supports1'),'bd-','LineWidth',lw)

xlabel('Noise Level \eta')
ylabel('||x_{Signal} .* x_{PCA}||_0')
legend('L2 PCA','L1 PCA','Location','NorthEast')
xlim([0,max(noiseLevel)])

