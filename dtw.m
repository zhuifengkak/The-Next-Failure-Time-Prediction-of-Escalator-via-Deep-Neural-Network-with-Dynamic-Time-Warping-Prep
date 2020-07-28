function [Dist,D,k,w,rw,tw]=dtw(r,t,pflag)
%
% [Dist,D,k,w,rw,tw]=dtw(r,t,pflag)
%
% Dynamic Time Warping Algorithm
% Dist is unnormalized distance between t and r
% D is the accumulated distance matrix
% k is the normalizing factor
% w is the optimal path
% t is the vector you are testing against
% r is the vector you are testing
% rw is the warped r vector
% tw is the warped t vector
% pflag  plot flag: 1 (yes), 0(no)
%
% Version comments:
% rw, tw and pflag added by Pau Mic
 
[r,ps1]=mapminmax(r,0,1);
[t,ps2]=mapminmax(t,0,1);

[row,M]=size(r); if (row > M) M=row; r=r'; end;
[row,N]=size(t); if (row > N) N=row; t=t'; end;
d=sqrt((repmat(r',1,N)-repmat(t,M,1)).^2); %this makes clear the above instruction Thanks Pau Mic
d;
D=zeros(size(d));
D(1,1)=d(1,1);
 
for m=2:M
    D(m,1)=d(m,1)+D(m-1,1);
end
for n=2:N
    D(1,n)=d(1,n)+D(1,n-1);
end
for m=2:M
    for n=2:N
        D(m,n)=d(m,n)+min(D(m-1,n),min(D(m-1,n-1),D(m,n-1))); % this double MIn construction improves in 10-fold the Speed-up. Thanks Sven Mensing
    end
end
 
Dist=D(M,N);
disp(Dist);
n=N;
m=M;
k=1;
w=[M N];
while ((n+m)~=2)
    if (n-1)==0
        m=m-1;
    elseif (m-1)==0
        n=n-1;
    else 
      [values,number]=min([D(m-1,n),D(m,n-1),D(m-1,n-1)]);
      switch number
      case 1
        m=m-1;
      case 2
        n=n-1;
      case 3
        m=m-1;
        n=n-1;
      end
  end
    k=k+1;
    w=[m n; w]; % this replace the above sentence. Thanks Pau Mic
end
 
% warped waves
rw=r(w(:,1));
tw=t(w(:,2));
 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if pflag
    
    % --- Accumulated distance matrix and optimal path
    figure('Name','DTW - Accumulated distance matrix and optimal path', 'NumberTitle','off');
     
    main1=subplot('position',[0.19 0.19 0.67 0.79]);
    image(D);
    cmap = contrast(D);
    colormap(cmap); % 'copper' 'bone', 'gray' imagesc(D);
    hold on;
    x=w(:,1); y=w(:,2);
    ind=find(x==1); x(ind)=1+0.2;
    ind=find(x==M); x(ind)=M-0.2;
    ind=find(y==1); y(ind)=1+0.2;
    ind=find(y==N); y(ind)=N-0.2;
    plot(y,x,'-w', 'LineWidth',2);
    hold off;
    axis([1 N 1 M]);
    set(main1, 'FontSize',12, 'XTickLabel','', 'YTickLabel','');

 
    colorb1=subplot('position',[0.88 0.19 0.05 0.79]);
    nticks=8;
    ticks=floor(1:(size(cmap,1)-1)/(nticks-1):size(cmap,1));
    mx=max(max(D));
    mn=min(min(D));
    ticklabels=floor(mn:(mx-mn)/(nticks-1):mx);
    colorbar(colorb1);
    set(colorb1, 'FontSize',12, 'YTick',ticks, 'YTickLabel',ticklabels);
    set(get(colorb1,'YLabel'), 'String','Distance', 'Rotation',-90, 'FontSize',12, 'VerticalAlignment','bottom');
   
    left1=subplot('position',[0.07 0.19 0.10 0.79]);
    plot(r,M:-1:1,'-b');
%     set(left1, 'YTick',mod(M,10):10:M, 'YTickLabel',10*rem(M,10):-10:0)
    set(left1, 'YTick',1:M, 'YTickLabel',M:-1:0)
    axis([min(r) 1.1*max(r)  1 M ]);% axis([min(r) 1.1*max(r) 1 M]);
    set(left1, 'FontSize',7);
    set(get(left1,'YLabel'), 'String','Failure time series of Escalator #6', 'FontSize',12, 'Rotation',-90, 'VerticalAlignment','cap');
    set(get(left1,'XLabel'), 'String','DoF', 'FontSize',12, 'VerticalAlignment','cap');
    
    bottom1=subplot('position',[0.19 0.07 0.67 0.10]);
    plot(t,'-r');
    axis([1 N min(t) 1.1*max(t)]);
    set(bottom1, 'FontSize',7, 'YAxisLocation','right');
    set(get(bottom1,'XLabel'), 'String','Failure time series of Escalator #3', 'FontSize',12, 'VerticalAlignment','middle');
    set(get(bottom1,'YLabel'), 'String','DoF', 'Rotation',-90, 'FontSize',12, 'VerticalAlignment','bottom');
  
    % --- Warped signals
    %% recover normalization 
    r=mapminmax('reverse',r,ps1);
    t=mapminmax('reverse',t,ps2);
    rw=mapminmax('reverse',rw,ps1);
    tw=mapminmax('reverse',tw,ps2);
    figure('Name','DTW - warped time series', 'NumberTitle','off');
    
    subplot(1,2,1);
    set(gca, 'FontSize',7);
    hold on;   
    plot(r,'-bx');
    plot(t,':r.');
    hold off;
    axis([1 max(M,N) min(min(r),min(t)) 1.1*max(max(r),max(t))]);
    grid;
   legend('Original curve of #6','Original curve of #3');
    title('Original Time Series','fontsize',12);
     xlabel('Failure times ','fontsize',12)
  ylabel('Day of Failure','fontsize',12,'Rotation',-90)
    
    subplot(1,2,2);
    set(gca, 'FontSize',7);
    hold on;
    plot(rw,'-bx');
    plot(tw,':r.');
    hold off;
    axis([1 k min(min([rw; tw])) 1.1*max(max([rw; tw]))]);
    grid;
    legend('Warped curve of #6','Warped curve of #3');
    title('Warped Time Series','fontsize',12);
     xlabel('Failure times','fontsize',12)
     ylabel('Day of Failure','fontsize',12,'Rotation',-90)
    
end
