% read in large block data
datlb=readtable('lb_data_la.csv','ReadVariableNames',true);
fipslb=table2array(datlb(:,1));
datlb=table2array(datlb(:,2:3));

% center and scale columns of datlb
datlb=datlb-repmat(mean(datlb),size(datlb,1),1);
datlb=datlb./repmat(std(datlb),size(datlb,1),1);

% read in small block data
datsb=readtable('sb_data_la.csv','ReadVariableNames',true);
fipssb=floor(table2array(datsb(:,1))/1000000);
fipssbfull=table2array(datsb(:,1));
popsb=table2array(datsb(:,2));
datsb=table2array(datsb(:,3:size(datsb,2)));

% center and scale columns of datsb
datsb=datsb-repmat(mean(datsb),size(datsb,1),1);
datsb=datsb./repmat(std(datsb),size(datsb,1),1);

% order small block data to correspond to ordering of large block data and
datsborder=[];
fipssborder=[];
fipssbfullorder=[];
for i=1:size(datlb,1)
    locs=find(fipssb(:,1)==fipslb(i,1));
    yi=datsb(locs,:);
    datsborder=[datsborder; yi];
    fipssborder=[fipssborder; fipssb(locs,1)];
    fipssbfullorder=[fipssbfullorder; fipssbfull(locs,1)];
end
datsb=datsborder;
fipssb=fipssborder;
fipssbfull=fipssbfullorder;

% make weights by dividing population in each small block by the population
% in its corresponding large block
weights=[];
for i=1:size(datlb,1);
    locs=find(fipssb(:,1)==fipslb(i,1));
    weights=[weights; (popsb(locs,1)./sum(popsb(locs,1)))];
end

% read in adjacency matrix for small blocks
Rtable=readtable('la_R.csv','ReadRowNames',true);
R=table2array(Rtable);
Rfips=str2num(cell2mat(Rtable.Properties.RowNames));

% align adjacency matrix to ordering of small block data
Rorder=[];
for i=1:size(datsb,1)
    locs=find(Rfips(:,1)==fipssbfull(i,1));
    Rorder=[Rorder locs'];
end

R=R(Rorder,Rorder);

% run function
joint_spatial_fa(datsb,datlb,fipssb,fipslb,weights,R,50000,50000,1,'sigma_m_phi',.5)
