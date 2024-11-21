
clc;
clear;

% 预加载数据
data = load("OrgAlk_get_phil.mat");
orgalk_2100 = data.orgalk_2100_avg;
orgalk_2010 = data.orgalk_2010_avg;

[numRows, numCols, numTimes] = size(orgalk_2010);

TA_2010 = data.TA_2010_avg;
DIC_2010 =data.DIC_2010_avg;
sal_2010 = data.sal_2010_avg;
Temp_2010 = data.Temp_2010_avg;
sil_2010 =data.sil_2010_avg;
po4_2010 = data.po4_2010_avg;


Korg1 = 2.68675 * 1E-6;        % 常数
Korg2 = 6.82858 * 1E-9;
rat = 0.4035;

% 初始化结果矩阵
Ar_norg_2010 = NaN(numRows, numCols, numTimes);
pHT_norg_2010 = NaN(numRows, numCols, numTimes);


for i = 1:numRows
    for j = 1:numCols
        for t = 1:numTimes

            par1type = 1; 
            par1 = TA_2010(i, j, t); 
            par2type = 2; 
            par2 = DIC_2010(i, j, t); 
            sal = sal_2010(i, j, t);
            tempin = Temp_2010(i, j, t);
            tempout = Temp_2010(i, j, t);
            presin = 0; 
            presout = 0; 
            sil = sil_2010(i, j, t); 
            po4 = po4_2010(i, j, t); 
            pHscale = 1;  
            k1k2c = 10;  
            kso4c = 1;  
                
            T_orgalk = 0;  % Convert to mol/kg
                

                % Perform the CO2SYS calculation
                A = CO2SYS(T_orgalk, Korg1, Korg2, rat, par1, par2, par1type, par2type, ...
                           sal, tempin, tempout, presin, presout, sil, po4, pHscale, k1k2c, kso4c);

                % Store the results
                Ar_norg_2010(i, j, t) = A(31);  % Aragonite saturati2100on state
                pHT_norg_2010(i, j, t) = A(37);  % Total pH
            end
    end
end