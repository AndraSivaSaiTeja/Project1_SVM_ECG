function [] = ECGFD2(ecg,Fs,Annotation)
%% Input ecg is ECG Signal sampled at Fs(>250Hz,multiple of 250), should be a row vector
%% Electro-CardioGram Features Detector(ECGFD)
%% Initialising P,Q,S,T peak,wave Matrices
Plocs = [];
Qlocs = [];
Slocs = [];
Tlocs = [];
Jlocs = [];
P_off = [];
T_off = [];

%% Downsampling to 250Hz
fn = Fs/250;
ecg0 = downsample(ecg,fn);

%% Noise Removal ,using 0.5-40Hz BPF
f1 = 0.5;
f2 = 40;
fs = 250;
w1 = (2*f1)/fs;
w2 = (2*f2)/fs;
N = 3;
[a,b] = butter(N,[w1 w2]);
ecgfiltered = filtfilt(a,b,ecg0);

%% Normalising and getting inverted signal
ecgfiltered1 = ecgfiltered./max(ecgfiltered);
out1 = ecgfiltered1;
out2 = -1*ecgfiltered1;

%% R peaks detecting
R = 0.5;
[Rpks,Rlocs] = findpeaks(out1,'MinPeakHeight',R,'MinPeakDistance',90);
nR = length(Rlocs);

%% P,Q,S,T Peaks detecting
for i = (1:1:(nR-1))
    window_start = Rlocs(1,i);
    window_end = Rlocs(1,(i+1));
    window1 = out1(window_start:window_end);
    window2 = out2(window_start:window_end);
    nw1 = length(window1);
    interval1 = floor(2*nw1/3);
    window1_1 = window1(1:interval1);
    window1_2 = window1((interval1+1):nw1);
    [pks1,locs1] = findpeaks(window1_1);
    [T1,Tl1] = max(pks1);
    Tlocs1 = (locs1(1,Tl1)+Rlocs(1,i)-1);%T peak if it is not inverted,probably U peak if T is inverted
    [pks2,locs2] = findpeaks(window1_2);
    [P1,Pl1] = max(pks2);
    Plocs1 = locs2(1,Pl1);
    Plocs = [Plocs (Plocs1+(interval1+1)-1+Rlocs(1,i)-1)];%P peaks
    nw2 = length(window2);
    interval2 = floor(2*nw2/3);
    window2_1 = window2(1:interval2);
    window2_2 = window2((interval2+1):nw2);
    [pks3,locs3] = findpeaks(window2);
    Slocs = [Slocs (locs3(1,1)+Rlocs(1,i)-1)];%S peaks 
    window3 = out2((Plocs1+(interval1+1)-1+window_start-1):window_end);
    [Q1,Ql1] = findpeaks(window3);
    [Q2,Ql2] = max(Q1);
    Qlocs1 = Ql1(1,Ql2);
    Qlocs = [Qlocs (Qlocs1+Plocs1-1+(interval1+1)-1+Rlocs(1,i)-1)];%Q peaks
    window4 = out2(Slocs(1,i):Plocs(1,i));
    [T2,Tl2] = findpeaks(window4);
    [Tpeaktemp,Tloctemp] = max(T2);
    Tlocs2 = (Tl2(1,Tloctemp)+locs3(1,1)-1+Rlocs(1,i)-1);%T peak if it is inverted 
    t0 = Plocs(1,i) - floor(0.50*(Qlocs(1,i) - Plocs(1,i)));
    Ttemp1 = T1 - out1(1,t0);
    Ttemp2 = Tpeaktemp - out2(1,t0);
    if (abs(Ttemp1) >= abs(Ttemp2))
        Tlocs = [Tlocs Tlocs1];
    else
        Tlocs = [Tlocs Tlocs2];
    end
end

%% J locations detecting
Rpks_avg = mean(Rpks);%Take this height as equivalent to 1mV
Jdiff_threshold = Rpks_avg/100;
nS = length(Slocs);
j = 1;
while (j < (nS+1))
    window_J = out1(Slocs(1,j):((Slocs(1,j))+18));
    k = 1;
    while (k <= 17)
        Jdiff_1 = window_J(1,k+1) - window_J(1,k);
        Jdiff_2 = window_J(1,k+2) - window_J(1,k+1);
        if (((Jdiff_1 < Jdiff_threshold) && (Jdiff_2 < Jdiff_threshold))||( k == 17))
            Jlocs = [Jlocs k+Slocs(1,j)-1];
            j = j + 1;
            k = 18;
        else
            k = k + 1;
        end
    end
end

%% T wave and P wave detecting
P_on = Plocs - floor(0.50*(Qlocs - Plocs));%P_on locations
T_on = Jlocs + floor(0.66*(Tlocs - Jlocs));%T_on locations
nP = length(Plocs);
for p1 = (1:1:nP)
    P_off = [P_off Plocs(1,p1)+floor(0.50*(Qlocs(1,p1) - Plocs(1,p1)))] ;%P_off locations
end
nT = length(Tlocs);
t1 = 1;
while (t1 < (nT+1))
    ttt0 = abs(P_on(1,t1) - Tlocs(1,t1));
    if (ttt0 < 2)
        ttt0 = 2;
    end
    th1 = ecgfiltered(1,P_on(1,t1));
    th2 = ecgfiltered(1,Tlocs(1,t1));
    t2 = 1;
    while (t2 < ttt0)
        tt0 = Tlocs(1,t1) + t2 ;
        th3 = ecgfiltered(1,tt0);
        temp3 = th3 - th1 ;
        temp4 = 0.05*(th2- th1);
        if ((abs(temp3) < abs(temp4))||(t2 == (ttt0-1)))
            T_off = [T_off Tlocs(1,t1)+t2-1];%T_off locations
            t2 = ttt0+1;
            t1 = t1 + 1;
        else
            t2 = t2 + 1;
        end
    end
end

%% Features for SVM
index = length(Rlocs);
for s=(2:1:(index-1))
    is = s;
    F1 = ecgfiltered(Tlocs(1,is)) - ecgfiltered(P_on(1,(is-1)));%T peak height
    F2 = ecgfiltered(1,(Jlocs(1,is) + 10)) - ecgfiltered(1,Jlocs(1,is));%ST elevation/depression
    F3 = -1*(ecgfiltered(Qlocs(1,(is-1))) - ecgfiltered(P_on(1,(is-1))));%Q height
    F = [F1 F2 F3 Annotation];
    temp_table = sprintf('%.4f,',F);
    temp_table(end)= [];
    disp(temp_table);
end

end