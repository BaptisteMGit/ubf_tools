clearvars
close all
clc

%% Chargement du fichier tdms (on peut aussi utiliser uigetfile pour choisir le fichier
%file=uigetfile('.tdms');
file='essai_sweep.tdms';

%Pour connaitre tous les paramètres du fichier, on utilise la fonction tdmsinfo
info=tdmsinfo(file)

% La liste des canaux est un tableau de 10 lignes sur 8 colonne. 
% Pour connaitre les champs, on affiche info.ChannelList
%On voit que les 5 premières lignes sont des données de type "acquisition hydro - données" 
% et qu'elles contiennent 1040000 échantillons (NumSamples)

info.ChannelList
%%
%En sélectionnant le type de données "Acquisition hydros - données" et le canal Hydro 1, 
% on peut obtenir les propriétés de la donnée, 
% en particulier son pas temporel wf_increment en seconde.

proprietes=tdmsreadprop(file, "ChannelGroupName","Acquisition Hydros - Données", "ChannelName", "Hydro1")

pas_temp=proprietes.wf_increment % incrément des données en seconde
Fe=1/pas_temp; % fréquence d'échantillonnage

%% Lecture des données : on lit les données grâce à tdmsread et on fixe la datation avec seconds(pas), 
% cela transforme la variable pas qui est une donnée numérique double en valeur timetable.

data=tdmsread(file,"TimeStep",seconds(pas_temp));

%Les valeurs des signaux sont dans la première boite de data. 

matrice_data=data{1};

% la variable matrice_data de type timetable contient
% 1 colonne de temps Time et 5 colonnes : Hydro1, Hydro2, ..., Hydro5

temps=matrice_data.Time;
hydro1=matrice_data.Hydro1;
hydro2=matrice_data.Hydro2;
hydro3=matrice_data.Hydro3;
hydro4=matrice_data.Hydro4;
hydro5=matrice_data.Hydro5;

%%

figure
plot(temps, hydro1)
title('hydro1')
xlim([seconds(0) seconds(6e-3)])

%%

%figure
%stackedplot(matrice_data) % représente les graphes tous en meme temps

% représentation 
figure
subplot(321)
plot(temps, hydro1)
title('hydro1')
xlabel('temps s')
subplot(322)
plot(temps, hydro2)
title('hydro2')
xlabel('temps s')
subplot(323)
plot(temps, hydro3)
title('hydro3')
xlabel('temps s')
subplot(324)
plot(temps, hydro4)
title('hydro4')
xlabel('temps s')
subplot(325)
plot(temps, hydro5)
title('hydro5')
xlabel('temps s')

%% Calcul du spectre sur le signal hydro1

N=length(hydro1);
spectre=fftshift(fft(hydro1))/N;
freq=linspace(-Fe/2, Fe/2-Fe/N, N);

figure
plot(freq,abs(spectre))
xlabel('fréquence Hz')
ylabel('amplitude spectre')
%xlim([-10e3 10e3])
zoom

grid on