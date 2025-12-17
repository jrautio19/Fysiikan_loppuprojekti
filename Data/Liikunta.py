import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import folium
from streamlit_folium import st_folium

url1 = './Data/Linear Acceleration.csv'
url2 = './Data/Location.csv'
df_a = pd.read_csv(url1)
df_l = pd.read_csv(url2)

df_a = df_a[df_a['Time (s)'] >= 126]
df_a = df_a.reset_index(drop = True)

st.title('Liikuntadata')

fig, ax = plt.subplots(figsize=(14,10))
plt.subplot(3,1,1)
plt.plot(df_a['Time (s)'],df_a['Linear Acceleration x (m/s^2)'])
plt.ylabel('Acceleration x')
plt.subplot(3,1,2)
plt.plot(df_a['Time (s)'],df_a['Linear Acceleration y (m/s^2)'])
plt.ylabel('Acceleration y')
plt.subplot(3,1,3)
plt.plot(df_a['Time (s)'],df_a['Linear Acceleration z (m/s^2)'])
plt.ylabel('Acceleration z')
plt.title('Askelmittaus')
plt.xlabel('Aika [s]')
#st.pyplot(fig)

from scipy.signal import butter,filtfilt
def butter_lowpass_filter(data, cutoff, nyq, order):
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

data = df_a['Linear Acceleration y (m/s^2)']
T_tot = df_a['Time (s)'].max()
n = len(df_a['Time (s)']) 
fs = n/T_tot 
nyq = fs/2 
order = 3
cutoff = 1/0.4

data_filt = butter_lowpass_filter(data, cutoff, nyq, order)

jaksot = 0
for i in range(n-1):
    if data_filt[i]/data_filt[i+1] < 0:
        jaksot = jaksot + 1/2

st.write('Askelmäärä laskettuna suodatuksen avulla:', jaksot, 'askelta')

signal = df_a['Linear Acceleration y (m/s^2)']
t = df_a['Time (s)']
N = len(signal) 
dt = np.max(t)/N 

fourier = np.fft.fft(signal,N) 
psd = fourier*np.conj(fourier)/N 
freq = np.fft.fftfreq(N,dt) 
L = np.arange(1,int(N/2)) 

f_max = freq[L][psd[L] == np.max(psd[L])][0] 
T = 1/f_max 
steps = f_max*np.max(t)
st.write('Askelmäärä laskettuna Fourier-analyysin avulla:', np.round(steps), 'askelta')

df_l = df_l[df_l['Horizontal Accuracy (m)'] <9]
df_l = df_l.reset_index(drop = True)

from math import radians, cos, sin, asin, sqrt
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r 

df_l['Distance_calc'] = np.zeros(len(df_l))
for i in range(len(df_l)-1):
    lon1 = df_l['Longitude (°)'][i]
    lon2 = df_l['Longitude (°)'][i+1]
    lat1 = df_l['Latitude (°)'][i]
    lat2 = df_l['Latitude (°)'][i+1]
    df_l.loc[i+1,'Distance_calc'] = haversine(lon1, lat1, lon2, lat2)

total_distance_km = df_l['Distance_calc'].sum()
total_distance_m = total_distance_km * 1000

askelpituus_m = total_distance_m / jaksot
askelpituus_cm = askelpituus_m * 100

st.write('Keskinopeus:', round(df_l['Velocity (m/s)'].mean(), 1), 'm/s')
st.write('Kokonaismatka:', round(total_distance_km, 2), 'km')
st.write('Askelpituus on ', round(askelpituus_cm), 'cm')
st.write('Huom! Olen ensin kävellyt hitaasti, sitten nopeasti ja lopuksi juossut. Jouduin pysähtymään liikennevaloihin kaksi kertaa mittauksen aikana, jolloin myös juoksin. Tämä näkyykin hyvin suodatetussa kiihtyvyysdatassa. Tämä vaikuttaa askelmäärään ja askelpituuteen.')

fig, ax = plt.subplots(figsize=(13,5))
plt.plot(df_a['Time (s)'],data_filt)
plt.grid()
plt.legend()
plt.xlabel('Aika (s)')
plt.ylabel('Suodatettu a_y (m/s^2)')
st.title('Suodatetun kiihtyvyysdatan y-komponentti')
st.pyplot(fig)

t0 = df_a['Time (s)'].min()
t1 = t0 + 60

mask = (df_a['Time (s)'] >= t0) & (df_a['Time (s)'] <= t1)

time_plot = df_a.loc[mask, 'Time (s)'] - t0
data_plot = data_filt[mask]

fig, ax = plt.subplots(figsize=(13,5))
plt.plot(time_plot, data_plot)
plt.grid()
plt.legend()
plt.xlabel('Aika (s)')
plt.ylabel('Suodatettu a_y (m/s^2)')
st.title('Suodatetun kiihtyvyysdatan y-komponentti (ensimmäiset 60s)')
st.pyplot(fig)

fig, ax = plt.subplots(figsize=(15,6))
plt.plot(freq[L],psd[L].real)
plt.xlabel('Taajuus [Hz]')
plt.ylabel('Teho')
plt.axis([0,16,0,16000])
st.title('Tehospektri')
st.pyplot(fig)

start_lat = df_l['Latitude (°)'].mean()
start_long = df_l['Longitude (°)'].mean()
m = folium.Map(location = [start_lat,start_long], zoom_start = 14)

folium.PolyLine(df_l[['Latitude (°)','Longitude (°)']], color = 'blue', weight = 3.5, opacity = 1).add_to(m)

st.title('Karttakuva')
st_map = st_folium(m, width=900, height=650)
