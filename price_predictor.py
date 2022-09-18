import streamlit as st
import numpy as np
import pickle as pk

# importing model
df = pk.load(open('df.pkl','rb'))
pipe = pk.load(open('pipe.pkl','rb'))

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(In GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Enter weight of laptop')

# touchscreen
touchscreen = st.selectbox('Touchscreen',['Yes','NO'])

# ips
ips = st.selectbox('IPS',['Yes','NO'])

# Screensize
screensize = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Resolution',['1920*1080','1366*768','1600*900','3840*2160','3200*1800','2880*1800'
    ,'2560*1600','2500*1440','2304*1440'])

# cpu
cpu = st.selectbox('CPU',df['Cpu brand'].unique())

# hdd
hdd = st.selectbox('HDD(IN GB)',[0,128,256,512,1024, 2048])

# SDD
sdd = st.selectbox('SDD(IN GB)',[0,8,128,256,512,1024])

# Gpu brand
gpu = st.selectbox('GPU Brand',df['Gpu brand'].unique())

# OS
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None

    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0


    if ips == 'Yes' :
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('*')[0])
    Y_res = int(resolution.split('*')[1])
    ppi = (((X_res**2)+(Y_res**2))**0.5)/screensize
    query_obj = np.array([company , type , ram , weight,touchscreen,ips,ppi,cpu,hdd,sdd,gpu,os])

    query_obj = query_obj.reshape(1,12)
    st.title("The price prediction for above parameters is: " + str(int(np.exp(pipe.predict(query_obj)).round())))
