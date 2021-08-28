# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:43:07 2021

@author: mraslan
"""
import joblib
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import ccxt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from streamlit import caching
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

api_key=st.secrets['api_key']
api_secret=st.secrets['api_secret']
client = Client(api_key, api_secret)

@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def get_orderbook(symbol):
    r = requests.get("https://api.binance.com/api/v3/depth",
                     params=dict(symbol=symbol,limit=1000))
    results = r.json()

    frames = {side: pd.DataFrame(data=results[side], columns=["price", "quantity"],
                                 dtype=float)
              for side in ["bids", "asks"]}

    frames_list = [frames[side].assign(side=side) for side in frames]
    data = pd.concat(frames_list, axis="index", 
                     ignore_index=True, sort=True)
    prices=pd.DataFrame(client.get_ticker())
    price=float(prices[prices['symbol']==symbol].lastPrice.max())
    data['change']=data['price']/price
    data=data[(abs(data['change'])<1.3) &((abs(data['change'])>0.7)) ]
    bid=data[data['side']=='bids'].sort_values('quantity',ascending=False)[:3]
    ask=data[data['side']=='asks'].sort_values('quantity',ascending=False)[:3]
    orderbook=pd.concat([bid,ask])
  # orderbook=data.sort_values('quantity',ascending=False)[:6]
    return orderbook
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def pump(symbol,profit_flag=1,tf='15m',duration=6):
    duration=str(duration) +"  ago UTC"
    df=pd.DataFrame(client.get_historical_klines(symbol.replace("/",""), tf, duration),columns=['Time','Open','High','Low','Close','Volume','Close time','Quote asset volume','Number of trades','Taker buy base asset volume','Taker buy quote asset volume','ignore'])
    df=df.astype( dtype={
                     'Open': float,
                     'High': float,
                     'Low': float,
                     'Close': float,
                     'Volume': float,

                     'Quote asset volume': float,
                     'Number of trades': float,
                      'Taker buy base asset volume': float,
                     'Taker buy quote asset volume': float,
                     'ignore': float

                     })
   
    z=pd.DataFrame()
    z['Date']=pd.to_datetime(df['Time']*1000000)
    z['Open']=df['Open']
    z['High']=df['High']
    z['Low']=df['Low']
    z['trades_taker']=df['Taker buy quote asset volume']/df['Quote asset volume']
    #z['trades_Vol_taker']=df['Taker buy quote asset volume']/df['Number of trades']
    z['trades_vol_trade']=df['Quote asset volume']/df['Number of trades']
    z['Taker buy quote asset volume']=df['Taker buy quote asset volume']
    z['Number of trades']=df['Number of trades']
    z['Quote asset volume']=df['Quote asset volume']
    z['trades_maker_volume']=df['Quote asset volume']-df['Taker buy quote asset volume']
    #z['trades_maker']=z['trades_maker_volume']/df['Number of trades']
    z['Delta']=df['Taker buy quote asset volume']-z['trades_maker_volume']
    z['percent_buy']=df['Taker buy quote asset volume']*100/df['Quote asset volume']
    z['Close']=df['Close']
    z['price_change']=(df['Close']-df['Open'])*100/df['Open']
    #z['Volume']=df['Volume']
  
    z=z.set_index('Date')
    #z['Close'].plot()
    #z['Delta'][:-1].plot(secondary_y=True)
    #z['trades_maker_volume'].plot(secondary_y=True)
    #z.sort_values('KPI',ascending=False)
    z['Delta_change']=(z['Delta']-z['Delta'].shift(1))/z['Delta']
     
    z=z[(abs(z['Delta_change'])<np.inf)]
    z['KPI']=z['Delta_change']*z['percent_buy']
 
    z['signal']=z['Delta_change'].apply(lambda x: signal(x))
    z['profit']=0
    if profit_flag==1:
        for i in z.index:
            t=z[z.index==i][:(4*24*7)]
            try:
                if (t['signal']!=0)[0]:
                    p=pd.DataFrame()
                    p=z[z.index>i]
                    z['profit'][i]=p.Close.max()*100/z[z.index==i].Close.max()
                    
               
            except:
                continue
            
    #z['Delta_change'].plot(secondary_y=True)
    return z
    #z.plot(subplots=True,layout=(6,3),figsize=(20,10))

ex=ccxt.binance()
f=pd.DataFrame(ex.fetch_markets())
symbols=f[f['active']==True].symbol.unique()


s=[]
u=[]
z=[]
for symbol in symbols:
    if symbol.split('/')[1]=='BTC':
        s.append(symbol)
    if symbol.split('/')[1]=='USDT':
        u.append(symbol.split('/')[0])
        z.append(symbol)

#since=since = ex.milliseconds () - (75*86380000)
symbo=[]
for i in s:
        if i.split('/')[0] not in u:
            if i!='YOYOW/BTC':
                    symbo.append(i)
symbols=[]        
for i in z:
    t=(i.find('UP/') + i.find('DOWN/') + i.find('BULL/') + i.find('BEAR/')+i.find('USDC/')+i.find('PAX/')+i.find('PAXG/'))
    #print(i,'  ',t)
    if(t==-7):
        symbols.append(i)  


#symbols=symbols[symbols not in ['YOYOW/BTC']]


import warnings
warnings.filterwarnings('ignore')


def signal(x):
    sig=0
    if x>500:
        sig=1
    elif x<-500:
        sig=-1
    return sig
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def scan(symbols,tf,duration):
    filename = 'secret_model.sav'
    model = joblib.load(filename)
    df1=pd.DataFrame()
    #st.write(len(symbols))
    for symbol in symbols:
  #  symbol='ARK/BTC'
  
        symbol=symbol.replace("/","")
        #st.write(symbol)
        z=pump(symbol,1,tf,duration)
        
        #z['Close'].plot()
        #z['Delta_change'].plot(secondary_y=True)
        #p=pd.DataFrame()
        
        #p=z[z.index>z[z.index==z.Delta_change.idxmax()].index[0]]

        #z['profit']=p.Close.max()*100/z[z.index==z[abs(z['Delta_change'])<np.inf].Delta_change.idxmax()].Close.max()
        #z=z[z.index==z[abs(z['Delta_change'])<np.inf].Delta_change.idxmax()]
        
        z['symbol']=symbol
        z=z[z['signal']!=0]
      
        #a=z.plot(subplots=True,layout=(6,3),figsize=(20,10))
        df1=pd.concat([z,df1])
    X_real=df1[['signal','Delta_change','percent_buy','Quote asset volume','Number of trades','price_change','Close','Delta','Taker buy quote asset volume']]
    st.write(len(X_real)
    df1['pred']= model.predict(X_real)
   
    return df1
      


# Create figure with secondary y-axis
#fig = make_subplots(specs=[[{"secondary_y": True}]])

@st.cache(allow_output_mutation=True)
def plot_symbol(symbol,profit=0):
    symbol=symbol.replace("/","")
    #st.write(symbol)
    z=pump(symbol,profit)
    fig=go.Figure(data=[go.Candlestick(x=z.index,
                    open=z['Open'],
                    high=z['High'],
                    low=z['Low'],
                    close=z['Close'])])


    z['tmp']=z['signal']*abs(z['Delta_change'])
    z['tmp']=z['tmp'].fillna(0)

    orderbook=get_orderbook(symbol.replace("/",""))
    def color(asks):
        color=0
        if asks=='bids':
            color='Green'
        elif asks=='asks':
            color='Red'
        return color

    for i in range(len(orderbook)):
        p=orderbook['price'].to_list()
        q=(2*orderbook['quantity']/orderbook['quantity'].mean()).to_list()
        asks=orderbook['side'].apply(lambda x: color(x)).to_list()
        fig.add_hline(y=p[i],line_width=q[i], line_dash="dash", line_color=asks[i])
     #           x=orderbook['price'],
      #          y=orderbook['quantity'],
       #         orientation='h'))

    def add_signal(z):
        x=z[z['signal']!=0].index.to_list()
        y=z[z['signal']!=0].Close.to_list()
        m=z[z['signal']!=0].tmp.to_list()
        for i in range(len(x)):
            a=x[i]
            b=y[i]
            c=m[i]
            t='Buy signal at '+ str(round(c,2))
            fig.add_annotation(x=a, y=b,  font=dict(
                family="Courier New, monospace",
                size=14,
                color="Red"
                ),

                text=t,
                showarrow=True,
                arrowhead=1,arrowsize=3,bordercolor="#c7c7c7",
                borderwidth=2,
                borderpad=4,
                bgcolor="lightgreen",
                opacity=0.8)
            fig.add_hline(y=y[i],line_width=3, line_dash="dot", line_color="Black")
    add_signal(z)

    #fig.add_vline(x=z['tmp'].max(), line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(showlegend=False)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,

    )
    fig.update_yaxes(automargin=True)
    #fig.show()
    #z['Close'].plot(figsize=(20,10))

    #z['tmp'].plot(secondary_y=True)

    # z     
    #z=z[z.index==z[abs(z['Delta_change'])<np.inf].Delta_change.idxmax()]
    #z['symbol']=symbol
    a=z.plot(subplots=True,layout=(6,3),figsize=(20,10))
    #return z
    return fig,z

tf=st.selectbox('Time Frame',['1m','5m','15m','1h','4h','1d','1w','1M'])
duration=st.text_input('Number of hours/days before','1 day') 
ss=st.selectbox('USDT or BTC',['USDT','BTC'])
if ss=='BTC':
    symbols=symbo
df1=scan(symbols,tf,duration)
strt=st.text_input('Date to filter with ','2021-08-26 00:00:00')
AI=st.selectbox('Add AI in prediction',['yes','no'])
if AI=='Yes':
    AI=1
else :
    AI=0

df=df1[df1.index>strt]
df=df[df['pred']==AI]
symbols_f=df[df['signal']!=0].symbol.unique()
st.write(len(symbols_f))
symbol=st.sidebar.radio('Symbol',symbols_f)
          
fig,z=plot_symbol(symbol,profit=0)
st.write(fig)
st.dataframe(df)
