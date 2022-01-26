# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 14:43:07 2021

@author: mraslan
"""
from xgboost import XGBClassifier
import joblib
import pickle
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime ,timedelta
import ccxt
import numpy as np
from finta import TA
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
                     params=dict(symbol=symbol,limit=5000))
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
    data=data[(abs(data['change'])<3.5) &((abs(data['change'])>0.1)) ]
    
    bid=data[data['side']=='bids']
    
   
    bid=bid.set_index('price')
    binwidth=price*0.02
    bins=np.arange(bid.index.min(), bid.index.max() + binwidth, binwidth)
    f1=[]
    for i in range(1,len(bins)):
        f1.append(bid[(bid.index>bins[i-1]) & (bid.index<bins[i])].quantity.sum())
        
                                                    
    f=pd.DataFrame([bins,f1]).T
    f.columns=columns=['price','quantity']
    f['side']='bids'
    
    bid=f.dropna().sort_values('quantity',ascending=False)[:4]
    ask=data[data['side']=='asks'].sort_values('quantity',ascending=False)
    ask=ask.set_index('price')
    binwidth=price*0.02
    bins=np.arange(ask.index.min(), ask.index.max() + binwidth, binwidth)
    f1=[]
    for i in range(1,len(bins)):
        f1.append(ask[(ask.index>bins[i-1]) & (ask.index<bins[i])].quantity.sum())
    f=pd.DataFrame([bins,f1]).T
    f.columns=columns=['price','quantity']
    f['side']='asks'
    
    ask=f.dropna().sort_values('quantity',ascending=False)[:4]
    orderbook=pd.concat([bid,ask])
    #orderbook=orderbook.reset_index()
  # orderbook=data.sort_values('quantity',ascending=False)[:6]
    return orderbook
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def pump(symbol,profit_flag=1,tf='15m',duration='2 days'):
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
    z['open']=df['Open']
    z['high']=df['High']
    z['low']=df['Low']
    z['close']=df['Close']
    z['RSI']=TA.RSI(z)
    z['RSI_shifted']=z['RSI'].shift(1)
    z['trades_taker']=df['Taker buy quote asset volume']/df['Quote asset volume']
    #z['trades_Vol_taker']=df['Taker buy quote asset volume']/df['Number of trades']
    z['trades_vol_trade']=df['Quote asset volume']/df['Number of trades']
    z['Taker buy quote asset volume']=df['Taker buy quote asset volume']
    z['Number of trades']=df['Number of trades']
    z['Quote asset volume']=df['Quote asset volume']
    z['trades_maker_volume']=df['Quote asset volume']-df['Taker buy quote asset volume']
    #z['trades_maker']=z['trades_maker_volume']/df['Number of trades']
    z['Delta']=z['trades_maker_volume']-df['Taker buy quote asset volume']
    z['percent_buy']=df['Taker buy quote asset volume']*100/df['Quote asset volume']
    z['Close']=df['Close']
    z['Delta/Total']=z['Delta']*100/z['Quote asset volume']
    z['price_change']=(df['Close']-df['Open'])*100/df['Open']
    #z['Volume']=df['Volume']
    z['Delta_shifted_old']=z['Delta'].shift(-1)
    z['Delta/Total_shifted']=z['Delta'].shift(-1)*100/z['Quote asset volume'].shift(-1)
    z['Delta_shifted_old_2']=z['Delta'].shift(-2)
    z['Delta/Total_shifted_2']=z['Delta'].shift(-2)*100/z['Quote asset volume'].shift(-2)
    z=z.set_index('Date')
    #z['Close'].plot()
    #z['Delta'][:-1].plot(secondary_y=True)
    #z['trades_maker_volume'].plot(secondary_y=True)
    #z.sort_values('KPI',ascending=False)
    z['Delta_change']=(z['Delta']-z['Delta'].shift(1))/z['Delta'].shift(1)
    z['Delta_shifted_old']=z['Delta'].shift(-1)
    z['Delta/Total_shifted']=z['Delta'].shift(-1)*100/z['Quote asset volume'].shift(-1)
    z=z[(abs(z['Delta_change'])<np.inf)]
    z['KPI']=z['Delta']*abs(z['Delta/Total'])/100
    #z['signal']= z[abs(z['Delta/Total'])>90]['KPI'].apply(lambda x: signal(x))
    #z['signal']=z['signal'].fillna(0)
    #z['signal']= z['Delta/Total'].apply(lambda x: signal(x))
    z['signal']=z['Delta_change'].apply(lambda x: signal(x))
    #z['signal']=z['signal'].fillna(0)
    z['profit']=0
    z['profit_duration']=0
    z['loss_duration']=0
    z['loss']=0
    if profit_flag==1:
        
        for i in z.index:
           
            t= z.loc[i:i+timedelta(2)]
            
            f= z.loc[i:i-timedelta(2)]
            #t=z[(z.index>=i) & (z.index<i+timedelta(days=2))]
           # f=z[(z.index<=i) & (z.index>(i-timedelta(days=2)))]
            try:
                if (t['signal']!=0)[0]:
                   
                    z['profit'][i]=t.High.max()*100/z[z.index==i].Close.max()
                    
                    z['Loss'][i]=t.Close.min()*100/z[z.index==i].Close.max()
                    z['profit_duration'][i]=t.High.idxmax()-i
                    z['Loss_duration'][i]=t.Close.idxmin()-i
                if (f['signal']!=0)[0]:
                    z['signal_count'][i]=f.signal.count()
                    
            except:
                continue
            
    #z['Delta_change'].plot(secondary_y=True)
    
    print(len(z))
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
            if (i!='YOYOW/BTC') and (i!='WBTC/BTC'):
                    symbo.append(i)
symbols=[]        
for i in z:
    t=(i.find('UP/') + i.find('DOWN/') + i.find('BULL/') + i.find('BEAR/')+i.find('USDC/')+i.find('PAX/')+i.find('PAXG/')+i.find('TUSD/')+i.find('USDP/')+i.find('EUR/')+i.find('SUSD/')+i.find('BUSD/'))
    #print(i,'  ',t)
    if(t==-12):
        symbols.append(i)  


#symbols=symbols[symbols not in ['YOYOW/BTC']]


import warnings
warnings.filterwarnings('ignore')


def signal(x):
    sig=0
    if x>100:
        sig=1
    elif x<-100:
        sig=-1
    return sig
@st.cache(allow_output_mutation=True,suppress_st_warning=True)
def scan(symbols,tf,duration):
    #pickle_in = open('classifier.pkl', 'rb') 
    #model = pickle.load(pickle_in)
    #filename = 'secret_model.sav'
    #model = joblib.load(filename)
    if symbols[0].split("/")[1]=='USDT':
        model = XGBClassifier()
        model.load_model('model_BTC_1h_40.sav')  # load data
    elif symbols[0].split("/")[1]=='BTC':
        model = XGBClassifier()
        st.write("BTC loaded")
        model.load_model('model_BTC_15m_40 (1).sav')  # load data
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
    df1=df1.dropna()
    #X_real=df1[['signal','Delta_change','percent_buy','Quote asset volume','Number of trades','price_change','Close','Delta','Taker buy quote asset volume']]
    #X_real=df1[['Delta_change','percent_buy','Quote asset volume','Number of trades','price_change','Close','Delta','Taker buy quote asset volume','Delta_shifted_old','Delta/Total_shifted','Delta_shifted_old_2','Delta/Total_shifted_2']]
    X_real=df1[['Delta_change','percent_buy','Quote asset volume','Number of trades','RSI',	'RSI_shifted','price_change','Close','Delta','Taker buy quote asset volume','Delta_shifted_old','Delta/Total_shifted','Delta_shifted_old_2','Delta/Total_shifted_2']]

    yy= model.predict(X_real)
   # yy=0
    df1['pred']=yy
   
    return df1
      


# Create figure with secondary y-axis
#fig = make_subplots(specs=[[{"secondary_y": True}]])

@st.cache(allow_output_mutation=True)
def plot_symbol(symbol,profit=0,tf='1h',duration='3 day'):
    #st.write(symbol)
    symbol=symbol.replace("/","")
    #st.write(symbol)
    z=pump(symbol,profit,tf,duration)
    fig=go.Figure(data=[go.Candlestick(x=z.index,
                    open=z['Open'],
                    high=z['High'],
                    low=z['Low'],
                    close=z['Close'])])


    z['tmp']=z['signal']*abs(z['KPI'])
    z['tmp']=z['tmp'].fillna(0)
    model = XGBClassifier()
    if symbols[0].split("/")[1]=='USDT':
        model = XGBClassifier()
        model.load_model('model_BTC_1h_40.sav')  # load data
    elif symbols[0].split("/")[1]=='BTC':
        model = XGBClassifier()
        st.write("BTC loaded")
        model.load_model('model_BTC_15m_40 (1).sav') 
    z=z.dropna()
   # X_real=z[['Delta_change','percent_buy','Quote asset volume','Number of trades','price_change','Close','Delta','Taker buy quote asset volume','Delta_shifted_old','Delta/Total_shifted','Delta_shifted_old_2','Delta/Total_shifted_2']]
    X_real=z[['Delta_change','percent_buy','Quote asset volume','Number of trades','RSI',	'RSI_shifted','price_change','Close','Delta','Taker buy quote asset volume','Delta_shifted_old','Delta/Total_shifted','Delta_shifted_old_2','Delta/Total_shifted_2']]

    yy= model.predict(X_real)
   # yy=0
    z['pred']=yy

    orderbook=get_orderbook(symbol.replace("/",""))
    #st.dataframe(orderbook)
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
        x=z[(z['signal']!=0) &  (z['pred']==1)].index.to_list()
        y=z[(z['signal']!=0) &  (z['pred']==1)].Close.to_list()
        m=z[(z['signal']!=0) &  (z['pred']==1)].tmp.to_list()
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
    fig.update_layout(
    dragmode='drawopenpath',
    newshape_line_color='black',
    title_text='Draw a path to separate versicolor and virginica'
    )

    #fig.add_vline(x=z['tmp'].max(), line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(showlegend=False)
    fig.update_layout( height=400,width=800, margin=dict(r=5, l=5, t=5, b=5))
    fig.update_layout( title={
        'text': symbol,
        'y':0.9,
        'x':0.5,
        'xanchor': 'left',
        'yanchor': 'top'})
    
    fig.update_yaxes(automargin=True,autorange=True,fixedrange=False)
    #fig.update_xaxes(automargin=True)
    #fig.show()
    #z['Close'].plot(figsize=(20,10))

    #z['tmp'].plot(secondary_y=True)

    # z     
    #z=z[z.index==z[abs(z['Delta_change'])<np.inf].Delta_change.idxmax()]
    #z['symbol']=symbol
    #a=z.plot(subplots=True,layout=(6,3),figsize=(20,10))
    #return z
    return fig,z

tf=st.selectbox('Time Frame',['1m','5m','15m','1h','4h','1d','1w','1M'])
duration=st.text_input('Number of hours/days before','1 day') 
ss=st.selectbox('USDT or BTC',['USDT','BTC'])

if ss=='BTC':
    symbols=symbo

df1=scan(symbols,tf,duration)
flags=st.button('rescan again')
if flags==1:
    caching.clear_cache()
    df1=scan(symbols,tf,duration)
strt=st.text_input('Date to filter with ','2022-01-01 00:00:00')
df=df1[df1.index>strt]
sig=st.selectbox('positive delta or negative delta ',['P','N'])
df1=df
#if sig=='P':
 #   df1=df[df['signal']==1]
#elif sig=='N':
 #   df1=df[df['signal']==-1]
total=len(df1[df1['signal']!=0].symbol.unique())
st.write('All signals detected are for symbols '+str(total))
AI=st.selectbox('Add AI in prediction',['yes','no'])
if AI=='yes':
    AI=1
    df1=df1[df1['pred']==AI]
    df1=df1[df1['pred']==AI]
    total=len(df[df['signal']!=0].symbol.unique())
    detected_AI=len(df[df['pred']!=0].symbol.unique())
    st.write('AI signals detected from all '+str(round(detected_AI/total,2)))
else :
    AI=0
    df=df1




symbols_f=df1[df1['signal']!=0].symbol.unique()
st.write(len(symbols_f))
pf=st.number_input('filter for profit %',100.0)
profit=len(df1[df1['profit']>=pf])
st.write('win rate ',str(round(profit*100/len(df1),2)))
symbol=st.sidebar.radio('Symbol',symbols_f)
print(fig)
fig,z=plot_symbol(symbol,0,tf=tf,duration=6)
config={'scrollZoom': True,'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]}
#st.write(fig,config=config)
st.plotly_chart(fig, use_container_width=False, **{'config': config})
df=df.drop(columns=['Open','High','Low'],axis=1)

st.dataframe(df1)


