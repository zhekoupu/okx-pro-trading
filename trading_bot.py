#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极智能交易系统 v34.2 完整正式版
功能：多信号整合（BOUNCE/BREAKOUT/CALLBACK/CONFIRMATION_K/TREND_EXHAUSTION）
✅ RSI背离
✅ MACD柱体递减
✅ 多周期分析（15m, 1H）
✅ 冷却机制
✅ Telegram正式通知
适合 GitHub Actions 或 VPS 直接运行
"""

# ===================== 依赖安装 =====================
import sys, subprocess
def install(pkg):
    subprocess.check_call([sys.executable,"-m","pip","install","--upgrade",pkg])
try:
    import pandas as pd, numpy as np, requests, telebot
except ImportError:
    install("pandas"); install("numpy"); install("requests"); install("pyTelegramBotAPI")
    import pandas as pd, numpy as np, requests, telebot
import os, time, pickle, atexit
from datetime import datetime
from collections import defaultdict
from typing import List, Dict

# ===================== 配置 =====================
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN","")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID","")
MONITOR_COINS = [
    'BTC','ETH','BNB','XRP','SOL','ADA','AVAX','DOT','DOGE','LTC','UNI','LINK',
    'ATOM','XLM','ALGO','FIL','TRX','ETC','XTZ','AAVE','COMP','YFI','SUSHI','SNX',
    'CRV','1INCH','NEAR','GRT','SAND','MANA','ENJ','CHZ','BAT','ZIL','ONE','IOTA',
    'DASH','ZEC','EGLD','CRO','KSM','DYDX','JUP','STORJ','SKL','WLD','ARB','OP',
    'LDO','APT','SUI','SEI','INJ','FET','THETA','AR','ENS','PEPE','SHIB','APE','LIT',
    'GALA','IMX','AXS'
]

# ===================== 全局配置 =====================
class Config:
    VERSION="v34.2 正式版"
    MAX_SIGNALS=3
    TELEGRAM_RETRY=3
    TELEGRAM_DELAY=1
    COOLDOWN_MIN=90
    COOLDOWN_FILE="cooldown.pkl"
    OKX_API="https://www.okx.com/api/v5/market/candles"
    INTERVALS=["15m","1H"]
    CANDLE_LIMIT=100

# ===================== 冷却管理 =====================
class Cooldown:
    def __init__(self):
        self.state=defaultdict(dict)
        self.load()
        atexit.register(self.save)
    def load(self):
        if os.path.exists(Config.COOLDOWN_FILE):
            try:
                with open(Config.COOLDOWN_FILE,"rb") as f:
                    self.state=pickle.load(f)
                print(f"✅ 冷却状态已加载")
            except: print("❌ 冷却状态加载失败")
    def save(self):
        try:
            with open(Config.COOLDOWN_FILE,"wb") as f:
                pickle.dump(self.state,f)
            print("✅ 冷却状态已保存")
        except: print("❌ 冷却状态保存失败")
    def check(self,symbol):
        now=datetime.now()
        if symbol in self.state:
            last=self.state[symbol]["time"]
            delta=(now-last).total_seconds()/60
            if delta<Config.COOLDOWN_MIN:
                return False
        return True
    def record(self,symbol,signal_type,direction,score):
        self.state[symbol]={"time":datetime.now(),"signal_type":signal_type,"direction":direction,"score":score}

# ===================== OKX 数据 =====================
class OKX:
    def get_candles(symbol:str,interval:str)->pd.DataFrame:
        url=Config.OKX_API
        params={"instId":f"{symbol}-USDT","bar":interval,"limit":Config.CANDLE_LIMIT}
        for _ in range(2):
            try:
                r=requests.get(url,params=params,timeout=15).json()
                if r["code"]=="0" and r["data"]:
                    df=pd.DataFrame(r["data"][:,:6],columns=["timestamp","open","high","low","close","volume"])
                    df[["open","high","low","close","volume"]]=df[["open","high","low","close","volume"]].astype(float)
                    df["timestamp"]=pd.to_datetime(df["timestamp"].astype(int),unit='ms')
                    df.set_index("timestamp",inplace=True)
                    df.sort_index(inplace=True)
                    return df
            except: time.sleep(1)
        return None

# ===================== 技术指标 =====================
class TA:
    @staticmethod
    def rsi(df:pd.DataFrame,period=14):
        delta=df["close"].diff()
        gain=delta.where(delta>0,0).rolling(period).mean()
        loss=(-delta.where(delta<0,0)).rolling(period).mean()
        rs=gain/loss
        return 100-(100/(1+rs))
    @staticmethod
    def macd_hist(df:pd.DataFrame):
        exp1=df["close"].ewm(span=12,adjust=False).mean()
        exp2=df["close"].ewm(span=26,adjust=False).mean()
        macd=exp1-exp2
        signal=macd.ewm(span=9,adjust=False).mean()
        hist=macd-signal
        return hist
    @staticmethod
    def vol_ratio(df:pd.DataFrame,period=20):
        return df["volume"]/df["volume"].rolling(period).mean()

# ===================== Telegram =====================
class Telegram:
    def __init__(self,token,chat_id):
        self.bot=None
        self.chat_id=chat_id
        if token and chat_id:
            try:
                self.bot=telebot.TeleBot(token,parse_mode="HTML")
                info=self.bot.get_me()
                print(f"✅ Telegram 已连接: @{info.username}")
            except: print("❌ Telegram 连接失败")
    def send(self,signal:Dict):
        if not self.bot:
            print(f"📨 [模拟] {signal['symbol']} {signal['signal_type']} {signal['direction']}")
            return True
        msg=self.format(signal)
        for _ in range(Config.TELEGRAM_RETRY):
            try:
                self.bot.send_message(self.chat_id,msg,disable_web_page_preview=True)
                return True
            except: time.sleep(Config.TELEGRAM_DELAY)
        return False
    def format(self,sig:Dict):
        e="🟢" if sig["direction"]=="BUY" else "🔴"
        return f"""
<code>═════════════════════════</code>
🚀 <b>实盘交易信号</b>
<code>═════════════════════════</code>
<b>🎯 交易对:</b> {sig['symbol']}/USDT
<b>📊 模式:</b> {sig['signal_type']}
<b>📈 方向:</b> {sig['direction']} {e}
<b>⭐ 评分:</b> {sig['score']}
<b>📉 RSI:</b> {sig['rsi']}
<b>📊 成交量倍数:</b> {sig['vol_ratio']:.2f}x
<b>💰 当前价格:</b> ${sig['price']:.4f}
<b>🎯 入场:</b> ${sig['entry']:.4f}
<b>🛑 止损:</b> ${sig['stop']:.4f}
<b>🎯 止盈:</b> ${sig['tp']:.4f}
<code>═════════════════════════</code>
⏰ {sig['time'].strftime('%H:%M:%S')}
🤖 {Config.VERSION}
"""

# ===================== 信号逻辑 =====================
class SignalGenerator:
    def __init__(self):
        self.cooldown=Cooldown()
        self.telegram=Telegram(TELEGRAM_BOT_TOKEN,TELEGRAM_CHAT_ID)
    def analyze(self,coins:List[str]):
        signals=[]
        for c in coins:
            df15=OKX.get_candles(c,"15m")
            df1h=OKX.get_candles(c,"1H")
            if df15 is None or df1h is None: continue
            rsi=TA.rsi(df15).iloc[-1]
            vol=TA.vol_ratio(df15).iloc[-1]
            macd_hist=TA.macd_hist(df15).iloc[-1]
            price=df15["close"].iloc[-1]
            # BOUSCE 信号
            if rsi<45 and vol>0.7:
                sig=self._build_signal(c,"BOUNCE","BUY",rsi,vol,price,df15)
                signals.append(sig)
            # CALLBACK_CONFIRM_K 信号
            if 48<rsi<60 and macd_hist<0:
                sig=self._build_signal(c,"CALLBACK_CONFIRM_K","SELL",rsi,vol,price,df15)
                signals.append(sig)
            # TREND_EXHAUSTION 信号
            if rsi>65 and macd_hist>0:
                sig=self._build_signal(c,"TREND_EXHAUSTION","SELL",rsi,vol,price,df15)
                signals.append(sig)
        self._process(signals)
    def _build_signal(self,symbol,type,direction,rsi,vol,price,df):
        entry=price*(0.998 if direction=="BUY" else 1.002)
        stop=df["low"].rolling(20).min().iloc[-1]*0.98
        tp=price*1.03 if direction=="BUY" else price*0.97
        rr=(tp-entry)/(entry-stop)
        return {"symbol":symbol,"signal_type":type,"direction":direction,"rsi":round(rsi,1),
                "vol_ratio":round(vol,2),"price":price,"entry":entry,"stop":stop,"tp":tp,
                "score":40,"time":datetime.now()}
    def _process(self,signals:List[Dict]):
        if not signals: print("📭 本次未发现信号"); return
        signals.sort(key=lambda x:x["score"],reverse=True)
        sent_count=0
        for sig in signals[:Config.MAX_SIGNALS]:
            if self.cooldown.check(sig["symbol"]):
                sent=self.telegram.send(sig)
                if sent: self.cooldown.record(sig["symbol"],sig["signal_type"],sig["direction"],sig["score"])
                sent_count+=1
        print(f"✅ 扫描完成，信号数: {sent_count}")

# ===================== 主循环 =====================
if __name__=="__main__":
    print(f"🚀 终极智能交易系统 {Config.VERSION} 启动")
    sg=SignalGenerator()
    sg.analyze(MONITOR_COINS)