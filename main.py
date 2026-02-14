#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极智能交易系统 v33.9 GitHub Actions 适配版
从环境变量读取 Telegram 配置，单次运行模式
"""

# ============ 自动安装依赖 ============
import subprocess
import sys
import os
import atexit
import time
import threading
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque

def install_packages():
    """自动安装缺失的Python包"""
    required_packages = ['pandas', 'numpy', 'requests', 'pyTelegramBotAPI', 'scipy']

    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"🔧 正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装完成")

# 安装依赖
print("🔧 检查并安装依赖...")
install_packages()

# ============ 导入库 ============
import pandas as pd
import numpy as np
import telebot
import requests
import json
import pickle
import hashlib

print("🔧 检查TA-Lib依赖...")
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib已安装，启用高级技术指标")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib未安装，将使用备用技术指标")

# ============ Telegram配置 - 从环境变量读取 ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8455563588:AAERqF8wtcQUOojByNPPpbb0oJG-7VMpr9s")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "2004655568")

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    print(f"🤖 Telegram配置: 已从环境变量读取")
    print(f"   令牌: {TELEGRAM_BOT_TOKEN[:10]}...{TELEGRAM_BOT_TOKEN[-10:]}")
    print(f"   聊天ID: {TELEGRAM_CHAT_ID}")
else:
    print("⚠️ Telegram环境变量未设置，通知功能将禁用")

# 🔧 OKX API配置
OKX_API_BASE_URL = "https://www.okx.com"
OKX_CANDLE_INTERVAL = ["15m", "1H"]
OKX_CANDLE_LIMIT = 100  # 优化：减少数据量，提高速度

# 🔧 监控币种列表 - 使用您的完整64个币种
MONITOR_COINS = [
    # 主流币
    'BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'AVAX', 'DOT',
    'DOGE', 'LTC', 'UNI', 'LINK', 'ATOM', 'XLM', 'ALGO',
    'FIL', 'TRX', 'ETC', 'XTZ', 'AAVE', 'COMP', 'YFI',
    'SUSHI', 'SNX', 'CRV', '1INCH', 'NEAR', 'GRT', 'SAND',
    'MANA', 'ENJ', 'CHZ', 'BAT', 'ZIL', 'ONE', 'IOTA',
    'DASH', 'ZEC', 'EGLD', 'CRO', 'KSM', 'DYDX', 'JUP',
    'STORJ', 'SKL', 'WLD',

    # Layer 2和热门币种
    'ARB', 'OP', 'LDO', 'APT', 'SUI', 'SEI', 'INJ',
    'FET', 'THETA', 'AR', 'ENS', 'PEPE', 'SHIB',
    'APE', 'LIT', 'GALA', 'IMX', 'AXS'
]

print(f"📊 监控币种列表: {len(MONITOR_COINS)}个币种")

# ============ 系统配置类 ============
class UltimateConfig:
    """终极系统配置类"""

    # 基础配置
    VERSION = "33.9-GitHubActions适配版"
    ANALYSIS_INTERVAL = 45  # 仅用于显示，实际由外部调度控制
    COINS_TO_MONITOR = len(MONITOR_COINS)
    MAX_SIGNALS = 8  # 最大信号数量调整为8个

    # 冷却配置 - 实盘优化
    COOLDOWN_CONFIG = {
        'same_coin_cooldown': 90,  # 同币种冷却时间保持90分钟
        'same_direction_cooldown': 45,  # 同方向冷却时间保持45分钟
        'max_signals_per_coin_per_day': 5,  # 每日最大信号数量保持5个
        'enable_cooldown': True
    }

    # 信号门槛优化
    SIGNAL_THRESHOLDS = {
        'BOUNCE': 25,           # 降低门槛
        'BREAKOUT': 25,         # 降低门槛
        'TREND_EXHAUSTION': 35, # 降低门槛
        'CALLBACK': 30,         # 降低门槛
        'CONFIRMATION_K': 40,   # 降低门槛
        'CALLBACK_CONFIRM_K': 45 # 大幅降低门槛
    }

    # 优化参数 - 提高信号发现率
    OPTIMIZATION_PARAMS = {
        'volume_ratio_min': 0.7,      # 最小成交量倍数降低
        'rsi_bounce_max': 45,         # 反弹RSI上限提高
        'rsi_callback_min': 48,       # 回调RSI下限降低
        'callback_pct_min': 2,        # 最小回调幅度降低
        'callback_pct_max': 25,       # 最大回调幅度提高
        'trend_exhaustion_rsi_min': 65 # 趋势衰竭RSI下限降低
    }

    # OKX API配置
    OKX_CONFIG = {
        'base_url': OKX_API_BASE_URL,
        'candle_endpoint': '/api/v5/market/candles',
        'intervals': OKX_CANDLE_INTERVAL,
        'limit': OKX_CANDLE_LIMIT,
        'rate_limit': 20,
        'retry_times': 2,
        'timeout': 15
    }

    # Telegram配置
    TELEGRAM_CONFIG = {
        'enabled': bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID),
        'parse_mode': 'HTML',
        'always_send_signals': True,
        'send_market_reports': False,
        'send_classification_reports': False
    }

# ============ 冷却管理器 ============
class CooldownManager:
    """冷却管理器 - 防止重复信号"""

    def __init__(self):
        self.config = UltimateConfig.COOLDOWN_CONFIG
        self.cooldown_db = {}
        self.signal_history = defaultdict(list)
        self.cooldown_file = 'cooldown_state.pkl'
        self.load_state()

        atexit.register(self.save_state)

    def load_state(self):
        """加载冷却状态"""
        try:
            if os.path.exists(self.cooldown_file):
                with open(self.cooldown_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cooldown_db = data.get('cooldown_db', {})
                    self.signal_history = defaultdict(list, data.get('signal_history', {}))
                print(f"✅ 冷却状态已加载: {len(self.cooldown_db)}个币种记录")
        except Exception as e:
            print(f"❌ 加载冷却状态失败: {e}")
            self.cooldown_db = {}
            self.signal_history = defaultdict(list)

    def save_state(self):
        """保存冷却状态"""
        try:
            data = {
                'cooldown_db': self.cooldown_db,
                'signal_history': dict(self.signal_history)
            }
            with open(self.cooldown_file, 'wb') as f:
                pickle.dump(data, f)
            print("✅ 冷却状态已保存")
        except Exception as e:
            print(f"❌ 保存冷却状态失败: {e}")

    def check_cooldown(self, symbol: str, direction: str) -> Tuple[bool, str]:
        """检查冷却状态"""
        if not self.config['enable_cooldown']:
            return True, ""

        now = datetime.now()

        # 检查同币种冷却
        if symbol in self.cooldown_db:
            last_signal_time = self.cooldown_db[symbol]['time']
            cooldown_minutes = self.config['same_coin_cooldown']

            if (now - last_signal_time).total_seconds() / 60 < cooldown_minutes:
                remaining = cooldown_minutes - (now - last_signal_time).total_seconds() / 60
                return False, f"同币种冷却中 ({remaining:.1f}分钟)"

        return True, ""

    def record_signal(self, symbol: str, direction: str, pattern: str, score: int):
        """记录信号"""
        now = datetime.now()

        # 更新冷却记录
        self.cooldown_db[symbol] = {
            'time': now,
            'direction': direction,
            'pattern': pattern,
            'score': score
        }

        # 更新历史记录
        self.signal_history[symbol].append({
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'direction': direction,
            'pattern': pattern,
            'score': score
        })

# ============ OKX数据获取器 ============
class OKXDataFetcher:
    """OKX数据获取器"""

    def __init__(self):
        self.config = UltimateConfig.OKX_CONFIG
        self.base_url = self.config['base_url']
        self.endpoint = self.config['candle_endpoint']
        self.intervals = self.config['intervals']
        self.limit = self.config['limit']
        self.retry_times = self.config['retry_times']
        self.timeout = self.config['timeout']

        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }

        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 120

    def get_candles(self, symbol: str, interval: str):
        """获取指定周期的K线数据"""
        cache_key = f"{symbol}_{interval}"

        # 检查缓存
        current_time = time.time()
        if cache_key in self.cache:
            if current_time - self.cache_time.get(cache_key, 0) < self.cache_duration:
                return self.cache[cache_key]

        inst_id = f"{symbol}-USDT"
        params = {'instId': inst_id, 'bar': interval, 'limit': self.limit}
        url = f"{self.base_url}{self.endpoint}"

        for retry in range(self.retry_times):
            try:
                response = requests.get(url, params=params, headers=self.headers, timeout=self.timeout)

                if response.status_code == 200:
                    data = response.json()

                    if data['code'] == '0' and len(data['data']) > 0:
                        candles = data['data']
                        df = pd.DataFrame(candles)

                        if len(df.columns) >= 6:
                            df = df.iloc[:, :6]
                            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

                            # 数据类型转换
                            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')

                            df.set_index('timestamp', inplace=True)
                            df.sort_index(inplace=True)

                            # 缓存数据
                            self.cache[cache_key] = df
                            self.cache_time[cache_key] = current_time

                            return df
                else:
                    if retry == self.retry_times - 1:
                        print(f"⚠️ {symbol}: 请求失败 {response.status_code}")

            except Exception as e:
                if retry < self.retry_times - 1:
                    time.sleep(1)
                else:
                    print(f"⚠️ {symbol}: 请求异常 {str(e)}")

        return None

    def get_all_coins_data(self, symbols: List[str]):
        """获取所有币种的多周期数据"""
        print(f"\n📡 开始获取 {len(symbols)} 个币种的实时数据...")

        coins_data = {}
        total = len(symbols)

        for i, symbol in enumerate(symbols, 1):
            data_dict = {}
            for interval in self.intervals:
                df = self.get_candles(symbol, interval)
                if df is not None and len(df) >= 30:
                    data_dict[interval] = df

            if data_dict:
                coins_data[symbol] = data_dict
                print(f"[{i}/{total}] {symbol}: ✅ 成功")
            else:
                print(f"[{i}/{total}] {symbol}: ⚠️ 数据不足")

        print(f"\n📊 数据获取完成: {len(coins_data)}/{total} 个币种")
        return coins_data

# ============ 技术指标计算器 ============
class TechnicalIndicators:
    """技术指标计算器"""

    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14):
        """计算RSI指标"""
        if len(data) < period:
            return pd.Series([50] * len(data), index=data.index)
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    @staticmethod
    def calculate_ma(data: pd.DataFrame, period: int):
        """计算移动平均线"""
        if len(data) < period:
            return pd.Series([data['close'].iloc[-1]] * len(data), index=data.index)
        return data['close'].rolling(window=period).mean()

    @staticmethod
    def calculate_volume_ratio(data: pd.DataFrame, period: int = 20):
        """计算成交量比率"""
        if len(data) < period:
            return pd.Series([1.0] * len(data), index=data.index)
        
        current_volume = data['volume']
        avg_volume = data['volume'].rolling(window=period).mean()
        volume_ratio = current_volume / avg_volume
        return volume_ratio.fillna(1.0)

# ============ 信号检查器 - 简化优化版 ============
class SignalChecker:
    """信号检查器 - 简化优化版"""

    def __init__(self):
        self.thresholds = UltimateConfig.SIGNAL_THRESHOLDS
        self.params = UltimateConfig.OPTIMIZATION_PARAMS

    def check_all_coins(self, coins_data):
        """检查所有币种信号"""
        print(f"\n🔍 开始信号扫描 ({len(coins_data)}个币种)...")

        all_signals = []
        signal_counts = defaultdict(int)

        for symbol, data_dict in coins_data.items():
            try:
                if '15m' not in data_dict:
                    continue

                data_15m = data_dict['15m']
                if len(data_15m) < 30:
                    continue

                # 计算技术指标
                current_price = data_15m['close'].iloc[-1]
                rsi = TechnicalIndicators.calculate_rsi(data_15m, 14).iloc[-1]
                volume_ratio = TechnicalIndicators.calculate_volume_ratio(data_15m, 20).iloc[-1]
                ma20 = TechnicalIndicators.calculate_ma(data_15m, 20).iloc[-1]
                ma50 = TechnicalIndicators.calculate_ma(data_15m, 50).iloc[-1]

                # 检查各种信号
                signals = []

                # 1. 反弹信号
                if rsi < self.params['rsi_bounce_max'] and volume_ratio > self.params['volume_ratio_min']:
                    score = self._calculate_bounce_score(rsi, volume_ratio)
                    if score >= self.thresholds['BOUNCE']:
                        signal = self._create_bounce_signal(symbol, data_15m, current_price, rsi, volume_ratio, ma20, score)
                        signals.append(signal)
                        signal_counts['BOUNCE'] += 1

                # 2. 回调信号
                if rsi > self.params['rsi_callback_min']:
                    recent_high = data_15m['high'].iloc[-30:].max()
                    callback_pct = ((recent_high - current_price) / recent_high) * 100
                    if self.params['callback_pct_min'] <= callback_pct <= self.params['callback_pct_max']:
                        score = self._calculate_callback_score(rsi, volume_ratio, callback_pct)
                        if score >= self.thresholds['CALLBACK']:
                            signal = self._create_callback_signal(symbol, data_15m, current_price, rsi, volume_ratio, recent_high, callback_pct, ma20, score)
                            signals.append(signal)
                            signal_counts['CALLBACK'] += 1

                # 3. 回调确认转强信号
                if 48 <= rsi <= 72 and volume_ratio > 1.2:
                    recent_high = data_15m['high'].iloc[-30:].max()
                    callback_pct = ((recent_high - current_price) / recent_high) * 100
                    if 2 <= callback_pct <= 15:
                        # 检查是否开始反弹
                        recent_3_closes = data_15m['close'].iloc[-3:].values
                        price_increasing = len(recent_3_closes) >= 2 and recent_3_closes[-1] > recent_3_closes[0]
                        
                        if price_increasing and ma20 > ma50 and current_price > ma20:
                            score = self._calculate_callback_confirm_score(rsi, volume_ratio, callback_pct)
                            if score >= self.thresholds['CALLBACK_CONFIRM_K']:
                                signal = self._create_callback_confirm_signal(symbol, data_15m, current_price, rsi, volume_ratio, recent_high, callback_pct, ma20, ma50, score)
                                signals.append(signal)
                                signal_counts['CALLBACK_CONFIRM_K'] += 1

                # 4. 趋势衰竭做空信号
                if rsi > self.params['trend_exhaustion_rsi_min'] and volume_ratio < 1.0:
                    score = self._calculate_trend_exhaustion_score(rsi, volume_ratio)
                    if score >= self.thresholds['TREND_EXHAUSTION']:
                        signal = self._create_trend_exhaustion_signal(symbol, data_15m, current_price, rsi, volume_ratio, ma20, score)
                        signals.append(signal)
                        signal_counts['TREND_EXHAUSTION'] += 1

                # 选择评分最高的信号
                if signals:
                    best_signal = max(signals, key=lambda x: x.get('score', 0))
                    all_signals.append(best_signal)

            except Exception as e:
                continue

        # 打印统计
        self._print_statistics(signal_counts, len(coins_data))
        
        print(f"✅ 扫描完成: 发现 {len(all_signals)} 个交易信号")
        return all_signals

    def _calculate_bounce_score(self, rsi, volume_ratio):
        """计算反弹信号评分"""
        score = 25
        score += (42 - max(20, rsi)) * 1.5
        score += min(30, (volume_ratio - 0.5) * 20)
        return int(score)

    def _calculate_callback_score(self, rsi, volume_ratio, callback_pct):
        """计算回调信号评分"""
        score = 30
        if 55 <= rsi <= 65:
            score += 20
        if 8 <= callback_pct <= 12:
            score += 20
        if 0.8 <= volume_ratio <= 1.5:
            score += 10
        return int(score)

    def _calculate_callback_confirm_score(self, rsi, volume_ratio, callback_pct):
        """计算回调确认转强信号评分"""
        score = 40
        if 50 <= rsi <= 65:
            score += 20
        if volume_ratio > 1.5:
            score += 25
        elif volume_ratio > 1.2:
            score += 15
        if 5 <= callback_pct <= 10:
            score += 15
        return int(score)

    def _calculate_trend_exhaustion_score(self, rsi, volume_ratio):
        """计算趋势衰竭信号评分"""
        score = 30
        score += min(30, (rsi - 65) * 2)
        if volume_ratio < 0.8:
            score += 20
        return int(score)

    def _create_bounce_signal(self, symbol, data, price, rsi, volume_ratio, ma20, score):
        """创建反弹信号"""
        recent_low = data['low'].rolling(20).min().iloc[-1]
        
        entry_main = price * 0.998
        stop_loss = recent_low * 0.98
        take_profit1 = price * 1.03
        take_profit2 = price * 1.06
        
        risk = entry_main - stop_loss
        reward = take_profit2 - entry_main
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        return {
            'symbol': symbol,
            'pattern': 'BOUNCE',
            'direction': 'BUY',
            'rsi': round(float(rsi), 1),
            'volume_ratio': round(volume_ratio, 2),
            'score': int(score),
            'current_price': round(price, 4),
            'signal_time': datetime.now(),
            'reason': f"🟢 <b>超卖反弹机会</b>\n\n"
                     f"• RSI({rsi:.1f})进入超卖区域\n"
                     f"• 成交量放大{volume_ratio:.1f}倍\n"
                     f"• 价格${price:.4f}接近近期低点${recent_low:.4f}\n"
                     f"• 建议在${entry_main:.4f}附近分批买入",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _create_callback_signal(self, symbol, data, price, rsi, volume_ratio, recent_high, callback_pct, ma20, score):
        """创建回调信号"""
        recent_low = data['low'].rolling(20).min().iloc[-1]
        
        entry_main = price * 0.998
        stop_loss = recent_low * 0.98
        take_profit1 = price * 1.04
        take_profit2 = price * 1.08
        
        risk = entry_main - stop_loss
        reward = take_profit2 - entry_main
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        return {
            'symbol': symbol,
            'pattern': 'CALLBACK',
            'direction': 'BUY',
            'rsi': round(float(rsi), 1),
            'volume_ratio': round(volume_ratio, 2),
            'score': int(score),
            'current_price': round(price, 4),
            'signal_time': datetime.now(),
            'reason': f"🔄 <b>健康回调机会</b>\n\n"
                     f"• 从高点${recent_high:.4f}回调{callback_pct:.1f}%\n"
                     f"• RSI({rsi:.1f})回调至理想区域\n"
                     f"• 价格在MA20(${ma20:.4f})上方获得支撑\n"
                     f"• 建议在${entry_main:.4f}附近分批建仓",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _create_callback_confirm_signal(self, symbol, data, price, rsi, volume_ratio, recent_high, callback_pct, ma20, ma50, score):
        """创建回调确认转强信号"""
        recent_low = data['low'].rolling(20).min().iloc[-1]
        
        entry_main = price * 1.002
        stop_loss = recent_low * 0.985
        take_profit1 = recent_high * 1.03
        take_profit2 = recent_high * 1.08
        
        risk = entry_main - stop_loss
        reward = take_profit2 - entry_main
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        return {
            'symbol': symbol,
            'pattern': 'CALLBACK_CONFIRM_K',
            'direction': 'BUY',
            'rsi': round(float(rsi), 1),
            'volume_ratio': round(volume_ratio, 2),
            'score': int(score),
            'current_price': round(price, 4),
            'signal_time': datetime.now(),
            'reason': f"🚀 <b>回调确认转强信号</b>\n\n"
                     f"• 健康回调{callback_pct:.1f}%后确认转强\n"
                     f"• RSI({rsi:.1f})重新进入强势区间\n"
                     f"• 成交量显著放大{volume_ratio:.1f}倍\n"
                     f"• 均线多头排列(MA20>MA50)\n"
                     f"• 趋势可能进入加速阶段\n"
                     f"• 建议在${entry_main:.4f}附近果断买入",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _create_trend_exhaustion_signal(self, symbol, data, price, rsi, volume_ratio, ma20, score):
        """创建趋势衰竭做空信号"""
        recent_high = data['high'].rolling(20).max().iloc[-1]
        
        entry_main = price * 1.002
        stop_loss = recent_high * 1.02
        take_profit1 = price * 0.97
        take_profit2 = price * 0.94
        
        risk = stop_loss - entry_main
        reward = entry_main - take_profit2
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        return {
            'symbol': symbol,
            'pattern': 'TREND_EXHAUSTION',
            'direction': 'SELL',
            'rsi': round(float(rsi), 1),
            'volume_ratio': round(volume_ratio, 2),
            'score': int(score),
            'current_price': round(price, 4),
            'signal_time': datetime.now(),
            'reason': f"🔴 <b>趋势衰竭做空机会</b>\n\n"
                     f"• RSI({rsi:.1f})严重超买\n"
                     f"• 上涨成交量萎缩({volume_ratio:.1f}x)\n"
                     f"• 价格${price:.4f}远离MA20(${ma20:.4f})\n"
                     f"• 存在回调风险\n"
                     f"• 建议在${entry_main:.4f}附近做空",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _print_statistics(self, signal_counts, total_coins):
        """打印信号统计信息"""
        print(f"\n📊 信号检查统计:")
        print(f"   检查币种数: {total_coins}")
        
        total_signals = sum(signal_counts.values())
        if total_signals > 0:
            print(f"   发现信号总数: {total_signals}")
            for pattern, count in sorted(signal_counts.items()):
                percentage = (count / total_signals) * 100
                print(f"   {pattern}: {count}个 ({percentage:.1f}%)")
        else:
            print(f"   未发现任何信号")

# ============ Telegram通知器 - 修复连接 ============
class TelegramNotifier:
    """Telegram通知器 - 修复连接"""

    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        
        print(f"\n🔧 初始化Telegram连接...")
        
        if bot_token and chat_id:
            try:
                # 清理可能的空格
                bot_token = bot_token.strip()
                chat_id = chat_id.strip()
                
                # 直接尝试连接
                self.bot = telebot.TeleBot(bot_token, parse_mode='HTML')
                bot_info = self.bot.get_me()
                print(f"✅ Telegram连接成功: @{bot_info.username}")
            except Exception as e:
                print(f"❌ Telegram连接失败: {str(e)}")
                print(f"💡 尝试使用简单连接方式...")
                
                # 尝试备用连接方式
                try:
                    test_url = f"https://api.telegram.org/bot{bot_token}/getMe"
                    response = requests.get(test_url, timeout=10)
                    if response.status_code == 200:
                        print(f"✅ Telegram API连接成功")
                        self.bot = telebot.TeleBot(bot_token, parse_mode='HTML')
                    else:
                        print(f"❌ Telegram API测试失败: {response.status_code}")
                        self.bot = None
                except Exception as e2:
                    print(f"❌ 备用连接也失败: {str(e2)}")
                    self.bot = None
        else:
            print("⚠️ Telegram配置缺失，禁用通知功能")
            self.bot = None

    def send_signal(self, signal, cooldown_reason=""):
        """发送交易信号"""
        if not self.bot:
            print(f"⚠️ Telegram未启用，跳过信号发送: {signal['symbol']}")
            return False

        try:
            message = self._format_signal_message(signal, cooldown_reason)
            self.bot.send_message(
                self.chat_id,
                message,
                parse_mode='HTML',
                disable_web_page_preview=True
            )
            print(f"✅ Telegram信号发送成功: {signal['symbol']} ({signal['pattern']})")
            return True
        except Exception as e:
            print(f"❌ 发送信号失败 {signal['symbol']}: {str(e)[:100]}")
            return False

    def _format_signal_message(self, signal, cooldown_reason=""):
        """格式化信号消息"""
        direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        pattern_emoji = {
            'BOUNCE': '🔺',
            'BREAKOUT': '⚡',
            'CALLBACK': '🔄',
            'CALLBACK_CONFIRM_K': '🚀',
            'TREND_EXHAUSTION': '📉'
        }.get(signal['pattern'], '💰')
        
        entry = signal['entry_points']

        return f"""
<code>═══════════════════════════</code>
🚀 <b>实盘交易信号</b>
<code>═══════════════════════════</code>

<b>🎯 交易对:</b> {signal['symbol']}/USDT
<b>📊 模式:</b> {signal['pattern']} {pattern_emoji}
<b>📈 方向:</b> {signal['direction']} {direction_emoji}
<b>⭐ 评分:</b> {signal['score']}/100
<b>📉 RSI:</b> {signal['rsi']}
<b>📊 成交量倍数:</b> {signal['volume_ratio']:.1f}x

<b>💰 当前价格:</b> ${signal['current_price']:.4f}
<code>───────────────────────────</code>

<b>🎯 入场点位:</b> ${entry['main_entry']:.4f}
<b>🛑 止损点位:</b> ${entry['stop_loss']:.4f}
<b>🎯 止盈点位:</b> ${entry['take_profit2']:.4f}
<b>⚖️ 风险回报比:</b> {entry['risk_reward']}:1

<code>───────────────────────────</code>
<b>🔍 信号理由:</b>
{signal['reason']}

<code>═══════════════════════════</code>
⏰ {signal['signal_time'].strftime('%H:%M:%S')}
🤖 {UltimateConfig.VERSION}
<code>═══════════════════════════</code>
"""

# ============ 实盘交易系统主类 ============
class UltimateTradingSystem:
    """终极交易系统"""

    def __init__(self):
        print("\n" + "="*60)
        print("🚀 终极智能交易系统 v33.9 - GitHub Actions适配版")
        print("="*60)

        # 初始化组件
        self.data_fetcher = OKXDataFetcher()
        self.cooldown_manager = CooldownManager()
        self.signal_checker = SignalChecker()
        
        # 初始化Telegram
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)

        # 统计数据
        self.cycle_count = 0
        self.total_signals = 0
        self.start_time = datetime.now()

        print(f"\n✅ 系统初始化完成")
        print(f"📡 监控币种: {len(MONITOR_COINS)}个")
        print(f"⏰ 分析间隔: {UltimateConfig.ANALYSIS_INTERVAL}分钟 (由外部调度)")
        print(f"🎯 信号模式: 4种实盘优化策略")
        print(f"🤖 Telegram通知: {'✅ 已启用' if self.telegram.bot else '⚠️ 已禁用'}")
        print("="*60)

    def run_analysis(self):
        """运行单次分析"""
        self.cycle_count += 1
        print(f"\n🔄 第 {self.cycle_count} 次实时分析开始...")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")

        try:
            # 1. 获取市场数据
            coins_data = self.data_fetcher.get_all_coins_data(MONITOR_COINS)
            if not coins_data or len(coins_data) < 10:
                print("❌ 数据获取失败或数据不足，等待重试")
                return []

            print(f"📊 有效数据: {len(coins_data)}/{len(MONITOR_COINS)} 个币种")

            # 2. 信号扫描
            signals = self.signal_checker.check_all_coins(coins_data)

            # 3. 处理并发送信号
            if signals:
                self._process_signals(signals)
            else:
                print("\n📭 本次分析未发现符合条件的交易信号")

            # 4. 显示统计
            elapsed = (datetime.now() - self.start_time).total_seconds() / 60
            print(f"\n📊 系统统计:")
            print(f"   运行周期: {self.cycle_count}次")
            print(f"   总信号数: {self.total_signals}个")
            print(f"   运行时间: {elapsed:.1f}分钟")

            return signals

        except Exception as e:
            print(f"\n❌ 分析过程出错: {str(e)}")
            traceback.print_exc()
            return []

    def _process_signals(self, signals):
        """处理并发送信号"""
        print(f"\n📨 准备发送 {len(signals)} 个交易信号...")

        # 按评分排序
        signals.sort(key=lambda x: x.get('score', 0), reverse=True)

        # 只发送评分最高的前5个信号
        max_signals_to_send = min(5, len(signals))
        top_signals = signals[:max_signals_to_send]

        sent_count = 0
        for i, signal in enumerate(top_signals, 1):
            symbol = signal.get('symbol', 'UNKNOWN')
            pattern = signal.get('pattern', 'UNKNOWN')
            score = signal.get('score', 0)
            
            print(f"\n[{i}] {symbol}: {pattern} ({score}分)")

            # 检查冷却状态
            cooldown_ok, cooldown_reason = self.cooldown_manager.check_cooldown(
                symbol, signal.get('direction', 'BUY')
            )

            if not cooldown_ok:
                print(f"   ⚠️ 冷却阻止: {cooldown_reason}")
                continue

            # 发送到Telegram
            if self.telegram and self.telegram.bot:
                success = self.telegram.send_signal(signal, cooldown_reason)
                if success:
                    # 记录信号
                    self.cooldown_manager.record_signal(
                        symbol, 
                        signal.get('direction', 'BUY'),
                        pattern,
                        score
                    )
                    self.total_signals += 1
                    sent_count += 1
                    time.sleep(2)  # 避免发送过快被限制
                else:
                    print(f"   ⚠️ 信号发送失败，跳过")
            else:
                print(f"   ⚠️ Telegram未启用，跳过发送")

        print(f"\n✅ 本次成功发送 {sent_count} 个交易信号")

# ============ 主程序入口 ============
def main():
    """主函数 - 单次运行模式"""
    print("="*60)
    print("🤖 终极智能交易系统 v33.9 - GitHub Actions适配版")
    print("="*60)
    print(f"📅 版本: {UltimateConfig.VERSION}")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 监控币种: {len(MONITOR_COINS)}个")
    print(f"🎯 信号模式: 4种优化策略（包含CALLBACK_CONFIRM_K）")
    print(f"⏰ 分析间隔: {UltimateConfig.ANALYSIS_INTERVAL}分钟 (由外部调度)")
    print(f"🤖 Telegram通知: {'已配置' if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID else '未配置'}")
    
    print("="*60)
    print("🚀 系统特性:")
    print("   1. 实盘信号检查器 - 门槛优化")
    print("   2. 回调确认转强判断 - CALLBACK_CONFIRM_K")
    print("   3. Telegram实时通知 - 从环境变量读取")
    print("   4. 冷却管理优化 - 防止重复信号")
    print("   5. 单次运行模式 - 适配GitHub Actions")
    print("="*60)

    try:
        # 创建系统实例
        system = UltimateTradingSystem()

        # 运行一次分析
        print("\n🎯 开始实时分析...")
        signals = system.run_analysis()

        if signals:
            print(f"\n✅ 分析完成！发现 {len(signals)} 个交易信号")
        else:
            print("\n📊 本次分析未发现信号")

        print("\n🏁 单次运行结束，退出。")
        return 0

    except KeyboardInterrupt:
        print("\n\n🛑 系统被用户停止")
        return 1
    except Exception as e:
        print(f"\n❌ 系统运行失败: {e}")
        traceback.print_exc()
        return 1

# ============ 立即启动 ============
if __name__ == "__main__":
    sys.exit(main())