#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极智能交易系统 v34.6 正式版
改进：趋势方向增强判断（EMA斜率）+ ATR止损百分比限制 + 冷却区分方向
适用于 GitHub Actions 定时运行，单次分析后退出
"""

import os
import sys
import time
import pickle
import atexit
import requests
import traceback
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd
import numpy as np
import telebot

# ============ 配置 ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("⚠️ 警告：未设置 TELEGRAM_BOT_TOKEN 或 TELEGRAM_CHAT_ID，Telegram 通知已禁用")
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

OKX_API_BASE_URL = "https://www.okx.com"
OKX_CANDLE_INTERVAL = ["15m", "1H"]
OKX_CANDLE_LIMIT = 100

# 监控币种列表
MONITOR_COINS = [
    'BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'AVAX', 'DOT',
    'DOGE', 'LTC', 'UNI', 'LINK', 'ATOM', 'XLM', 'ALGO',
    'FIL', 'TRX', 'ETC', 'XTZ', 'AAVE', 'COMP', 'YFI',
    'SUSHI', 'SNX', 'CRV', '1INCH', 'NEAR', 'GRT', 'SAND',
    'MANA', 'ENJ', 'CHZ', 'BAT', 'ZIL', 'ONE', 'IOTA',
    'DASH', 'ZEC', 'EGLD', 'CRO', 'KSM', 'DYDX', 'JUP',
    'STORJ', 'SKL', 'WLD',
    'ARB', 'OP', 'LDO', 'APT', 'SUI', 'SEI', 'INJ',
    'FET', 'THETA', 'AR', 'ENS', 'PEPE', 'SHIB',
    'APE', 'LIT', 'GALA', 'IMX', 'AXS'
]

print(f"📊 监控币种列表: {len(MONITOR_COINS)} 个币种")

# ============ 配置类 ============
class UltimateConfig:
    VERSION = "34.6-正式版（趋势增强+ATR上限+方向冷却）"
    MAX_SIGNALS_TO_SEND = 3
    TELEGRAM_RETRY = 3
    TELEGRAM_RETRY_DELAY = 1
    
    COOLDOWN_CONFIG = {
        'same_coin_cooldown': 90,          # 默认冷却（当没有动态冷却是使用）
        'same_direction_cooldown': 45,      # 同一方向额外冷却（可选，但我们现在按方向独立冷却，此配置可保留）
        'max_signals_per_coin_per_day': 5,
        'enable_cooldown': True
    }
    
    SIGNAL_THRESHOLDS = {
        'BOUNCE': 32,
        'BREAKOUT': 25,
        'TREND_EXHAUSTION': 35,
        'CALLBACK': 30,
        'CONFIRMATION_K': 40,
        'CALLBACK_CONFIRM_K': 45
    }
    
    OPTIMIZATION_PARAMS = {
        'volume_ratio_min': 0.7,
        'rsi_bounce_max': 45,
        'rsi_callback_min': 48,
        'callback_pct_min': 2,
        'callback_pct_max': 25,
        'trend_exhaustion_rsi_min': 65
    }
    
    OKX_CONFIG = {
        'base_url': OKX_API_BASE_URL,
        'candle_endpoint': '/api/v5/market/candles',
        'intervals': OKX_CANDLE_INTERVAL,
        'limit': OKX_CANDLE_LIMIT,
        'interval_limits': {'15m': 100, '1H': 60},
        'rate_limit': 20,
        'retry_times': 2,
        'timeout': 15
    }
    
    # CONFIRMATION_K 权重配置
    CONFIRMATION_K_WEIGHTS = {
        'structure': 0.40,
        'momentum': 0.25,
        'volume': 0.15,
        'trend': 0.20
    }
    
    # 趋势模式阈值
    TREND_MODES = {
        'RANGE': 15,
        'TRANSITION': 25,
    }
    
    # 趋势方向匹配得分
    TREND_MATCH_SCORE = 1.0      # 顺趋势
    TREND_MISMATCH_SCORE = 0.2   # 逆趋势
    
    # 背离复合强度系数
    DIVERGENCE_WEIGHTS = {
        'rsi': 0.6,
        'price': 0.4
    }
    PRICE_STRENGTH_FACTOR = 25   # 4%回调 => 强度1.0
    
    MACD_EXHAUSTION_FACTOR = 0.6
    MACD_EXHAUSTION_LOOKBACK = 3
    
    COOLDOWN_DYNAMIC = {
        (80, 101): 60,
        (60, 80): 90,
        (0, 60): 120
    }
    
    ATR_STOP_MULTIPLIER = 1.5
    MAX_STOP_PERCENT = 0.06       # 最大止损距离为价格的6%，防止ATR过大导致止损过远

    # 趋势模式与信号类型的匹配规则
    TREND_SIGNAL_ALLOW = {
        'TREND': ['CONFIRMATION_K', 'TREND_EXHAUSTION', 'CALLBACK_CONFIRM_K'],
        'TRANSITION': ['CONFIRMATION_K', 'CALLBACK', 'BOUNCE', 'TREND_EXHAUSTION', 'CALLBACK_CONFIRM_K'],
        'RANGE': ['BOUNCE', 'CALLBACK', 'CONFIRMATION_K']
    }

# ============ 冷却管理器（按方向独立冷却）============
class CooldownManager:
    def __init__(self):
        self.config = UltimateConfig.COOLDOWN_CONFIG
        self.cooldown_db = {}          # 键为 f"{symbol}_{direction}"
        self.signal_history = defaultdict(list)
        self.cooldown_file = 'cooldown_state.pkl'
        self.load_state()
        atexit.register(self.save_state)

    def load_state(self):
        try:
            if os.path.exists(self.cooldown_file):
                with open(self.cooldown_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cooldown_db = data.get('cooldown_db', {})
                    # 兼容旧版：如果键不含方向，尝试转换（可选，但新版我们强制使用带方向的键）
                    self.signal_history = defaultdict(list, data.get('signal_history', {}))
                print(f"✅ 冷却状态已加载: {len(self.cooldown_db)}个记录")
        except Exception as e:
            print(f"❌ 加载冷却状态失败: {e}")

    def save_state(self):
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

    def _get_key(self, symbol: str, direction: str) -> str:
        return f"{symbol}_{direction}"

    def check_cooldown(self, symbol: str, direction: str) -> Tuple[bool, str]:
        if not self.config['enable_cooldown']:
            return True, ""
        now = datetime.now()
        key = self._get_key(symbol, direction)
        if key in self.cooldown_db:
            last_signal = self.cooldown_db[key]
            last_time = last_signal['time']
            cooldown_minutes = last_signal.get('cooldown_minutes', self.config['same_coin_cooldown'])
            elapsed = (now - last_time).total_seconds() / 60
            if elapsed < cooldown_minutes:
                remaining = cooldown_minutes - elapsed
                return False, f"同币种同方向冷却中 ({remaining:.1f}分钟)"
        return True, ""

    def record_signal(self, symbol: str, direction: str, pattern: str, score: int):
        now = datetime.now()
        key = self._get_key(symbol, direction)
        cooldown_minutes = self.config['same_coin_cooldown']
        for (low, high), minutes in UltimateConfig.COOLDOWN_DYNAMIC.items():
            if low <= score < high:
                cooldown_minutes = minutes
                break
        self.cooldown_db[key] = {
            'time': now,
            'symbol': symbol,
            'direction': direction,
            'pattern': pattern,
            'score': score,
            'cooldown_minutes': cooldown_minutes
        }
        self.signal_history[symbol].append({
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'direction': direction,
            'pattern': pattern,
            'score': score
        })

# ============ OKX 数据获取器 ============
class OKXDataFetcher:
    def __init__(self):
        self.config = UltimateConfig.OKX_CONFIG
        self.base_url = self.config['base_url']
        self.endpoint = self.config['candle_endpoint']
        self.intervals = self.config['intervals']
        self.default_limit = self.config['limit']
        self.interval_limits = self.config.get('interval_limits', {})
        self.retry_times = self.config['retry_times']
        self.timeout = self.config['timeout']
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 120

    def get_candles(self, symbol: str, interval: str):
        cache_key = f"{symbol}_{interval}"
        current_time = time.time()
        if cache_key in self.cache and current_time - self.cache_time.get(cache_key, 0) < self.cache_duration:
            return self.cache[cache_key]

        inst_id = f"{symbol}-USDT"
        limit = self.interval_limits.get(interval, self.default_limit)
        params = {'instId': inst_id, 'bar': interval, 'limit': limit}
        url = f"{self.base_url}{self.endpoint}"

        for retry in range(self.retry_times):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    if data['code'] == '0' and len(data['data']) > 0:
                        candles = data['data']
                        df = pd.DataFrame(candles)
                        if len(df.columns) >= 6:
                            df = df.iloc[:, :6]
                            df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            df.set_index('timestamp', inplace=True)
                            df.sort_index(inplace=True)
                            self.cache[cache_key] = df
                            self.cache_time[cache_key] = current_time
                            return df
            except Exception as e:
                if retry < self.retry_times - 1:
                    time.sleep(1)
        return None

    def get_all_coins_data(self, symbols: List[str]):
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
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14):
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
        if len(data) < period:
            return pd.Series([data['close'].iloc[-1]] * len(data), index=data.index)
        return data['close'].rolling(window=period).mean()

    @staticmethod
    def calculate_ema(data: pd.DataFrame, period: int):
        return data['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_volume_ratio(data: pd.DataFrame, period: int = 20):
        if len(data) < period:
            return pd.Series([1.0] * len(data), index=data.index)
        current_volume = data['volume']
        avg_volume = data['volume'].rolling(window=period).mean()
        volume_ratio = current_volume / avg_volume
        return volume_ratio.fillna(1.0)

    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast_period=12, slow_period=26, signal_period=9):
        close = data['close']
        exp1 = close.ewm(span=fast_period, adjust=False).mean()
        exp2 = close.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        histogram = macd - signal
        return pd.DataFrame({'macd': macd, 'signal': signal, 'histogram': histogram}, index=data.index)

    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14):
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        up_move = high - high.shift()
        down_move = low.shift() - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.rolling(window=period).mean()
        return adx.fillna(25)

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14):
        high = data['high']
        low = data['low']
        close = data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr.fillna(method='bfill').fillna(0)

# ============ 信号检查器 ============
class SignalChecker:
    def __init__(self):
        self.thresholds = UltimateConfig.SIGNAL_THRESHOLDS
        self.params = UltimateConfig.OPTIMIZATION_PARAMS

    # ---------- 摆动点检测 ----------
    def _find_swing_highs_lows(self, data: pd.DataFrame, window: int = 5):
        highs = data['high'].values
        lows = data['low'].values
        swing_highs = []
        swing_lows = []
        for i in range(window, len(data) - window):
            if highs[i] == max(highs[i-window:i+window+1]):
                swing_highs.append(i)
            if lows[i] == min(lows[i-window:i+window+1]):
                swing_lows.append(i)
        return swing_highs, swing_lows

    # ---------- 复合背离检测 ----------
    def _detect_rsi_divergence_swing(self, data: pd.DataFrame, rsi_series: pd.Series, lookback=30) -> tuple:
        if len(data) < lookback:
            return None, 0.0
        
        swing_highs, swing_lows = self._find_swing_highs_lows(data.iloc[-lookback:], window=3)
        base_idx = len(data) - lookback
        swing_highs = [base_idx + i for i in swing_highs]
        swing_lows = [base_idx + i for i in swing_lows]
        
        w_rsi = UltimateConfig.DIVERGENCE_WEIGHTS['rsi']
        w_price = UltimateConfig.DIVERGENCE_WEIGHTS['price']
        price_factor = UltimateConfig.PRICE_STRENGTH_FACTOR
        
        if len(swing_lows) >= 2:
            last_low_idx = swing_lows[-1]
            prev_low_idx = swing_lows[-2]
            last_low_price = data['low'].iloc[last_low_idx]
            prev_low_price = data['low'].iloc[prev_low_idx]
            last_rsi = rsi_series.iloc[last_low_idx]
            prev_rsi = rsi_series.iloc[prev_low_idx]
            
            if last_low_price < prev_low_price and last_rsi > prev_rsi:
                rsi_diff = min((last_rsi - prev_rsi) / 20, 1.0)
                price_drop_pct = (prev_low_price - last_low_price) / prev_low_price
                price_strength = min(price_drop_pct * price_factor, 1.0)
                strength = rsi_diff * w_rsi + price_strength * w_price
                return 'bullish', strength
        
        if len(swing_highs) >= 2:
            last_high_idx = swing_highs[-1]
            prev_high_idx = swing_highs[-2]
            last_high_price = data['high'].iloc[last_high_idx]
            prev_high_price = data['high'].iloc[prev_high_idx]
            last_rsi = rsi_series.iloc[last_high_idx]
            prev_rsi = rsi_series.iloc[prev_high_idx]
            
            if last_high_price > prev_high_price and last_rsi < prev_rsi:
                rsi_diff = min((prev_rsi - last_rsi) / 20, 1.0)
                price_rise_pct = (last_high_price - prev_high_price) / prev_high_price
                price_strength = min(price_rise_pct * price_factor, 1.0)
                strength = rsi_diff * w_rsi + price_strength * w_price
                return 'bearish', strength
        
        return None, 0.0

    # ---------- MACD衰竭检测 ----------
    def _detect_macd_hist_decline_adv(self, hist_series: pd.Series, direction: str, periods=3) -> tuple:
        if len(hist_series) < periods:
            return False, 0.0
        
        recent = hist_series.iloc[-periods:].values
        factor = UltimateConfig.MACD_EXHAUSTION_FACTOR
        
        if direction == 'BUY':
            if all(h > 0 for h in recent) and all(recent[i] < recent[i-1] for i in range(1, len(recent))):
                if abs(recent[-1]) < abs(recent[0]) * factor:
                    decline_ratio = (recent[0] - recent[-1]) / (recent[0] + 1e-6)
                    strength = min(decline_ratio, 1.0)
                    return True, strength
        else:
            if all(h < 0 for h in recent) and all(recent[i] > recent[i-1] for i in range(1, len(recent))):
                if abs(recent[-1]) < abs(recent[0]) * factor:
                    increase_ratio = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-6)
                    strength = min(increase_ratio, 1.0)
                    return True, strength
        
        return False, 0.0

    # ---------- 趋势模式 ----------
    def _get_trend_mode(self, data: pd.DataFrame) -> str:
        adx = TechnicalIndicators.calculate_adx(data).iloc[-1]
        if adx <= UltimateConfig.TREND_MODES['RANGE']:
            return 'RANGE'
        elif adx <= UltimateConfig.TREND_MODES['TRANSITION']:
            return 'TRANSITION'
        else:
            return 'TREND'

    # ---------- 趋势方向匹配评分（改进版：考虑EMA斜率） ----------
    def _get_trend_score(self, data: pd.DataFrame, signal_direction: str, trend_mode: str) -> float:
        """
        根据趋势模式和信号方向返回趋势维度的得分（0~1）
        在TREND模式下，使用EMA20斜率判断趋势方向，避免价格刺破EMA的假突破
        """
        if trend_mode == 'RANGE':
            return 0.3
        elif trend_mode == 'TRANSITION':
            return 0.6
        
        # TREND模式：需要判断方向是否匹配
        ema20 = TechnicalIndicators.calculate_ema(data, 20)
        ema20_current = ema20.iloc[-1]
        ema20_prev = ema20.iloc[-2] if len(ema20) > 1 else ema20_current
        
        # 趋势向上：EMA20上升且当前价格在EMA20之上（可选，这里主要用EMA20斜率）
        trend_up = ema20_current > ema20_prev
        trend_down = ema20_current < ema20_prev
        
        # 信号方向与趋势方向匹配判断
        if (signal_direction == 'BUY' and trend_up) or (signal_direction == 'SELL' and trend_down):
            return UltimateConfig.TREND_MATCH_SCORE
        else:
            return UltimateConfig.TREND_MISMATCH_SCORE

    # ---------- 趋势模式过滤 ----------
    def _is_signal_allowed(self, pattern: str, trend_mode: str) -> bool:
        allow_map = UltimateConfig.TREND_SIGNAL_ALLOW
        if trend_mode in allow_map:
            return pattern in allow_map[trend_mode]
        return True

    # ---------- 吞没形态检测 ----------
    def _detect_engulfing(self, data: pd.DataFrame) -> tuple:
        if len(data) < 2:
            return '', 0.0
        prev = data.iloc[-2]
        curr = data.iloc[-1]
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        prev_open, prev_close = prev['open'], prev['close']
        curr_open, curr_close = curr['open'], curr['close']

        if (prev_close < prev_open) and (curr_close > curr_open) and \
           curr_open < prev_close and curr_close > prev_open:
            strength = min(curr_body / prev_body, 2.0) if prev_body > 0 else 1.0
            return 'BUY', strength
        if (prev_close > prev_open) and (curr_close < curr_open) and \
           curr_open > prev_close and curr_close < prev_open:
            strength = min(curr_body / prev_body, 2.0) if prev_body > 0 else 1.0
            return 'SELL', strength
        return '', 0.0

    # ---------- 增强版 CONFIRMATION_K 评分 ----------
    def _calculate_confirmation_k_score_advanced(self, direction: str, rsi: float, volume_ratio: float,
                                                 engulf_strength: float, div_info: tuple, decline_info: tuple,
                                                 data: pd.DataFrame, macd_df: pd.DataFrame) -> int:
        # 1. 结构强度 (40%)
        div_type, div_str = div_info
        structure = 0.0
        structure += engulf_strength * 0.6
        if div_type == direction.lower():
            structure += div_str * 0.4
        structure = min(structure, 1.0)
        
        # 2. 动能确认 (25%)
        momentum = 0.0
        is_fading, fade_str = decline_info
        if is_fading:
            momentum += fade_str * 0.7
        if direction == 'BUY':
            if rsi < 60:
                rsi_score = (60 - rsi) / 30
            else:
                rsi_score = 0
        else:
            if rsi > 40:
                rsi_score = (rsi - 40) / 30
            else:
                rsi_score = 0
        momentum += min(rsi_score, 1.0) * 0.3
        momentum = min(momentum, 1.0)
        
        # 3. 量能确认 (15%)
        volume = min(volume_ratio / 2.0, 1.0)
        
        # 4. 趋势匹配 (20%) —— 使用改进的趋势方向评分
        trend_mode = self._get_trend_mode(data)
        trend_score = self._get_trend_score(data, direction, trend_mode)
        
        w = UltimateConfig.CONFIRMATION_K_WEIGHTS
        total = (structure * w['structure'] +
                 momentum * w['momentum'] +
                 volume * w['volume'] +
                 trend_score * w['trend']) * 100
        
        return int(total)

    # ---------- 主扫描函数 ----------
    def check_all_coins(self, coins_data):
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

                current_price = data_15m['close'].iloc[-1]
                rsi = TechnicalIndicators.calculate_rsi(data_15m, 14).iloc[-1]
                volume_ratio = TechnicalIndicators.calculate_volume_ratio(data_15m, 20).iloc[-1]
                ma20 = TechnicalIndicators.calculate_ma(data_15m, 20).iloc[-1]
                ma50 = TechnicalIndicators.calculate_ma(data_15m, 50).iloc[-1]

                trend_mode = self._get_trend_mode(data_15m)
                signals = []

                # 反弹信号
                if rsi < self.params['rsi_bounce_max'] and volume_ratio > self.params['volume_ratio_min']:
                    if self._is_signal_allowed('BOUNCE', trend_mode):
                        score = self._calculate_bounce_score(rsi, volume_ratio)
                        if score >= self.thresholds['BOUNCE']:
                            signals.append(self._create_bounce_signal(symbol, data_15m, current_price, rsi, volume_ratio, ma20, score))
                            signal_counts['BOUNCE'] += 1

                # 回调信号
                if rsi > self.params['rsi_callback_min']:
                    if self._is_signal_allowed('CALLBACK', trend_mode):
                        recent_high = data_15m['high'].iloc[-30:].max()
                        callback_pct = ((recent_high - current_price) / recent_high) * 100
                        if self.params['callback_pct_min'] <= callback_pct <= self.params['callback_pct_max']:
                            score = self._calculate_callback_score(rsi, volume_ratio, callback_pct)
                            if score >= self.thresholds['CALLBACK']:
                                signals.append(self._create_callback_signal(symbol, data_15m, current_price, rsi, volume_ratio, recent_high, callback_pct, ma20, score))
                                signal_counts['CALLBACK'] += 1

                # 回调确认转强信号
                if 48 <= rsi <= 72 and volume_ratio > 1.2:
                    if self._is_signal_allowed('CALLBACK_CONFIRM_K', trend_mode):
                        recent_high = data_15m['high'].iloc[-30:].max()
                        callback_pct = ((recent_high - current_price) / recent_high) * 100
                        if 2 <= callback_pct <= 15:
                            recent_3_closes = data_15m['close'].iloc[-3:].values
                            price_increasing = len(recent_3_closes) >= 2 and recent_3_closes[-1] > recent_3_closes[0]
                            if price_increasing and ma20 > ma50 and current_price > ma20:
                                score = self._calculate_callback_confirm_score(rsi, volume_ratio, callback_pct)
                                if score >= self.thresholds['CALLBACK_CONFIRM_K']:
                                    signals.append(self._create_callback_confirm_signal(symbol, data_15m, current_price, rsi, volume_ratio, recent_high, callback_pct, ma20, ma50, score))
                                    signal_counts['CALLBACK_CONFIRM_K'] += 1

                # 趋势衰竭做空信号
                if rsi > self.params['trend_exhaustion_rsi_min'] and volume_ratio < 1.0:
                    if self._is_signal_allowed('TREND_EXHAUSTION', trend_mode):
                        score = self._calculate_trend_exhaustion_score(rsi, volume_ratio)
                        if score >= self.thresholds['TREND_EXHAUSTION']:
                            signals.append(self._create_trend_exhaustion_signal(symbol, data_15m, current_price, rsi, volume_ratio, ma20, score))
                            signal_counts['TREND_EXHAUSTION'] += 1

                # 吞没形态信号 CONFIRMATION_K
                engulf_dir, engulf_strength = self._detect_engulfing(data_15m)
                if engulf_dir and self._is_signal_allowed('CONFIRMATION_K', trend_mode):
                    rsi_series = TechnicalIndicators.calculate_rsi(data_15m, 14)
                    macd_df = TechnicalIndicators.calculate_macd(data_15m)
                    hist_series = macd_df['histogram']
                    
                    div_info = self._detect_rsi_divergence_swing(data_15m, rsi_series, lookback=30)
                    decline_info = self._detect_macd_hist_decline_adv(hist_series, engulf_dir, periods=3)
                    
                    score = self._calculate_confirmation_k_score_advanced(
                        engulf_dir, rsi, volume_ratio, engulf_strength,
                        div_info, decline_info, data_15m, macd_df
                    )
                    
                    if score >= self.thresholds['CONFIRMATION_K']:
                        signals.append(self._create_confirmation_k_signal_advanced(
                            symbol, data_15m, current_price, rsi, volume_ratio,
                            ma20, ma50, engulf_dir, engulf_strength,
                            div_info, decline_info, score
                        ))
                        signal_counts['CONFIRMATION_K'] += 1

                if signals:
                    best_signal = max(signals, key=lambda x: x.get('score', 0))
                    all_signals.append(best_signal)

            except Exception as e:
                continue

        self._print_statistics(signal_counts, len(coins_data))
        print(f"✅ 扫描完成: 发现 {len(all_signals)} 个交易信号")
        return all_signals

    # ---------- 其他评分函数 ----------
    def _calculate_bounce_score(self, rsi, volume_ratio):
        score = 25
        score += (42 - max(20, rsi)) * 1.5
        score += min(30, (volume_ratio - 0.5) * 20)
        return int(score)

    def _calculate_callback_score(self, rsi, volume_ratio, callback_pct):
        score = 30
        if 55 <= rsi <= 65:
            score += 20
        if 8 <= callback_pct <= 12:
            score += 20
        if 0.8 <= volume_ratio <= 1.5:
            score += 10
        return int(score)

    def _calculate_callback_confirm_score(self, rsi, volume_ratio, callback_pct):
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
        score = 30
        score += min(30, (rsi - 65) * 2)
        if volume_ratio < 0.8:
            score += 20
        return int(score)

    # ---------- 信号创建函数（含ATR止损+百分比上限） ----------
    def _create_confirmation_k_signal_advanced(self, symbol, data, price, rsi, volume_ratio,
                                               ma20, ma50, direction, engulf_strength,
                                               div_info, decline_info, score):
        atr = TechnicalIndicators.calculate_atr(data).iloc[-1]
        atr_mult = UltimateConfig.ATR_STOP_MULTIPLIER
        max_stop_pct = UltimateConfig.MAX_STOP_PERCENT
        
        if direction == 'BUY':
            recent_low = data['low'].rolling(10).min().iloc[-1]
            entry_main = price * 1.002
            # ATR止损
            stop_loss_candidate1 = recent_low * 0.985
            stop_loss_candidate2 = price - atr * atr_mult
            # 取较大者（离现价更远）但不超过最大止损百分比
            stop_loss = max(stop_loss_candidate1, stop_loss_candidate2)
            # 限制止损距离不超过 max_stop_pct * price
            min_stop = price * (1 - max_stop_pct)
            stop_loss = max(stop_loss, min_stop)
            take_profit1 = price * 1.04
            take_profit2 = price * 1.08
            risk = entry_main - stop_loss
            reward = take_profit2 - entry_main

            div_text = f"• 看涨背离强度: {div_info[1]:.2f}\n" if div_info[0] == 'bullish' else ""
            decl_text = f"• 多头衰竭强度: {decline_info[1]:.2f}\n" if decline_info[0] else ""
            reason = (
                f"🟢 <b>看涨吞没形态确认</b>\n\n"
                f"• 吞没强度: {engulf_strength:.2f}\n"
                f"• 成交量{volume_ratio:.1f}倍\n"
                f"• RSI({rsi:.1f})\n"
                f"{div_text}{decl_text}"
                f"• 建议在${entry_main:.4f}附近买入"
            )
        else:
            recent_high = data['high'].rolling(10).max().iloc[-1]
            entry_main = price * 0.998
            stop_loss_candidate1 = recent_high * 1.02
            stop_loss_candidate2 = price + atr * atr_mult
            # 取较小者（离现价更远）但不超过最大止损百分比
            stop_loss = min(stop_loss_candidate1, stop_loss_candidate2)
            max_stop = price * (1 + max_stop_pct)
            stop_loss = min(stop_loss, max_stop)
            take_profit1 = price * 0.96
            take_profit2 = price * 0.93
            risk = stop_loss - entry_main
            reward = entry_main - take_profit2

            div_text = f"• 看跌背离强度: {div_info[1]:.2f}\n" if div_info[0] == 'bearish' else ""
            decl_text = f"• 空头衰竭强度: {decline_info[1]:.2f}\n" if decline_info[0] else ""
            reason = (
                f"🔴 <b>看跌吞没形态确认</b>\n\n"
                f"• 吞没强度: {engulf_strength:.2f}\n"
                f"• 成交量{volume_ratio:.1f}倍\n"
                f"• RSI({rsi:.1f})\n"
                f"{div_text}{decl_text}"
                f"• 建议在${entry_main:.4f}附近做空"
            )

        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        return {
            'symbol': symbol,
            'pattern': 'CONFIRMATION_K',
            'direction': direction,
            'rsi': round(float(rsi), 1),
            'volume_ratio': round(volume_ratio, 2),
            'score': int(score),
            'current_price': round(price, 4),
            'signal_time': datetime.now(),
            'reason': reason,
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _create_bounce_signal(self, symbol, data, price, rsi, volume_ratio, ma20, score):
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
            'reason': f"🟢 <b>超卖反弹机会</b>\n\n• RSI({rsi:.1f})进入超卖\n• 成交量放大{volume_ratio:.1f}倍\n• 价格${price:.4f}接近低点${recent_low:.4f}\n• 建议在${entry_main:.4f}附近买入",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _create_callback_signal(self, symbol, data, price, rsi, volume_ratio, recent_high, callback_pct, ma20, score):
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
            'reason': f"🔄 <b>健康回调机会</b>\n\n• 从高点${recent_high:.4f}回调{callback_pct:.1f}%\n• RSI({rsi:.1f})理想\n• 价格在MA20(${ma20:.4f})上方\n• 建议在${entry_main:.4f}附近建仓",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _create_callback_confirm_signal(self, symbol, data, price, rsi, volume_ratio, recent_high, callback_pct, ma20, ma50, score):
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
            'reason': f"🚀 <b>回调确认转强</b>\n\n• 回调{callback_pct:.1f}%后转强\n• RSI({rsi:.1f})强势\n• 成交量{volume_ratio:.1f}倍\n• 均线多头\n• 建议${entry_main:.4f}买入",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _create_trend_exhaustion_signal(self, symbol, data, price, rsi, volume_ratio, ma20, score):
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
            'reason': f"🔴 <b>趋势衰竭做空</b>\n\n• RSI({rsi:.1f})超买\n• 成交量萎缩{volume_ratio:.1f}x\n• 价格远离MA20\n• 建议${entry_main:.4f}做空",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            }
        }

    def _print_statistics(self, signal_counts, total_coins):
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

# ============ Telegram 通知器 ============
class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        if bot_token and chat_id:
            try:
                self.bot = telebot.TeleBot(bot_token, parse_mode='HTML')
                bot_info = self.bot.get_me()
                print(f"✅ Telegram 连接成功: @{bot_info.username}")
            except Exception as e:
                print(f"❌ Telegram 连接失败: {e}")
                self.bot = None
        else:
            print("⚠️ Telegram 未配置，通知功能已禁用")

    def send_signal(self, signal, cooldown_reason=""):
        if not self.bot:
            print(f"\n📨 [模拟发送] {signal['symbol']} - {signal['pattern']} ({signal['score']}分)")
            return True

        message = self._format_signal_message(signal, cooldown_reason)
        for attempt in range(1, UltimateConfig.TELEGRAM_RETRY + 1):
            try:
                self.bot.send_message(
                    self.chat_id,
                    message,
                    parse_mode='HTML',
                    disable_web_page_preview=True
                )
                print(f"✅ Telegram 信号发送成功: {signal['symbol']} ({signal['pattern']})")
                return True
            except Exception as e:
                print(f"❌ 发送失败 (尝试 {attempt}/{UltimateConfig.TELEGRAM_RETRY}): {signal['symbol']} - {str(e)[:100]}")
                if attempt < UltimateConfig.TELEGRAM_RETRY:
                    time.sleep(UltimateConfig.TELEGRAM_RETRY_DELAY)
                else:
                    print(f"   ⚠️ 信号 {signal['symbol']} 最终发送失败")
        return False

    def _format_signal_message(self, signal, cooldown_reason=""):
        direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        pattern_emoji = {
            'BOUNCE': '🔺', 'BREAKOUT': '⚡', 'CALLBACK': '🔄',
            'CALLBACK_CONFIRM_K': '🚀', 'CONFIRMATION_K': '🔰', 'TREND_EXHAUSTION': '📉'
        }.get(signal['pattern'], '💰')
        entry = signal['entry_points']
        return f"""
        
 <b>🚀实盘交易信号</b>

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

# ============ 交易系统主类 ============
class UltimateTradingSystem:
    def __init__(self):
        print("\n" + "="*60)
        print(f"🚀 终极智能交易系统 {UltimateConfig.VERSION}")
        print("="*60)
        self.data_fetcher = OKXDataFetcher()
        self.cooldown_manager = CooldownManager()
        self.signal_checker = SignalChecker()
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        self.cycle_count = 0
        self.total_signals = 0
        self.start_time = datetime.now()
        print(f"\n✅ 系统初始化完成")
        print(f"📡 监控币种: {len(MONITOR_COINS)}个")
        print(f"🤖 Telegram 通知: {'✅ 已启用' if self.telegram.bot else '⚠️ 已禁用'}")
        print("="*60)

    def run_analysis(self):
        self.cycle_count += 1
        print(f"\n🔄 第 {self.cycle_count} 次实时分析开始...")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")

        try:
            coins_data = self.data_fetcher.get_all_coins_data(MONITOR_COINS)
            if not coins_data or len(coins_data) < 10:
                print("❌ 数据获取失败或数据不足，等待重试")
                return []

            print(f"📊 有效数据: {len(coins_data)}/{len(MONITOR_COINS)} 个币种")
            signals = self.signal_checker.check_all_coins(coins_data)

            if signals:
                self._process_signals(signals)
            else:
                print("\n📭 本次分析未发现符合条件的交易信号")

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
        print(f"\n📨 准备发送 {len(signals)} 个交易信号...")
        signals.sort(key=lambda x: x.get('score', 0), reverse=True)
        max_to_send = min(UltimateConfig.MAX_SIGNALS_TO_SEND, len(signals))
        top_signals = signals[:max_to_send]

        sent_count = 0
        for i, signal in enumerate(top_signals, 1):
            symbol = signal.get('symbol', 'UNKNOWN')
            pattern = signal.get('pattern', 'UNKNOWN')
            score = signal.get('score', 0)
            direction = signal.get('direction', 'BUY')
            print(f"\n[{i}] {symbol} {direction}: {pattern} ({score}分)")

            cooldown_ok, cooldown_reason = self.cooldown_manager.check_cooldown(symbol, direction)
            if not cooldown_ok:
                print(f"   ⚠️ 冷却阻止: {cooldown_reason}")
                continue

            success = self.telegram.send_signal(signal, cooldown_reason)
            if success:
                self.cooldown_manager.record_signal(symbol, direction, pattern, score)
                self.total_signals += 1
                sent_count += 1
                time.sleep(2)
            else:
                print(f"   ⚠️ 信号最终发送失败，跳过记录冷却")

        print(f"\n✅ 本次成功发送 {sent_count} 个交易信号")

# ============ 主程序入口 ============
def main():
    print("="*60)
    print("🤖 终极智能交易系统 - GitHub Actions 优化版")
    print("="*60)
    print(f"📅 版本: {UltimateConfig.VERSION}")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 监控币种: {len(MONITOR_COINS)}个")
    print(f"🎯 信号模式: 5种策略 + 增强型吞没(复合背离/方向MACD/动态冷却/ATR/趋势增强/方向冷却)")
    print("="*60)

    try:
        system = UltimateTradingSystem()
        print("\n🎯 运行实时分析...")
        signals = system.run_analysis()

        if signals:
            print(f"\n✅ 分析完成！发现 {len(signals)} 个交易信号")
        else:
            print("\n📊 本次分析未发现信号")

        print("\n🏁 单次运行完成，退出。")
        sys.exit(0)

    except KeyboardInterrupt:
        print("\n\n🛑 系统被用户停止")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 系统运行失败: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()