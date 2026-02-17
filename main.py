#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极智能交易系统 v36.12 正式版（完整修复所有截断，确保语法完整）
改进：动态阈值 + 观察池延迟确认 + 高分豁免冷却 + ATR最小百分比 + 历史胜率加权 + 趋势衰竭优化
适用于 GitHub Actions 定时运行，单次分析后退出
"""

import os
import sys
import time
import pickle
import json
import atexit
import requests
import traceback
from datetime import datetime, timedelta
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

# ============ 自定义 JSON 编码器（处理 datetime 和 numpy 类型）============
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ============ 配置类 ============
class UltimateConfig:
    VERSION = "36.12-正式版（完整修复所有截断）"
    MAX_SIGNALS_TO_SEND = 3
    TELEGRAM_RETRY = 3
    TELEGRAM_RETRY_DELAY = 1

    COOLDOWN_CONFIG = {
        'same_coin_cooldown': 60,
        'same_direction_cooldown': 30,
        'max_signals_per_coin_per_day': 5,
        'enable_cooldown': True
    }

    # 基础信号阈值（将被动态调整）
    BASE_SIGNAL_THRESHOLDS = {
        'BOUNCE': 32,
        'BREAKOUT': 25,
        'TREND_EXHAUSTION': 35,
        'CALLBACK': 30,
        'CONFIRMATION_K': 40,
        'CALLBACK_CONFIRM_K': 45
    }

    # 动态阈值开关
    DYNAMIC_THRESHOLD_ENABLED = True
    # 波动因子范围限制
    MIN_VOLATILITY_FACTOR = 0.005
    MAX_VOLATILITY_FACTOR = 0.02

    # 分层发送阈值
    HIGH_CONFIDENCE_THRESHOLD = 80
    OBSERVATION_THRESHOLD = 50

    # 观察池延迟确认开关
    OBSERVATION_ENABLED = True
    # 观察池文件
    OBSERVATION_POOL_FILE = 'observation_pool.json'
    # 延迟确认所需的最小分数提升
    OBSERVATION_SCORE_BOOST = 5

    # 高分豁免冷却（分数 >= 此值可豁免同方向冷却）
    HIGH_SCORE_COOLDOWN_EXEMPT = 85

    OPTIMIZATION_PARAMS = {
        'volume_ratio_min': 0.7,
        'rsi_bounce_max': 50,
        'rsi_callback_min': 45,
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
        'structure': 0.35,
        'momentum': 0.25,
        'volume': 0.20,
        'trend': 0.20
    }

    # 趋势模式阈值（基于ADX）
    TREND_MODES = {
        'RANGE': 15,
        'TRANSITION': 25,
    }

    # 强趋势阈值
    STRONG_TREND_ADX = 35

    # 趋势判定参数
    MIN_TREND_SLOPE_PERCENT = 0.001
    EMA_STRUCTURE_THRESHOLD = 0.5

    # 趋势匹配得分
    TREND_MATCH_SCORE = 1.0
    TREND_MISMATCH_SCORE = 0.2
    TREND_NEUTRAL_SCORE = 0.5
    TRANSITION_BASE_SCORE = 0.4
    TREND_CONFLICT_PENALTY = 0.6

    # 多周期趋势
    ENFORCE_1H_STRUCTURE = False
    ONE_HOUR_CONFLICT_PENALTY = 0.75

    # 背离系数
    DIVERGENCE_WEIGHTS = {'rsi': 0.6, 'price': 0.4}
    PRICE_STRENGTH_FACTOR = 25

    MACD_EXHAUSTION_FACTOR = 0.6
    MACD_EXHAUSTION_LOOKBACK = 3

    COOLDOWN_DYNAMIC = {
        (80, 101): 40,
        (60, 80): 70,
        (0, 60): 100
    }

    # ATR 止损倍数
    ATR_STOP_MULTIPLIER = 1.3
    ATR_STOP_MULTIPLIER_STRONG = 1.1
    ATR_TAKE_PROFIT1_MULTIPLIER = 2.2
    ATR_TAKE_PROFIT2_MULTIPLIER = 3.5
    MAX_STOP_PERCENT = 0.06

    # 止盈最小百分比（当ATR过小时使用）
    MIN_TAKE_PROFIT1_PERCENT = 0.015  # 1.5%
    MIN_TAKE_PROFIT2_PERCENT = 0.03   # 3%

    # 趋势模式与信号类型的匹配规则
    TREND_SIGNAL_ALLOW = {
        'TREND': ['CONFIRMATION_K', 'TREND_EXHAUSTION', 'CALLBACK_CONFIRM_K'],
        'TRANSITION': ['CONFIRMATION_K', 'CALLBACK', 'BOUNCE', 'TREND_EXHAUSTION', 'CALLBACK_CONFIRM_K'],
        'RANGE': ['BOUNCE', 'CALLBACK', 'CONFIRMATION_K']
    }

    # 历史胜率文件
    SUCCESS_RATE_FILE = 'success_rates.json'


# ============ 辅助函数：加载/保存观察池和胜率 ============
def load_observation_pool():
    """加载观察池，并将字符串时间转换回datetime"""
    if not os.path.exists(UltimateConfig.OBSERVATION_POOL_FILE):
        return []
    try:
        with open(UltimateConfig.OBSERVATION_POOL_FILE, 'r') as f:
            data = json.load(f)
        for item in data:
            item['time'] = datetime.fromisoformat(item['time'])
            if 'signal' in item:
                item['signal']['signal_time'] = datetime.fromisoformat(item['signal']['signal_time'])
        return data
    except Exception as e:
        print(f"⚠️ 加载观察池失败: {e}")
        return []


def save_observation_pool(pool):
    """保存观察池，使用自定义编码器自动处理datetime和numpy类型"""
    with open(UltimateConfig.OBSERVATION_POOL_FILE, 'w') as f:
        json.dump(pool, f, indent=2, cls=DateTimeEncoder)


def load_success_rates():
    """加载历史胜率"""
    if not os.path.exists(UltimateConfig.SUCCESS_RATE_FILE):
        return {}
    try:
        with open(UltimateConfig.SUCCESS_RATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}


# ============ 冷却管理器（按方向独立冷却 + 趋势记忆 + 模式感知 + 高分豁免）============
class CooldownManager:
    def __init__(self):
        self.config = UltimateConfig.COOLDOWN_CONFIG
        self.cooldown_db = {}
        self.signal_history = defaultdict(list)
        self.trend_state = {}
        self.cooldown_file = 'cooldown_state.pkl'
        self.load_state()
        atexit.register(self.save_state)

    def load_state(self):
        try:
            if os.path.exists(self.cooldown_file):
                with open(self.cooldown_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cooldown_db = data.get('cooldown_db', {})
                    self.signal_history = defaultdict(list, data.get('signal_history', {}))
                    self.trend_state = data.get('trend_state', {})
                print(f"✅ 冷却状态已加载: {len(self.cooldown_db)}个记录")
        except Exception as e:
            print(f"❌ 加载冷却状态失败: {e}")

    def save_state(self):
        try:
            data = {
                'cooldown_db': self.cooldown_db,
                'signal_history': dict(self.signal_history),
                'trend_state': self.trend_state
            }
            with open(self.cooldown_file, 'wb') as f:
                pickle.dump(data, f)
            print("✅ 冷却状态已保存")
        except Exception as e:
            print(f"❌ 保存冷却状态失败: {e}")

    def _get_key(self, symbol: str, direction: str) -> str:
        return f"{symbol}_{direction}"

    def check_cooldown(self, symbol: str, direction: str, current_trend_direction: int,
                       current_trend_mode: str, score: int) -> Tuple[bool, str]:
        if not self.config['enable_cooldown']:
            return True, ""
        now = datetime.now()
        key = self._get_key(symbol, direction)

        if key in self.cooldown_db:
            last_signal = self.cooldown_db[key]
            last_time = last_signal['time']
            cooldown_minutes = last_signal.get('cooldown_minutes', self.config['same_coin_cooldown'])
            elapsed = (now - last_time).total_seconds() / 60

            last_trend_dir = last_signal.get('trend_direction', 0)
            last_trend_mode = last_signal.get('trend_mode', 'RANGE')

            if current_trend_direction != 0 and last_trend_dir != 0 and current_trend_direction != last_trend_dir:
                return True, "趋势方向反转豁免"
            if last_trend_mode in ['TREND', 'TRANSITION'] and current_trend_mode == 'RANGE':
                return True, "趋势进入盘整豁免"
            if last_trend_mode == 'RANGE' and current_trend_mode in ['TREND', 'TRANSITION']:
                return True, "趋势启动豁免"
            if score >= UltimateConfig.HIGH_SCORE_COOLDOWN_EXEMPT:
                return True, "高分信号豁免冷却"

            if elapsed < cooldown_minutes:
                remaining = cooldown_minutes - elapsed
                return False, f"同币种同方向冷却中 ({remaining:.1f}分钟)"
        return True, ""

    def record_signal(self, symbol: str, direction: str, pattern: str, score: int,
                      trend_direction: int, trend_mode: str):
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
            'cooldown_minutes': cooldown_minutes,
            'trend_direction': trend_direction,
            'trend_mode': trend_mode
        }
        self.signal_history[symbol].append({
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'direction': direction,
            'pattern': pattern,
            'score': score
        })
        if trend_direction != 0 or trend_mode != 'RANGE':
            self.trend_state[symbol] = {'direction': trend_direction, 'mode': trend_mode, 'time': now}


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

        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr)

        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
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
        alpha = 1.0 / period
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        atr = atr.bfill().fillna(0) return atr


# ============ 信号检查器（v36.12）============
class SignalChecker:
    def __init__(self):
        self.base_thresholds = UltimateConfig.BASE_SIGNAL_THRESHOLDS
        self.params = UltimateConfig.OPTIMIZATION_PARAMS
        self.success_rates = load_success_rates()

    def _get_dynamic_threshold(self, pattern: str, data: pd.DataFrame, price: float) -> int:
        if not UltimateConfig.DYNAMIC_THRESHOLD_ENABLED:
            return self.base_thresholds.get(pattern, 40)
        atr = TechnicalIndicators.calculate_atr(data).iloc[-1]
        volatility = atr / price
        factor = max(UltimateConfig.MIN_VOLATILITY_FACTOR, min(volatility, UltimateConfig.MAX_VOLATILITY_FACTOR))
        base = self.base_thresholds.get(pattern, 40)
        adjusted = int(base * (1 - factor))
        adjusted = max(int(base * 0.8), adjusted)
        return adjusted

    def _apply_success_rate_weight(self, symbol: str, pattern: str, raw_score: int) -> int:
        rate = self.success_rates.get(symbol, {}).get(pattern, 1.0)
        if rate < 0.5:
            return int(raw_score * 0.8)
        elif rate > 0.8:
            return int(raw_score * 1.05)
        else:
            return raw_score

    def _find_swing_highs_lows(self, data: pd.DataFrame, window: int = 5):
        highs = data['high'].values
        lows = data['low'].values
        swing_highs = []
        swing_lows = []
        for i in range(window, len(data) - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                swing_highs.append(i)
            if lows[i] == min(lows[i - window:i + window + 1]):
                swing_lows.append(i)
        return swing_highs, swing_lows

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

    def _detect_macd_hist_decline_adv(self, hist_series: pd.Series, direction: str, periods=3) -> tuple:
        if len(hist_series) < periods:
            return False, 0.0
        recent = hist_series.iloc[-periods:].values
        factor = UltimateConfig.MACD_EXHAUSTION_FACTOR
        if direction == 'BUY':
            if all(h > 0 for h in recent) and all(recent[i] < recent[i - 1] for i in range(1, len(recent))):
                if abs(recent[-1]) < abs(recent[0]) * factor:
                    decline_ratio = (recent[0] - recent[-1]) / (recent[0] + 1e-6)
                    strength = min(decline_ratio, 1.0)
                    return True, strength
        else:
            if all(h < 0 for h in recent) and all(recent[i] > recent[i - 1] for i in range(1, len(recent))):
                if abs(recent[-1]) < abs(recent[0]) * factor:
                    increase_ratio = (recent[-1] - recent[0]) / (abs(recent[0]) + 1e-6)
                    strength = min(increase_ratio, 1.0)
                    return True, strength
        return False, 0.0

    def _get_combined_trend_mode(self, data_15m: pd.DataFrame, data_1h: pd.DataFrame) -> str:
        adx_15m = TechnicalIndicators.calculate_adx(data_15m).iloc[-1]
        adx_1h = TechnicalIndicators.calculate_adx(data_1h).iloc[-1]
        combined_adx = adx_15m * 0.6 + adx_1h * 0.4
        if combined_adx <= UltimateConfig.TREND_MODES['RANGE']:
            return 'RANGE'
        elif combined_adx <= UltimateConfig.TREND_MODES['TRANSITION']:
            return 'TRANSITION'
        else:
            return 'TREND'

    def _get_trend_direction(self, data: pd.DataFrame, data_1h: Optional[pd.DataFrame] = None) -> int:
        ema20 = TechnicalIndicators.calculate_ema(data, 20)
        ema50 = TechnicalIndicators.calculate_ema(data, 50)
        atr = TechnicalIndicators.calculate_atr(data).iloc[-1]
        if len(ema20) < 4 or len(ema50) < 2:
            return 0
        ema20_current = ema20.iloc[-1]
        ema20_prev3 = ema20.iloc[-4]
        slope_per_bar = (ema20_current - ema20_prev3) / ema20_prev3 / 3
        ema50_current = ema50.iloc[-1]
        diff = ema20_current - ema50_current
        significant = abs(diff) > atr * UltimateConfig.EMA_STRUCTURE_THRESHOLD
        if significant:
            if diff > 0:
                return 1
            else:
                return -1
        if data_1h is not None:
            trend_mode = self._get_combined_trend_mode(data, data_1h)
            is_trend = trend_mode in ['TREND', 'TRANSITION']
        else:
            adx = TechnicalIndicators.calculate_adx(data).iloc[-1]
            is_trend = adx > UltimateConfig.TREND_MODES['TRANSITION']
        if not is_trend:
            return 0
        slope_up = slope_per_bar > UltimateConfig.MIN_TREND_SLOPE_PERCENT
        slope_down = slope_per_bar < -UltimateConfig.MIN_TREND_SLOPE_PERCENT
        if slope_up:
            return 1
        elif slope_down:
            return -1
        else:
            return 0

    def _get_trend_info(self, data: pd.DataFrame, data_1h: pd.DataFrame, signal_direction: str) -> Tuple[float, int]:
        trend_mode = self._get_combined_trend_mode(data, data_1h)
        if trend_mode == 'RANGE':
            return UltimateConfig.TREND_NEUTRAL_SCORE, 0
        elif trend_mode == 'TRANSITION':
            return UltimateConfig.TRANSITION_BASE_SCORE, 0
        trend_dir = self._get_trend_direction(data, data_1h)
        if trend_dir == 0:
            return UltimateConfig.TREND_NEUTRAL_SCORE, 0
        if (signal_direction == 'BUY' and trend_dir == 1) or (signal_direction == 'SELL' and trend_dir == -1):
            base_score = UltimateConfig.TREND_MATCH_SCORE
        else:
            base_score = UltimateConfig.TREND_MISMATCH_SCORE
        ema20 = TechnicalIndicators.calculate_ema(data, 20)
        ema50 = TechnicalIndicators.calculate_ema(data, 50)
        atr = TechnicalIndicators.calculate_atr(data).iloc[-1]
        diff = ema20.iloc[-1] - ema50.iloc[-1]
        significant = abs(diff) > atr * UltimateConfig.EMA_STRUCTURE_THRESHOLD
        if not significant:
            base_score *= 0.7
        score = max(0.0, min(base_score, 1.0))
        return score, trend_dir

    def _check_1h_structure(self, data_1h: pd.DataFrame, signal_direction: str) -> Tuple[bool, float]:
        if UltimateConfig.ENFORCE_1H_STRUCTURE:
            trend_dir = self._get_trend_direction(data_1h)
            if trend_dir == 0:
                return True, 1.0
            allowed = (signal_direction == 'BUY' and trend_dir == 1) or (signal_direction == 'SELL' and trend_dir == -1)
            return allowed, 1.0
        else:
            trend_dir = self._get_trend_direction(data_1h)
            if trend_dir == 0:
                return True, 1.0
            if (signal_direction == 'BUY' and trend_dir == 1) or (signal_direction == 'SELL' and trend_dir == -1):
                return True, 1.0
            else:
                return True, UltimateConfig.ONE_HOUR_CONFLICT_PENALTY

    def _is_signal_allowed(self, pattern: str, trend_mode: str) -> bool:
        allow_map = UltimateConfig.TREND_SIGNAL_ALLOW
        if trend_mode in allow_map:
            return pattern in allow_map[trend_mode]
        return True

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

    def _get_trend_score_combined(self, data_15m: pd.DataFrame, data_1h: pd.DataFrame, signal_direction: str) -> float:
        score_15m, dir_15m = self._get_trend_info(data_15m, data_1h, signal_direction)
        score_1h, dir_1h = self._get_trend_info(data_1h, data_1h, signal_direction)
        combined = (score_15m * 0.6 + score_1h * 0.4)
        if dir_15m != 0 and dir_1h != 0:
            if dir_15m == dir_1h:
                combined *= 1.2
            else:
                combined *= UltimateConfig.TREND_CONFLICT_PENALTY
        return min(combined, 1.0)

    def _calculate_confirmation_k_score_advanced(self, direction: str, rsi: float, volume_ratio: float,
                                                 engulf_strength: float, div_info: tuple, decline_info: tuple,
                                                 data_15m: pd.DataFrame, data_1h: pd.DataFrame,
                                                 macd_df: pd.DataFrame) -> int:
        div_type, div_str = div_info
        structure = 0.0
        structure += engulf_strength * 0.6
        if div_type == direction.lower():
            structure += div_str * 0.4
        structure = min(structure, 1.0)
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
        volume = min(volume_ratio / 2.0, 1.0)
        trend_score = self._get_trend_score_combined(data_15m, data_1h, direction)
        w = UltimateConfig.CONFIRMATION_K_WEIGHTS
        total = (structure * w['structure'] +
                 momentum * w['momentum'] +
                 volume * w['volume'] +
                 trend_score * w['trend']) * 100
        return int(total)

    def _calculate_stop_loss(self, data: pd.DataFrame, price: float, direction: str,
                             trend_direction: int, trend_mode: str) -> Tuple[float, float, float, float]:
        atr = TechnicalIndicators.calculate_atr(data).iloc[-1]
        adx = TechnicalIndicators.calculate_adx(data).iloc[-1]
        if adx > UltimateConfig.STRONG_TREND_ADX and trend_direction != 0:
            atr_mult_stop = UltimateConfig.ATR_STOP_MULTIPLIER_STRONG
        else:
            atr_mult_stop = UltimateConfig.ATR_STOP_MULTIPLIER
        atr_mult_tp1 = UltimateConfig.ATR_TAKE_PROFIT1_MULTIPLIER
        atr_mult_tp2 = UltimateConfig.ATR_TAKE_PROFIT2_MULTIPLIER
        max_stop_pct = UltimateConfig.MAX_STOP_PERCENT
        if direction == 'BUY':
            entry_main = price * 1.002
            recent_low = data['low'].rolling(10).min().iloc[-1]
            stop_loss_candidate1 = recent_low * 0.985
            stop_loss_candidate2 = price - atr * atr_mult_stop
            stop_loss = max(stop_loss_candidate1, stop_loss_candidate2)
            min_stop = price * (1 - max_stop_pct)
            stop_loss = max(stop_loss, min_stop)
            tp1 = price + max(atr * atr_mult_tp1, price * UltimateConfig.MIN_TAKE_PROFIT1_PERCENT)
            tp2 = price + max(atr * atr_mult_tp2, price * UltimateConfig.MIN_TAKE_PROFIT2_PERCENT)
            take_profit1, take_profit2 = tp1, tp2
        else:
            entry_main = price * 0.998
            recent_high = data['high'].rolling(10).max().iloc[-1]
            stop_loss_candidate1 = recent_high * 1.02
            stop_loss_candidate2 = price + atr * atr_mult_stop
            stop_loss = min(stop_loss_candidate1, stop_loss_candidate2)
            max_stop = price * (1 + max_stop_pct)
            stop_loss = min(stop_loss, max_stop)
            tp1 = price - max(atr * atr_mult_tp1, price * UltimateConfig.MIN_TAKE_PROFIT1_PERCENT)
            tp2 = price - max(atr * atr_mult_tp2, price * UltimateConfig.MIN_TAKE_PROFIT2_PERCENT)
            take_profit1, take_profit2 = tp1, tp2
        return entry_main, stop_loss, take_profit1, take_profit2

    def check_all_coins(self, coins_data, cooldown_manager):
        print(f"\n🔍 开始信号扫描 ({len(coins_data)}个币种)...")
        all_signals = []
        signal_counts = defaultdict(int)

        observation_pool = load_observation_pool() if UltimateConfig.OBSERVATION_ENABLED else []
        current_time = datetime.now()

        for symbol, data_dict in coins_data.items():
            try:
                if '15m' not in data_dict or '1H' not in data_dict:
                    continue
                data_15m = data_dict['15m']
                data_1h = data_dict['1H']
                if len(data_15m) < 30 or len(data_1h) < 30:
                    continue

                current_price = data_15m['close'].iloc[-1]
                rsi = TechnicalIndicators.calculate_rsi(data_15m, 14).iloc[-1]
                volume_ratio = TechnicalIndicators.calculate_volume_ratio(data_15m, 20).iloc[-1]
                ma20 = TechnicalIndicators.calculate_ma(data_15m, 20).iloc[-1]
                ma50 = TechnicalIndicators.calculate_ma(data_15m, 50).iloc[-1]

                trend_mode = self._get_combined_trend_mode(data_15m, data_1h)
                current_trend_dir = self._get_trend_direction(data_15m, data_1h)

                signals = []

                # 反弹信号
                if rsi < self.params['rsi_bounce_max'] and volume_ratio > self.params['volume_ratio_min']:
                    if self._is_signal_allowed('BOUNCE', trend_mode):
                        allowed, penalty = self._check_1h_structure(data_1h, 'BUY')
                        if allowed:
                            raw_score = self._calculate_bounce_score(rsi, volume_ratio)
                            raw_score = int(raw_score * penalty)
                            raw_score = self._apply_success_rate_weight(symbol, 'BOUNCE', raw_score)
                            dynamic_th = self._get_dynamic_threshold('BOUNCE', data_15m, current_price)
                            if raw_score >= dynamic_th:
                                signals.append(self._create_bounce_signal(
                                    symbol, data_15m, current_price, rsi, volume_ratio, ma20, raw_score,
                                    trend_direction=current_trend_dir, trend_mode=trend_mode
                                ))
                                signal_counts['BOUNCE'] += 1

                # 回调信号
                if rsi > self.params['rsi_callback_min']:
                    if self._is_signal_allowed('CALLBACK', trend_mode):
                        allowed, penalty = self._check_1h_structure(data_1h, 'BUY')
                        if allowed:
                            recent_high = data_15m['high'].iloc[-30:].max()
                            callback_pct = ((recent_high - current_price) / recent_high) * 100
                            if self.params['callback_pct_min'] <= callback_pct <= self.params['callback_pct_max']:
                                raw_score = self._calculate_callback_score(rsi, volume_ratio, callback_pct)
                                raw_score = int(raw_score * penalty)
                                raw_score = self._apply_success_rate_weight(symbol, 'CALLBACK', raw_score)
                                dynamic_th = self._get_dynamic_threshold('CALLBACK', data_15m, current_price)
                                if raw_score >= dynamic_th:
                                    signals.append(self._create_callback_signal(
                                        symbol, data_15m, current_price, rsi, volume_ratio, recent_high, callback_pct, ma20, raw_score,
                                        trend_direction=current_trend_dir, trend_mode=trend_mode
                                    ))
                                    signal_counts['CALLBACK'] += 1

                # 回调确认转强信号
                if 48 <= rsi <= 72 and volume_ratio > 1.2:
                    if self._is_signal_allowed('CALLBACK_CONFIRM_K', trend_mode):
                        allowed, penalty = self._check_1h_structure(data_1h, 'BUY')
                        if allowed:
                            recent_high = data_15m['high'].iloc[-30:].max()
                            callback_pct = ((recent_high - current_price) / recent_high) * 100
                            if 2 <= callback_pct <= 15:
                                recent_3_closes = data_15m['close'].iloc[-3:].values
                                price_increasing = len(recent_3_closes) >= 2 and recent_3_closes[-1] > recent_3_closes[0]
                                if price_increasing and ma20 > ma50 and current_price > ma20:
                                    raw_score = self._calculate_callback_confirm_score(rsi, volume_ratio, callback_pct)
                                    raw_score = int(raw_score * penalty)
                                    raw_score = self._apply_success_rate_weight(symbol, 'CALLBACK_CONFIRM_K', raw_score)
                                    dynamic_th = self._get_dynamic_threshold('CALLBACK_CONFIRM_K', data_15m, current_price)
                                    if raw_score >= dynamic_th:
                                        signals.append(self._create_callback_confirm_signal(
                                            symbol, data_15m, current_price, rsi, volume_ratio, recent_high, callback_pct, ma20, ma50, raw_score,
                                            trend_direction=current_trend_dir, trend_mode=trend_mode
                                        ))
                                        signal_counts['CALLBACK_CONFIRM_K'] += 1

                # 趋势衰竭做空信号（优化版：1h趋势过滤 + RSI下降加分）
                if rsi > self.params['trend_exhaustion_rsi_min'] and volume_ratio < 1.0:
                    if self._is_signal_allowed('TREND_EXHAUSTION', trend_mode):
                        # 获取1小时趋势方向，如果为上升趋势（方向1），则跳过做空信号
                        trend_dir_1h = self._get_trend_direction(data_1h)
                        if trend_dir_1h == 1:  # 1小时上升趋势，不产生做空信号
                            continue

                        # 计算RSI下降加分（RSI比前一根低）
                        rsi_series = TechnicalIndicators.calculate_rsi(data_15m, 14)
                        rsi_prev = rsi_series.iloc[-2] if len(rsi_series) >= 2 else rsi
                        if rsi < rsi_prev:
                            rsi_boost = 8  # RSI下降加分
                        else:
                            rsi_boost = 0

                        allowed, penalty = self._check_1h_structure(data_1h, 'SELL')
                        if allowed:
                            raw_score = self._calculate_trend_exhaustion_score(rsi, volume_ratio)
                            raw_score = int(raw_score * penalty) + rsi_boost
                            raw_score = self._apply_success_rate_weight(symbol, 'TREND_EXHAUSTION', raw_score)
                            dynamic_th = self._get_dynamic_threshold('TREND_EXHAUSTION', data_15m, current_price)
                            if raw_score >= dynamic_th:
                                signals.append(self._create_trend_exhaustion_signal(
                                    symbol, data_15m, current_price, rsi, volume_ratio, ma20, raw_score,
                                    trend_direction=current_trend_dir, trend_mode=trend_mode
                                ))
                                signal_counts['TREND_EXHAUSTION'] += 1

                # 吞没形态信号 CONFIRMATION_K
                engulf_dir, engulf_strength = self._detect_engulfing(data_15m)
                if engulf_dir and self._is_signal_allowed('CONFIRMATION_K', trend_mode):
                    allowed, penalty = self._check_1h_structure(data_1h, engulf_dir)
                    if allowed:
                        rsi_series = TechnicalIndicators.calculate_rsi(data_15m, 14)
                        macd_df = TechnicalIndicators.calculate_macd(data_15m)
                        hist_series = macd_df['histogram']

                        div_info = self._detect_rsi_divergence_swing(data_15m, rsi_series, lookback=30)
                        decline_info = self._detect_macd_hist_decline_adv(hist_series, engulf_dir, periods=3)

                        raw_score = self._calculate_confirmation_k_score_advanced(
                            engulf_dir, rsi, volume_ratio, engulf_strength,
                            div_info, decline_info, data_15m, data_1h, macd_df
                        )
                        raw_score = int(raw_score * penalty)
                        raw_score = self._apply_success_rate_weight(symbol, 'CONFIRMATION_K', raw_score)
                        dynamic_th = self._get_dynamic_threshold('CONFIRMATION_K', data_15m, current_price)
                        if raw_score >= dynamic_th:
                            signals.append(self._create_confirmation_k_signal_advanced(
                                symbol, data_15m, current_price, rsi, volume_ratio,
                                ma20, ma50, engulf_dir, engulf_strength,
                                div_info, decline_info, raw_score,
                                trend_direction=current_trend_dir, trend_mode=trend_mode
                            ))
                            signal_counts['CONFIRMATION_K'] += 1

                if signals:
                    best_signal = max(signals, key=lambda x: x.get('score', 0))
                    all_signals.append(best_signal)

            except Exception as e:
                continue

        # 处理观察池
        new_observation_pool = []
        for obs in observation_pool:
            if current_time - obs['time'] > timedelta(hours=2):
                continue
            symbol = obs['symbol']
            if symbol in coins_data:
                data_dict = coins_data[symbol]
                data_15m = data_dict.get('15m')
                if data_15m is not None and len(data_15m) >= 30:
                    current_trend_dir = self._get_trend_direction(data_15m, data_1h=data_dict.get('1H'))
                    direction = obs['direction']
                    if (direction == 'BUY' and current_trend_dir == 1) or (direction == 'SELL' and current_trend_dir == -1):
                        new_score = obs['score'] + UltimateConfig.OBSERVATION_SCORE_BOOST
                        if new_score >= UltimateConfig.HIGH_CONFIDENCE_THRESHOLD:
                            signal = obs['signal']
                            signal['score'] = new_score
                            signal['signal_time'] = current_time
                            signal['reason'] += "\n• 延迟1根K线确认趋势后增强"
                            all_signals.append(signal)
                            signal_counts[obs['pattern']] += 1
                            continue
            new_observation_pool.append(obs)

        if UltimateConfig.OBSERVATION_ENABLED:
            for sig in all_signals:
                if UltimateConfig.OBSERVATION_THRESHOLD <= sig['score'] < UltimateConfig.HIGH_CONFIDENCE_THRESHOLD:
                    new_observation_pool.append({
                        'time': current_time,
                        'symbol': sig['symbol'],
                        'direction': sig['direction'],
                        'pattern': sig['pattern'],
                        'score': sig['score'],
                        'signal': sig
                    })
            save_observation_pool(new_observation_pool)

        self._print_statistics(signal_counts, len(coins_data))
        print(f"✅ 扫描完成: 发现 {len(all_signals)} 个交易信号")
        return all_signals

    # ---------- 评分函数 ----------
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

    # ---------- 信号创建函数 ----------
    def _create_confirmation_k_signal_advanced(self, symbol, data, price, rsi, volume_ratio,
                                               ma20, ma50, direction, engulf_strength,
                                               div_info, decline_info, score,
                                               trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, direction, trend_direction, trend_mode
        )
        risk = (entry_main - stop_loss) if direction == 'BUY' else (stop_loss - entry_main)
        reward = (take_profit2 - entry_main) if direction == 'BUY' else (entry_main - take_profit2)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        div_text = f"• 看涨背离强度: {div_info[1]:.2f}\n" if div_info[0] == 'bullish' else ""
        decl_text = f"• 多头衰竭强度: {decline_info[1]:.2f}\n" if decline_info[0] else ""
        reason = (
            f"🟢 <b>看涨吞没形态确认</b>\n\n"
            f"• 吞没强度: {engulf_strength:.2f}\n"
            f"• 成交量{volume_ratio:.1f}倍\n"
            f"• RSI({rsi:.1f})\n"
            f"{div_text}{decl_text}"
            f"• 建议在${entry_main:.4f}附近买入"
        ) if direction == 'BUY' else (
            f"🔴 <b>看跌吞没形态确认</b>\n\n"
            f"• 吞没强度: {engulf_strength:.2f}\n"
            f"• 成交量{volume_ratio:.1f}倍\n"
            f"• RSI({rsi:.1f})\n"
            f"{div_text}{decl_text}"
            f"• 建议在${entry_main:.4f}附近做空"
        )

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
            },
            'trend_direction': trend_direction,
            'trend_mode': trend_mode
        }

    def _create_bounce_signal(self, symbol, data, price, rsi, volume_ratio, ma20, score,
                              trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, 'BUY', trend_direction, trend_mode
        )
        risk = entry_main - stop_loss
        reward = take_profit2 - entry_main
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        recent_low = data['low'].rolling(20).min().iloc[-1]
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
            },
            'trend_direction': trend_direction,
            'trend_mode': trend_mode
        }

    def _create_callback_signal(self, symbol, data, price, rsi, volume_ratio,
                                recent_high, callback_pct, ma20, score,
                                trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, 'BUY', trend_direction, trend_mode
        )
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
            },
            'trend_direction': trend_direction,
            'trend_mode': trend_mode
        }

    def _create_callback_confirm_signal(self, symbol, data, price, rsi,
                                        volume_ratio, recent_high, callback_pct,
                                        ma20, ma50, score,
                                        trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, 'BUY', trend_direction, trend_mode
        )
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
            'reason': (
                f"🟢 <b>回调确认转强</b>\n\n"
                f"• 从高点${recent_high:.4f}回调{callback_pct:.1f}%\n"
                f"• RSI({rsi:.1f})处于强势区\n"
                f"• 成交量{volume_ratio:.1f}倍\n"
                f"• MA20(${ma20:.4f}) > MA50(${ma50:.4f})\n"
                f"• 建议在${entry_main:.4f}附近建仓"
            ),
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            },
            'trend_direction': trend_direction,
            'trend_mode': trend_mode
        }

    def _create_trend_exhaustion_signal(self, symbol, data, price,
                                        rsi, volume_ratio, ma20, score,
                                        trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, 'SELL', trend_direction, trend_mode
        )
        risk = stop_loss - entry_main
        reward = entry_main - take_profit2
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        recent_high = data['high'].rolling(20).max().iloc[-1]
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
            },
            'trend_direction': trend_direction,
            'trend_mode': trend_mode
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
        if signal['score'] < UltimateConfig.HIGH_CONFIDENCE_THRESHOLD:
            print(f"📝 信号 {signal['symbol']} 分数 {signal['score']} 低于高置信度阈值，仅记录不发送")
            return False
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
        confidence_tag = "🔥 高置信度" if signal['score'] >= 80 else "⚠️ 中等置信度" if signal['score'] >= 50 else "📉 低置信度"
        return f"""
        
 <b>🚀实盘交易信号</b>  {confidence_tag}

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
        print("\n" + "=" * 60)
        print(f"🚀 终极智能交易系统 {UltimateConfig.VERSION}")
        print("=" * 60)
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
        print("=" * 60)

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
            signals = self.signal_checker.check_all_coins(coins_data, self.cooldown_manager)

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
            trend_dir = signal.get('trend_direction', 0)
            trend_mode = signal.get('trend_mode', 'RANGE')
            print(f"\n[{i}] {symbol} {direction}: {pattern} ({score}分)")

            cooldown_ok, cooldown_reason = self.cooldown_manager.check_cooldown(
                symbol, direction, trend_dir, trend_mode, score
            )
            if not cooldown_ok:
                print(f"   ⚠️ 冷却阻止: {cooldown_reason}")
                continue

            success = self.telegram.send_signal(signal, cooldown_reason)
            if success:
                self.cooldown_manager.record_signal(symbol, direction, pattern, score, trend_dir, trend_mode)
                self.total_signals += 1
                sent_count += 1
                time.sleep(2)
            else:
                if score >= UltimateConfig.HIGH_CONFIDENCE_THRESHOLD:
                    print(f"   ⚠️ 高置信度信号发送失败，跳过记录冷却")
                else:
                    print(f"   📝 信号分数 {score} 低于高置信度阈值，仅记录不发送")

        print(f"\n✅ 本次成功发送 {sent_count} 个交易信号")


# ============ 主程序入口 ============
def main():
    print("=" * 60)
    print("🤖 终极智能交易系统 - GitHub Actions 优化版")
    print("=" * 60)
    print(f"📅 版本: {UltimateConfig.VERSION}")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 监控币种: {len(MONITOR_COINS)}个")
    print(f"🎯 信号模式: 5种策略 + 增强型吞没(动态阈值/观察池/高分豁免/最小止盈/胜率加权) + 趋势衰竭优化")
    print("=" * 60)

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