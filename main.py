#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极智能交易系统 v37.0 正式版（全面优化版）
优化特性：
1. 多周期背离共振检测（15m+1h+4h）
2. 成交量协同性检查（缩量回调/放量突破）
3. RSI极端区域过滤
4. 趋势衰竭增加MACD死叉确认
5. 动态仓位计算（基于评分+波动率）
6. 相关性风险控制（板块限制）
7. 时间衰减因子
8. 数据源冗余（OKX+Binance）
9. 异常K线过滤
10. 内置轻量回测引擎
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
from collections import defaultdict, deque
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import numpy as np
import telebot

# scikit-learn 用于高级分析，如果不需要可以移除（但建议保留）
try:
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None

# ============ 配置 ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

OKX_API_BASE_URL = "https://www.okx.com"
BINANCE_API_BASE_URL = "https://api.binance.com"

OKX_CANDLE_INTERVAL = ["15m", "1H", "4H"]  # 增加4小时周期
BINANCE_CANDLE_INTERVAL = ["15m", "1h", "4h"]
CANDLE_LIMIT = 200  # 增加数据量

# 监控币种分组（用于相关性控制）
COIN_GROUPS = {
     'MAJOR': ['BTC','ETH','SOL','ADA','XRP','BCH'],
    'LAYER1': ['AVAX','DOT','NEAR','APT','SUI','SEI'],
    'LAYER2': ['ARB','OP','LDO','IMX','STRK','MANTA'],
    'DEFI': ['UNI','LINK','AAVE','COMP','CRV','MKR','DYDX','SUSHI','GMX','RDNT'],
    'GAMING': ['SAND','MANA','ENJ','GALA','AXS','RON','GMT'],
    'MEME': ['DOGE','SHIB','PEPE','FLOKI','BONK'],
    'PRIVACY': ['LIT','ZEC','DASH'],
    'EXCHANGE': ['CRO','TRX','BNB','HYPE','KCS'],
    'AI': ['RNDR','OCEAN','GRT','NMR'],
    'RWA': ['ONDO','CFG','POLYX'],
    'STORAGE': ['FIL','AR','STX'],
    'OTHERS': []
}

# 所有币种
MONITOR_COINS = []
for group in COIN_GROUPS.values():
    MONITOR_COINS.extend(group)
MONITOR_COINS = list(set(MONITOR_COINS))  # 去重

# 板块映射
COIN_TO_GROUP = {}
for group_name, coins in COIN_GROUPS.items():
    for coin in coins:
        COIN_TO_GROUP[coin] = group_name

print(f"📊 监控币种: {len(MONITOR_COINS)}个, 分为{len(set(COIN_TO_GROUP.values()))}个板块")

# 调试开关
DEBUG = os.environ.get("DEBUG", "0") == "1"
BACKTEST_MODE = os.environ.get("BACKTEST", "0") == "1"  # 回测模式

# ============ 自定义 JSON 编码器 ============
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
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# ============ 数据类定义 ============
@dataclass
class Signal:
    symbol: str
    pattern: str
    direction: str  # 'BUY' or 'SELL'
    score: int
    rsi: float
    volume_ratio: float
    current_price: float
    entry_points: Dict[str, float]
    reason: str
    signal_time: datetime
    trend_direction: int
    trend_mode: str
    position_size: float
    group: str = 'OTHERS'
    time_decay: float = 1.0
    confidence: float = 0.5

@dataclass
class BacktestResult:
    symbol: str
    pattern: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    direction: str
    profit_pct: float
    score: int
    reason: str

# ============ 高级配置类 ============
class UltimateConfig:
    VERSION = "37.0-全面优化版"
    
    # ===== 基础设置 =====
    MAX_SIGNALS_TO_SEND = 3
    TELEGRAM_RETRY = 3
    TELEGRAM_RETRY_DELAY = 1
    
    # ===== 冷却配置 =====
    COOLDOWN_CONFIG = {
        'same_coin_cooldown': 60,
        'same_direction_cooldown': 30,
        'max_signals_per_coin_per_day': 5,
        'enable_cooldown': True
    }
    
    # ===== 信号阈值 =====
    BASE_SIGNAL_THRESHOLDS = {
        'BOUNCE': 32,
        'BREAKOUT': 25,
        'TREND_EXHAUSTION': 35,
        'CALLBACK': 30,
        'CONFIRMATION_K': 40,
        'CALLBACK_CONFIRM_K': 45
    }
    
    # ===== 动态阈值 =====
    DYNAMIC_THRESHOLD_ENABLED = True
    MIN_VOLATILITY_FACTOR = 0.005
    MAX_VOLATILITY_FACTOR = 0.02
    
    # ===== 发送阈值 =====
    HIGH_CONFIDENCE_THRESHOLD = 80
    OBSERVATION_THRESHOLD = 50
    OBSERVATION_ENABLED = True
    OBSERVATION_POOL_FILE = 'observation_pool.json'
    OBSERVATION_SCORE_BOOST = 5
    
    # ===== 仓位配置 =====
    POSITION_SIZE_BASE = 1.0          # 基础仓位
    POSITION_SIZE_MIN = 0.1            # 最小仓位
    POSITION_SIZE_MAX = 1.0            # 最大仓位
    VOLATILITY_POSITION_ADJUST = True  # 根据波动率调整仓位
    MAX_POSITION_PER_GROUP = 0.3       # 单个板块最大总仓位
    
    # ===== 时间衰减 =====
    TIME_DECAY_ENABLED = True
    TIME_DECAY_HOURS = 24               # 24小时内衰减
    TIME_DECAY_MIN_FACTOR = 0.5         # 最小衰减因子
    
    # ===== 相关性控制 =====
    CORRELATION_CONTROL_ENABLED = True
    MAX_SIGNALS_PER_GROUP = 2            # 每个板块最多同时发几个信号
    
    # ===== 多周期配置 =====
    MULTI_TIMEFRAME_WEIGHTS = {
        '15m': 0.4,
        '1H': 0.35,
        '4H': 0.25
    }
    
    # ===== 背离检测 =====
    DIVERGENCE_CONFIG = {
        'min_lookback': 30,
        'swing_window': 3,
        'rsi_weight': 0.6,
        'price_weight': 0.4,
        'macd_weight': 0.3,
        'consecutive_bonus': 1.5,
        'multi_tf_bonus': 1.3
    }
    
    # ===== RSI配置 =====
    RSI_CONFIG = {
        'overbought': 75,
        'oversold': 25,
        'extreme_penalty': 0.8,
        'bounce_limits': {
            'TREND': 40,
            'TRANSITION': 45,
            'RANGE': 50
        }
    }
    
    # ===== 成交量配置 =====
    VOLUME_CONFIG = {
        'min_ratio': 0.7,
        'surge_threshold': 1.5,
        'shrink_threshold': 0.8,
        'ultra_low': 0.3,
        'low_penalty': 6,
        'lookback_period': 10
    }
    
    # ===== 趋势衰竭专用 =====
    TREND_EXHAUSTION = {
        'volume_ultra_low': 0.3,
        'volume_low_penalty': 6,
        'structure_high_window': 10,
        'structure_low_window': 10,
        'stop_buffer': 0.0015,
        'require_macd_cross': True,      # 要求MACD死叉
        'macd_lookback': 3
    }
    
    # ===== K线形态 =====
    CANDLE_PATTERNS = {
        'engulfing_min_ratio': 1.2,
        'doji_body_ratio': 0.1,
        'hammer_ratio': 2.0,
        'shooting_star_ratio': 2.0
    }
    
    # ===== 止损止盈 =====
    STOP_LOSS = {
        'atr_multiplier': 1.3,
        'atr_multiplier_strong': 1.1,
        'tp1_multiplier': 2.2,
        'tp2_multiplier': 3.5,
        'max_stop_percent': 0.06,
        'min_tp1_percent': 0.015,
        'min_tp2_percent': 0.03
    }
    
    # ===== ATR动态调整 =====
    ATR_CONFIG = {
        'period': 14,
        'smooth_period': 5
    }
    
    # ===== 趋势判定 =====
    TREND = {
        'range_adx': 15,
        'transition_adx': 25,
        'strong_trend_adx': 35,
        'min_slope_percent': 0.001,
        'ema_structure_threshold': 0.5
    }
    
    # ===== 趋势匹配得分 =====
    TREND_SCORES = {
        'match': 1.0,
        'mismatch': 0.2,
        'neutral': 0.5,
        'transition': 0.4,
        'conflict_penalty': 0.6
    }
    
    # ===== 多周期趋势 =====
    ENFORCE_1H_STRUCTURE = False
    ONE_HOUR_CONFLICT_PENALTY = 0.75
    
    # ===== CONFIRMATION_K权重 =====
    CONFIRMATION_K_WEIGHTS = {
        'structure': 0.35,
        'momentum': 0.25,
        'volume': 0.20,
        'trend': 0.20
    }
    
    # ===== 冷却动态调整 =====
    COOLDOWN_DYNAMIC = {
        (80, 101): 40,
        (60, 80): 70,
        (0, 60): 100
    }
    
    # ===== 趋势模式与信号匹配 =====
    TREND_SIGNAL_ALLOW = {
        'TREND': ['CONFIRMATION_K', 'TREND_EXHAUSTION', 'CALLBACK_CONFIRM_K'],
        'TRANSITION': ['CONFIRMATION_K', 'CALLBACK', 'BOUNCE', 'TREND_EXHAUSTION', 'CALLBACK_CONFIRM_K'],
        'RANGE': ['BOUNCE', 'CALLBACK', 'CONFIRMATION_K']
    }
    
    # ===== 文件 =====
    SUCCESS_RATE_FILE = 'success_rates.json'
    BACKTEST_RESULTS_FILE = 'backtest_results.json'
    PERFORMANCE_LOG_FILE = 'performance.log'
    
    # ===== 异常检测 =====
    ANOMALY_DETECTION = {
        'enabled': True,
        'price_jump_threshold': 0.1,      # 10%价格跳空
        'volume_spike_threshold': 10,      # 10倍成交量异常
        'lookback_period': 5
    }
    
    # ===== 优化参数（兼容旧代码）=====
    OPTIMIZATION_PARAMS = {
        'volume_ratio_min': VOLUME_CONFIG['min_ratio'],
        'rsi_callback_min': 45,
        'callback_pct_min': 2,
        'callback_pct_max': 25,
        'trend_exhaustion_rsi_min': 65,
    }

# ============ 辅助函数：加载/保存观察池和胜率 ============
def load_observation_pool():
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
    with open(UltimateConfig.OBSERVATION_POOL_FILE, 'w') as f:
        json.dump(pool, f, indent=2, cls=DateTimeEncoder)

def load_success_rates():
    if not os.path.exists(UltimateConfig.SUCCESS_RATE_FILE):
        return {}
    try:
        with open(UltimateConfig.SUCCESS_RATE_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_success_rates(rates):
    with open(UltimateConfig.SUCCESS_RATE_FILE, 'w') as f:
        json.dump(rates, f, indent=2)

def log_performance(message):
    """记录性能日志"""
    try:
        with open(UltimateConfig.PERFORMANCE_LOG_FILE, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"[{timestamp}] {message}\n")
    except:
        pass

# ============ 冷却管理器 ============
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
        
        old_record = self.cooldown_db.get(key)
        old_score = old_record.get('score', 0) if old_record else 0
        
        if score >= old_score:
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
            if DEBUG and score > old_score:
                print(f"📈 信号评分提高，更新冷却记录: {key} {score} (原{old_score})")
        else:
            if DEBUG:
                print(f"⏭️ 新信号评分({score})低于现有记录({old_score})，跳过更新")
            return

        self.signal_history[symbol].append({
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'direction': direction,
            'pattern': pattern,
            'score': score
        })
        if trend_direction != 0 or trend_mode != 'RANGE':
            self.trend_state[symbol] = {'direction': trend_direction, 'mode': trend_mode, 'time': now}

# ============ 多源数据获取器 ============
class MultiSourceDataFetcher:
    def __init__(self):
        self.okx = OKXDataFetcher()
        self.binance = BinanceDataFetcher()
        self.cache = {}
        self.cache_time = {}
        self.cache_duration = 60  # 缓存1分钟
        self.failed_sources = defaultdict(int)
        self.source_priority = ['okx', 'binance']
        
    def get_candles(self, symbol: str, interval: str, source: str = 'auto'):
        """从指定源获取数据，失败时自动切换"""
        cache_key = f"{symbol}_{interval}"
        current_time = time.time()
        
        if cache_key in self.cache and current_time - self.cache_time.get(cache_key, 0) < self.cache_duration:
            return self.cache[cache_key]
        
        sources = [source] if source != 'auto' else self.source_priority
        
        for src in sources:
            if src == 'okx':
                df = self.okx.get_candles(symbol, interval)
            elif src == 'binance':
                df = self.binance.get_candles(symbol, interval)
            else:
                continue
                
            if df is not None and len(df) >= 30:
                self.cache[cache_key] = df
                self.cache_time[cache_key] = current_time
                self.failed_sources[src] = max(0, self.failed_sources[src] - 1)
                return df
            else:
                self.failed_sources[src] += 1
                
        return None
    
    def get_all_coins_data(self, symbols: List[str], intervals: List[str] = None):
        """并行获取所有币种数据"""
        if intervals is None:
            intervals = OKX_CANDLE_INTERVAL
            
        print(f"\n📡 开始并行获取 {len(symbols)} 个币种的实时数据...")
        coins_data = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_symbol = {}
            for symbol in symbols:
                future = executor.submit(self._get_symbol_data, symbol, intervals)
                future_to_symbol[future] = symbol
            
            total = len(symbols)
            for i, future in enumerate(as_completed(future_to_symbol), 1):
                symbol = future_to_symbol[future]
                try:
                    data_dict = future.result(timeout=10)
                    if data_dict:
                        coins_data[symbol] = data_dict
                        print(f"[{i}/{total}] {symbol}: ✅ 成功")
                    else:
                        print(f"[{i}/{total}] {symbol}: ⚠️ 数据不足")
                except Exception as e:
                    print(f"[{i}/{total}] {symbol}: ❌ 失败 - {str(e)[:50]}")
        
        print(f"\n📊 数据获取完成: {len(coins_data)}/{total} 个币种")
        return coins_data
    
    def _get_symbol_data(self, symbol: str, intervals: List[str]):
        """获取单个币种的多周期数据"""
        data_dict = {}
        for interval in intervals:
            df = self.get_candles(symbol, interval)
            if df is not None and len(df) >= 30:
                data_dict[interval] = df
        return data_dict if data_dict else None

class OKXDataFetcher:
    def __init__(self):
        self.base_url = OKX_API_BASE_URL
        self.endpoint = '/api/v5/market/candles'
        self.retry_times = 3
        self.timeout = 10
        self.headers = {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_candles(self, symbol: str, interval: str):
        """获取OKX K线数据"""
        inst_id = f"{symbol}-USDT"
        
        interval_map = {
            '15m': '15m',
            '1H': '1H',
            '4H': '4H'
        }
        bar = interval_map.get(interval, interval)
        
        params = {
            'instId': inst_id,
            'bar': bar,
            'limit': CANDLE_LIMIT
        }
        url = f"{self.base_url}{self.endpoint}"

        for retry in range(self.retry_times):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    data = response.json()
                    if data['code'] == '0' and len(data['data']) > 0:
                        return self._parse_candles(data['data'])
            except Exception as e:
                if retry < self.retry_times - 1:
                    time.sleep(1)
        return None
    
    def _parse_candles(self, candles):
        df = pd.DataFrame(candles)
        if len(df.columns) < 6:
            return None
        df = df.iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

class BinanceDataFetcher:
    def __init__(self):
        self.base_url = BINANCE_API_BASE_URL
        self.endpoint = '/api/v3/klines'
        self.retry_times = 3
        self.timeout = 10
        self.session = requests.Session()

    def get_candles(self, symbol: str, interval: str):
        """获取Binance K线数据"""
        symbol_pair = f"{symbol}USDT"
        
        interval_map = {
            '15m': '15m',
            '1H': '1h',
            '4H': '4h'
        }
        binance_interval = interval_map.get(interval, interval)
        
        params = {
            'symbol': symbol_pair,
            'interval': binance_interval,
            'limit': CANDLE_LIMIT
        }
        url = f"{self.base_url}{self.endpoint}"

        for retry in range(self.retry_times):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code == 200:
                    return self._parse_candles(response.json())
            except Exception as e:
                if retry < self.retry_times - 1:
                    time.sleep(1)
        return None
    
    def _parse_candles(self, candles):
        df = pd.DataFrame(candles)
        if len(df.columns) < 6:
            return None
        df = df.iloc[:, :6]
        df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

# ============ 高级技术指标计算器 ============
class AdvancedIndicators:
    
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
    def calculate_ema(data: pd.DataFrame, period: int):
        return data['close'].ewm(span=period, adjust=False).mean()

    @staticmethod
    def calculate_sma(data: pd.DataFrame, period: int):
        return data['close'].rolling(window=period).mean()

    @staticmethod
    def calculate_volume_ratio(data: pd.DataFrame, period: int = 20):
        if len(data) < period:
            return pd.Series([1.0] * len(data), index=data.index)
        current_volume = data['volume']
        avg_volume = data['volume'].rolling(window=period).mean()
        return (current_volume / avg_volume).fillna(1.0)

    @staticmethod
    def calculate_macd(data: pd.DataFrame, fast=12, slow=26, signal=9):
        close = data['close']
        exp1 = close.ewm(span=fast, adjust=False).mean()
        exp2 = close.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        return pd.DataFrame({
            'macd': macd,
            'signal': signal_line,
            'histogram': histogram
        }, index=data.index)

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
    def calculate_atr(data: pd.DataFrame, period: int = 14, smooth: bool = True):
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        if smooth:
            alpha = 1.0 / period
            atr = tr.ewm(alpha=alpha, adjust=False).mean()
        else:
            atr = tr.rolling(window=period).mean()
            
        return atr.bfill().fillna(0)

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0):
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        return pd.DataFrame({
            'sma': sma,
            'upper': upper,
            'lower': lower
        }, index=data.index)

    @staticmethod
    def calculate_obv(data: pd.DataFrame):
        obv = (np.sign(data['close'].diff()) * data['volume']).fillna(0).cumsum()
        return obv

    @staticmethod
    def calculate_mfi(data: pd.DataFrame, period: int = 14):
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        money_flow = typical_price * data['volume']
        
        positive_flow = pd.Series(0, index=data.index)
        negative_flow = pd.Series(0, index=data.index)
        
        for i in range(1, len(typical_price)):
            if typical_price.iloc[i] > typical_price.iloc[i-1]:
                positive_flow.iloc[i] = money_flow.iloc[i]
            else:
                negative_flow.iloc[i] = money_flow.iloc[i]
        
        pos_sum = positive_flow.rolling(window=period).sum()
        neg_sum = negative_flow.rolling(window=period).sum()
        
        money_ratio = pos_sum / neg_sum
        mfi = 100 - (100 / (1 + money_ratio))
        return mfi.fillna(50)

    @staticmethod
    def detect_anomalies(data: pd.DataFrame) -> bool:
        """检测异常K线"""
        if not UltimateConfig.ANOMALY_DETECTION['enabled']:
            return False
            
        config = UltimateConfig.ANOMALY_DETECTION
        
        price_jump = abs(data['close'].pct_change().iloc[-1])
        if price_jump > config['price_jump_threshold']:
            return True
            
        recent_volume = data['volume'].iloc[-config['lookback_period']:]
        avg_volume = recent_volume.mean()
        current_volume = data['volume'].iloc[-1]
        
        if current_volume > avg_volume * config['volume_spike_threshold']:
            return True
            
        recent_high = data['high'].iloc[-config['lookback_period']:].max()
        recent_low = data['low'].iloc[-config['lookback_period']:].min()
        current_price = data['close'].iloc[-1]
        
        if current_price > recent_high * 1.05 or current_price < recent_low * 0.95:
            return True
            
        return False

# ============ 信号检查器（v37.0）============
class SignalChecker:
    def __init__(self):
        self.base_thresholds = UltimateConfig.BASE_SIGNAL_THRESHOLDS
        self.params = UltimateConfig.OPTIMIZATION_PARAMS
        self.success_rates = load_success_rates()
        self.recent_signals = deque(maxlen=100)  # 记录最近信号用于时间衰减

    def _get_dynamic_threshold(self, pattern: str, data: pd.DataFrame, price: float) -> int:
        if not UltimateConfig.DYNAMIC_THRESHOLD_ENABLED:
            return self.base_thresholds.get(pattern, 40)
            
        atr = AdvancedIndicators.calculate_atr(data).iloc[-1]
        volatility = atr / price
        factor = max(UltimateConfig.MIN_VOLATILITY_FACTOR, 
                    min(volatility, UltimateConfig.MAX_VOLATILITY_FACTOR))
        base = self.base_thresholds.get(pattern, 40)
        adjusted = int(base * (1 - factor))
        return max(int(base * 0.8), adjusted)

    def _apply_success_rate_weight(self, symbol: str, pattern: str, raw_score: int) -> int:
        rate = self.success_rates.get(symbol, {}).get(pattern, 1.0)
        if rate < 0.5:
            return int(raw_score * 0.8)
        elif rate > 0.8:
            return int(raw_score * 1.05)
        return raw_score

    def _calculate_time_decay(self, symbol: str, direction: str) -> float:
        if not UltimateConfig.TIME_DECAY_ENABLED:
            return 1.0
            
        now = datetime.now()
        decay_factor = 1.0
        
        for signal in self.recent_signals:
            if signal['symbol'] == symbol and signal['direction'] == direction:
                hours_ago = (now - signal['time']).total_seconds() / 3600
                if hours_ago < UltimateConfig.TIME_DECAY_HOURS:
                    decay = 1.0 - (hours_ago / UltimateConfig.TIME_DECAY_HOURS) * (1 - UltimateConfig.TIME_DECAY_MIN_FACTOR)
                    decay_factor = min(decay_factor, decay)
                    
        return decay_factor

    def _check_group_limit(self, symbol: str, all_signals: List[Dict]) -> bool:
        if not UltimateConfig.CORRELATION_CONTROL_ENABLED:
            return True
            
        group = COIN_TO_GROUP.get(symbol, 'OTHERS')
        group_signals = [s for s in all_signals if COIN_TO_GROUP.get(s['symbol'], 'OTHERS') == group]
        
        return len(group_signals) < UltimateConfig.MAX_SIGNALS_PER_GROUP

    def _calculate_position_size(self, score: int, data: pd.DataFrame, price: float) -> float:
        base_size = UltimateConfig.POSITION_SIZE_MIN + (score / 100.0) * (UltimateConfig.POSITION_SIZE_BASE - UltimateConfig.POSITION_SIZE_MIN)
        
        if UltimateConfig.VOLATILITY_POSITION_ADJUST:
            atr = AdvancedIndicators.calculate_atr(data).iloc[-1]
            volatility = atr / price
            vol_factor = 1.0 - min(volatility / UltimateConfig.MAX_VOLATILITY_FACTOR, 0.5)
            base_size *= vol_factor
        
        return round(min(UltimateConfig.POSITION_SIZE_MAX, 
                        max(UltimateConfig.POSITION_SIZE_MIN, base_size)), 2)

    def _find_swing_points(self, data: pd.DataFrame, window: int = 3):
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

    def _detect_divergence_multi_tf(self, data_dict: Dict[str, pd.DataFrame], 
                                   rsi_dict: Dict[str, pd.Series]) -> Tuple[str, float]:
        divergences = []
        strengths = []
        
        for tf, data in data_dict.items():
            if tf not in rsi_dict:
                continue
                
            div_type, strength = self._detect_divergence_single_tf(data, rsi_dict[tf])
            if div_type:
                weight = UltimateConfig.MULTI_TIMEFRAME_WEIGHTS.get(tf, 0.3)
                divergences.append(div_type)
                strengths.append(strength * weight)
        
        if not divergences:
            return None, 0.0
            
        if all(d == divergences[0] for d in divergences):
            bonus = UltimateConfig.DIVERGENCE_CONFIG['multi_tf_bonus']
            avg_strength = sum(strengths) / len(strengths)
            return divergences[0], min(avg_strength * bonus, 1.0)
        
        return None, 0.0

    def _detect_divergence_single_tf(self, data: pd.DataFrame, rsi_series: pd.Series) -> tuple:
        lookback = UltimateConfig.DIVERGENCE_CONFIG['min_lookback']
        if len(data) < lookback:
            return None, 0.0

        sub_data = data.iloc[-lookback:]
        sub_rsi = rsi_series.iloc[-lookback:]
        swing_highs, swing_lows = self._find_swing_points(sub_data, 
                                                         window=UltimateConfig.DIVERGENCE_CONFIG['swing_window'])
        
        base_idx = len(data) - lookback
        swing_highs = [base_idx + i for i in swing_highs]
        swing_lows = [base_idx + i for i in swing_lows]

        w_rsi = UltimateConfig.DIVERGENCE_CONFIG['rsi_weight']
        w_price = UltimateConfig.DIVERGENCE_CONFIG['price_weight']

        # 底背离
        if len(swing_lows) >= 2:
            last_idx = swing_lows[-1]
            prev_idx = swing_lows[-2]
            last_price = data['low'].iloc[last_idx]
            prev_price = data['low'].iloc[prev_idx]
            last_rsi = rsi_series.iloc[last_idx]
            prev_rsi = rsi_series.iloc[prev_idx]

            if last_price < prev_price and last_rsi > prev_rsi:
                x = np.array([prev_idx, last_idx])
                y_price = np.array([prev_price, last_price])
                y_rsi = np.array([prev_rsi, last_rsi])
                slope_price = np.polyfit(x, y_price, 1)[0]
                slope_rsi = np.polyfit(x, y_rsi, 1)[0]
                strength = min(abs(slope_rsi) / (abs(slope_price) + 1e-6), 1.0) * 0.7

                if len(swing_lows) >= 3:
                    prev2_idx = swing_lows[-3]
                    prev2_rsi = rsi_series.iloc[prev2_idx]
                    if last_rsi > prev_rsi > prev2_rsi:
                        strength *= UltimateConfig.DIVERGENCE_CONFIG['consecutive_bonus']
                return 'bullish', min(strength, 1.0)

        # 顶背离
        if len(swing_highs) >= 2:
            last_idx = swing_highs[-1]
            prev_idx = swing_highs[-2]
            last_price = data['high'].iloc[last_idx]
            prev_price = data['high'].iloc[prev_idx]
            last_rsi = rsi_series.iloc[last_idx]
            prev_rsi = rsi_series.iloc[prev_idx]

            if last_price > prev_price and last_rsi < prev_rsi:
                x = np.array([prev_idx, last_idx])
                y_price = np.array([prev_price, last_price])
                y_rsi = np.array([prev_rsi, last_rsi])
                slope_price = np.polyfit(x, y_price, 1)[0]
                slope_rsi = np.polyfit(x, y_rsi, 1)[0]
                strength = min(abs(slope_rsi) / (abs(slope_price) + 1e-6), 1.0) * 0.7

                if len(swing_highs) >= 3:
                    prev2_idx = swing_highs[-3]
                    prev2_rsi = rsi_series.iloc[prev2_idx]
                    if last_rsi < prev_rsi < prev2_rsi:
                        strength *= UltimateConfig.DIVERGENCE_CONFIG['consecutive_bonus']
                return 'bearish', min(strength, 1.0)

        return None, 0.0

    def _detect_macd_cross(self, macd_df: pd.DataFrame, direction: str, lookback: int = 3) -> bool:
        if len(macd_df) < lookback + 1:
            return False
            
        macd = macd_df['macd'].iloc[-lookback:]
        signal = macd_df['signal'].iloc[-lookback:]
        
        if direction == 'SELL':
            for i in range(1, len(macd)):
                if macd.iloc[i-1] > signal.iloc[i-1] and macd.iloc[i] <= signal.iloc[i]:
                    return True
        else:
            for i in range(1, len(macd)):
                if macd.iloc[i-1] < signal.iloc[i-1] and macd.iloc[i] >= signal.iloc[i]:
                    return True
                    
        return False

    def _detect_engulfing(self, data: pd.DataFrame) -> tuple:
        if len(data) < 2:
            return '', 0.0
            
        prev = data.iloc[-2]
        curr = data.iloc[-1]
        
        prev_body = abs(prev['close'] - prev['open'])
        curr_body = abs(curr['close'] - curr['open'])
        prev_open, prev_close = prev['open'], prev['close']
        curr_open, curr_close = curr['open'], curr['close']
        
        min_ratio = UltimateConfig.CANDLE_PATTERNS['engulfing_min_ratio']
        
        if (prev_close < prev_open) and (curr_close > curr_open) and \
           curr_open < prev_close and curr_close > prev_open:
            strength = min(curr_body / prev_body, 2.0) if prev_body > 0 else 1.0
            if strength >= min_ratio:
                return 'BUY', strength
                
        if (prev_close > prev_open) and (curr_close < curr_open) and \
           curr_open > prev_close and curr_close < prev_open:
            strength = min(curr_body / prev_body, 2.0) if prev_body > 0 else 1.0
            if strength >= min_ratio:
                return 'SELL', strength
                
        return '', 0.0

    def _get_combined_trend_mode(self, data_dict: Dict[str, pd.DataFrame]) -> str:
        scores = []
        weights = []
        
        for tf, data in data_dict.items():
            if tf in UltimateConfig.MULTI_TIMEFRAME_WEIGHTS:
                adx = AdvancedIndicators.calculate_adx(data).iloc[-1]
                weight = UltimateConfig.MULTI_TIMEFRAME_WEIGHTS[tf]
                
                if adx <= UltimateConfig.TREND['range_adx']:
                    scores.append(0)  # RANGE
                elif adx <= UltimateConfig.TREND['transition_adx']:
                    scores.append(1)  # TRANSITION
                else:
                    scores.append(2)  # TREND
                weights.append(weight)
        
        if not scores:
            return 'RANGE'
            
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
        
        if weighted_score < 0.5:
            return 'RANGE'
        elif weighted_score < 1.5:
            return 'TRANSITION'
        else:
            return 'TREND'

    def _get_trend_direction(self, data: pd.DataFrame) -> int:
        ema20 = AdvancedIndicators.calculate_ema(data, 20)
        ema50 = AdvancedIndicators.calculate_ema(data, 50)
        atr = AdvancedIndicators.calculate_atr(data).iloc[-1]
        
        if len(ema20) < 4 or len(ema50) < 2:
            return 0
            
        ema20_current = ema20.iloc[-1]
        ema20_prev3 = ema20.iloc[-4]
        slope_per_bar = (ema20_current - ema20_prev3) / ema20_prev3 / 3
        
        ema50_current = ema50.iloc[-1]
        diff = ema20_current - ema50_current
        significant = abs(diff) > atr * UltimateConfig.TREND['ema_structure_threshold']
        
        if significant:
            return 1 if diff > 0 else -1
            
        slope_up = slope_per_bar > UltimateConfig.TREND['min_slope_percent']
        slope_down = slope_per_bar < -UltimateConfig.TREND['min_slope_percent']
        
        if slope_up:
            return 1
        elif slope_down:
            return -1
        else:
            return 0

    def _get_trend_score(self, data: pd.DataFrame, signal_direction: str) -> float:
        trend_dir = self._get_trend_direction(data)
        
        if trend_dir == 0:
            return UltimateConfig.TREND_SCORES['neutral']
            
        if (signal_direction == 'BUY' and trend_dir == 1) or \
           (signal_direction == 'SELL' and trend_dir == -1):
            return UltimateConfig.TREND_SCORES['match']
        else:
            return UltimateConfig.TREND_SCORES['mismatch']

    def _calculate_stop_loss(self, data: pd.DataFrame, price: float, direction: str,
                            trend_direction: int) -> Tuple[float, float, float, float]:
        atr = AdvancedIndicators.calculate_atr(data).iloc[-1]
        adx = AdvancedIndicators.calculate_adx(data).iloc[-1]
        
        if adx > UltimateConfig.TREND['strong_trend_adx'] and trend_direction != 0:
            atr_mult_stop = UltimateConfig.STOP_LOSS['atr_multiplier_strong']
        else:
            atr_mult_stop = UltimateConfig.STOP_LOSS['atr_multiplier']
            
        if direction == 'BUY':
            entry_main = price * 1.002
            recent_low = data['low'].rolling(10).min().iloc[-1]
            stop_loss_candidate1 = recent_low * 0.985
            stop_loss_candidate2 = price - atr * atr_mult_stop
            stop_loss = max(stop_loss_candidate1, stop_loss_candidate2)
            min_stop = price * (1 - UltimateConfig.STOP_LOSS['max_stop_percent'])
            stop_loss = max(stop_loss, min_stop)
            
            tp1 = price + max(atr * UltimateConfig.STOP_LOSS['tp1_multiplier'],
                            price * UltimateConfig.STOP_LOSS['min_tp1_percent'])
            tp2 = price + max(atr * UltimateConfig.STOP_LOSS['tp2_multiplier'],
                            price * UltimateConfig.STOP_LOSS['min_tp2_percent'])
            take_profit1, take_profit2 = tp1, tp2
            
        else:  # SELL
            entry_main = price * 0.998
            recent_high = data['high'].rolling(10).max().iloc[-1]
            stop_loss_candidate1 = recent_high * 1.02
            stop_loss_candidate2 = price + atr * atr_mult_stop
            stop_loss = min(stop_loss_candidate1, stop_loss_candidate2)
            max_stop = price * (1 + UltimateConfig.STOP_LOSS['max_stop_percent'])
            stop_loss = min(stop_loss, max_stop)
            
            tp1 = price - max(atr * UltimateConfig.STOP_LOSS['tp1_multiplier'],
                            price * UltimateConfig.STOP_LOSS['min_tp1_percent'])
            tp2 = price - max(atr * UltimateConfig.STOP_LOSS['tp2_multiplier'],
                            price * UltimateConfig.STOP_LOSS['min_tp2_percent'])
            take_profit1, take_profit2 = tp1, tp2
            
        return entry_main, stop_loss, take_profit1, take_profit2

    def _calculate_stop_loss_structure(self, data: pd.DataFrame, price: float, direction: str) -> Tuple[float, float, float, float]:
        atr = AdvancedIndicators.calculate_atr(data).iloc[-1]
        window = UltimateConfig.TREND_EXHAUSTION['structure_high_window']
        buffer = UltimateConfig.TREND_EXHAUSTION['stop_buffer']

        if direction == 'SELL':
            recent_high = data['high'].rolling(window).max().iloc[-1]
            stop_loss = recent_high * (1 + buffer)
            max_stop_pct = UltimateConfig.STOP_LOSS['max_stop_percent']
            stop_loss = min(stop_loss, price * (1 + max_stop_pct))

            entry_main = price * 0.998
            tp1 = price - max(atr * UltimateConfig.STOP_LOSS['tp1_multiplier'],
                            price * UltimateConfig.STOP_LOSS['min_tp1_percent'])
            tp2 = price - max(atr * UltimateConfig.STOP_LOSS['tp2_multiplier'],
                            price * UltimateConfig.STOP_LOSS['min_tp2_percent'])
            take_profit1, take_profit2 = tp1, tp2
            
        else:
            recent_low = data['low'].rolling(window).min().iloc[-1]
            stop_loss = recent_low * (1 - buffer)
            max_stop_pct = UltimateConfig.STOP_LOSS['max_stop_percent']
            stop_loss = max(stop_loss, price * (1 - max_stop_pct))

            entry_main = price * 1.002
            tp1 = price + max(atr * UltimateConfig.STOP_LOSS['tp1_multiplier'],
                            price * UltimateConfig.STOP_LOSS['min_tp1_percent'])
            tp2 = price + max(atr * UltimateConfig.STOP_LOSS['tp2_multiplier'],
                            price * UltimateConfig.STOP_LOSS['min_tp2_percent'])
            take_profit1, take_profit2 = tp1, tp2

        return entry_main, stop_loss, take_profit1, take_profit2

    def _is_signal_allowed(self, pattern: str, trend_mode: str) -> bool:
        allow_map = UltimateConfig.TREND_SIGNAL_ALLOW
        return pattern in allow_map.get(trend_mode, [])

    def _apply_penalties(self, score: int, rsi: float, volume_ratio: float, data: pd.DataFrame) -> int:
        if rsi > UltimateConfig.RSI_CONFIG['overbought'] or rsi < UltimateConfig.RSI_CONFIG['oversold']:
            score = int(score * UltimateConfig.RSI_CONFIG['extreme_penalty'])
            
        if volume_ratio < UltimateConfig.VOLUME_CONFIG['ultra_low']:
            score -= UltimateConfig.VOLUME_CONFIG['low_penalty']
            
        return max(0, score)

    # ----- 评分函数 -----
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

    def _calculate_confirmation_k_score(self, direction, rsi, volume_ratio, engulf_strength,
                                       div_type, div_strength):
        score = 40
        score += engulf_strength * 20
        
        if div_type == direction.lower():
            score += div_strength * 30
            
        if direction == 'BUY':
            if 40 <= rsi <= 60:
                score += 15
        else:
            if 40 <= rsi <= 60:
                score += 15
                
        score += min(20, volume_ratio * 10)
        return int(score)

    # ----- 信号创建函数 -----
    def _create_bounce_signal(self, symbol, data, price, rsi, volume_ratio, ma20, score,
                              trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, 'BUY', trend_direction
        )
        risk = entry_main - stop_loss
        reward = take_profit2 - entry_main
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        recent_low = data['low'].rolling(20).min().iloc[-1]
        
        position_size = self._calculate_position_size(score, data, price)

        return {
            'symbol': symbol,
            'pattern': 'BOUNCE',
            'direction': 'BUY',
            'rsi': round(float(rsi), 1),
            'volume_ratio': round(volume_ratio, 2),
            'score': int(score),
            'current_price': round(price, 4),
            'signal_time': datetime.now(),
            'reason': f"🟢 <b>超卖反弹机会</b>\n\n• RSI({rsi:.1f})进入超卖\n• 成交量放大{volume_ratio:.1f}倍\n• 价格接近低点${recent_low:.4f}",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            },
            'trend_direction': trend_direction,
            'trend_mode': trend_mode,
            'position_size': position_size
        }

    def _create_callback_signal(self, symbol, data, price, rsi, volume_ratio,
                                recent_high, callback_pct, ma20, score,
                                trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, 'BUY', trend_direction
        )
        risk = entry_main - stop_loss
        reward = take_profit2 - entry_main
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        
        position_size = self._calculate_position_size(score, data, price)

        return {
            'symbol': symbol,
            'pattern': 'CALLBACK',
            'direction': 'BUY',
            'rsi': round(float(rsi), 1),
            'volume_ratio': round(volume_ratio, 2),
            'score': int(score),
            'current_price': round(price, 4),
            'signal_time': datetime.now(),
            'reason': f"🔄 <b>健康回调机会</b>\n\n• 从高点${recent_high:.4f}回调{callback_pct:.1f}%\n• RSI({rsi:.1f})理想\n• 价格在MA20(${ma20:.4f})上方",
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            },
            'trend_direction': trend_direction,
            'trend_mode': trend_mode,
            'position_size': position_size
        }

    def _create_callback_confirm_signal(self, symbol, data, price, rsi,
                                        volume_ratio, recent_high, callback_pct,
                                        ma20, ma50, score,
                                        trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, 'BUY', trend_direction
        )
        risk = entry_main - stop_loss
        reward = take_profit2 - entry_main
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        
        position_size = self._calculate_position_size(score, data, price)

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
                f"• 结构确认: 多周期RSI底背离"
            ),
            'entry_points': {
                'main_entry': round(entry_main, 6),
                'stop_loss': round(stop_loss, 6),
                'take_profit1': round(take_profit1, 6),
                'take_profit2': round(take_profit2, 6),
                'risk_reward': risk_reward
            },
            'trend_direction': trend_direction,
            'trend_mode': trend_mode,
            'position_size': position_size
        }

    def _create_trend_exhaustion_signal(self, symbol, data, price,
                                        rsi, volume_ratio, ma20, score,
                                        trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss_structure(
            data, price, 'SELL'
        )
        risk = stop_loss - entry_main
        reward = entry_main - take_profit2
        risk_reward = round(reward / risk, 2) if risk > 0 else 0
        recent_high = data['high'].rolling(20).max().iloc[-1]

        position_size = self._calculate_position_size(score, data, price)

        reason = (
            f"🔴 <b>趋势衰竭做空</b>\n\n"
            f"• RSI({rsi:.1f})超买\n"
            f"• 成交量萎缩{volume_ratio:.1f}x\n"
            f"• 上一根K线实体缩小或上影线较长\n"
            f"• MACD死叉确认\n"
            f"• 止损设在结构高点${stop_loss:.4f} (+0.15% buffer)"
        )
        
        return {
            'symbol': symbol,
            'pattern': 'TREND_EXHAUSTION',
            'direction': 'SELL',
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
            'trend_mode': trend_mode,
            'position_size': position_size
        }

    def _create_confirmation_k_signal(self, symbol, data, price, rsi, volume_ratio,
                                      ma20, ma50, direction, engulf_strength,
                                      div_type, div_strength, score,
                                      trend_direction, trend_mode):
        entry_main, stop_loss, take_profit1, take_profit2 = self._calculate_stop_loss(
            data, price, direction, trend_direction
        )
        risk = (entry_main - stop_loss) if direction == 'BUY' else (stop_loss - entry_main)
        reward = (take_profit2 - entry_main) if direction == 'BUY' else (entry_main - take_profit2)
        risk_reward = round(reward / risk, 2) if risk > 0 else 0

        div_text = f"• 看涨背离强度: {div_strength:.2f}\n" if div_type == 'bullish' else ""
        
        position_size = self._calculate_position_size(score, data, price)

        reason = (
            f"🟢 <b>看涨吞没形态确认</b>\n\n"
            f"• 吞没强度: {engulf_strength:.2f}\n"
            f"• 成交量{volume_ratio:.1f}倍\n"
            f"• RSI({rsi:.1f})\n"
            f"{div_text}"
            f"• 建议在${entry_main:.4f}附近买入"
        ) if direction == 'BUY' else (
            f"🔴 <b>看跌吞没形态确认</b>\n\n"
            f"• 吞没强度: {engulf_strength:.2f}\n"
            f"• 成交量{volume_ratio:.1f}倍\n"
            f"• RSI({rsi:.1f})\n"
            f"{div_text}"
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
            'trend_mode': trend_mode,
            'position_size': position_size
        }

    def check_all_coins(self, coins_data, cooldown_manager):
        print(f"\n🔍 开始信号扫描 ({len(coins_data)}个币种)...")
        all_signals = []
        signal_counts = defaultdict(int)
        group_signals = defaultdict(list)

        observation_pool = load_observation_pool() if UltimateConfig.OBSERVATION_ENABLED else []
        current_time = datetime.now()

        for symbol, data_dict in coins_data.items():
            try:
                required_tfs = ['15m', '1H', '4H']
                if not all(tf in data_dict for tf in required_tfs):
                    continue
                    
                if AdvancedIndicators.detect_anomalies(data_dict['15m']):
                    if DEBUG:
                        print(f"⚠️ {symbol}: 检测到异常K线，跳过")
                    continue

                data_15m = data_dict['15m']
                data_1h = data_dict['1H']
                data_4h = data_dict['4H']
                
                if any(len(df) < 50 for df in [data_15m, data_1h, data_4h]):
                    continue

                current_price = data_15m['close'].iloc[-1]
                rsi_15m = AdvancedIndicators.calculate_rsi(data_15m, 14)
                rsi_1h = AdvancedIndicators.calculate_rsi(data_1h, 14)
                rsi_4h = AdvancedIndicators.calculate_rsi(data_4h, 14)
                
                current_rsi = rsi_15m.iloc[-1]
                volume_ratio = AdvancedIndicators.calculate_volume_ratio(data_15m, 20).iloc[-1]
                ma20 = AdvancedIndicators.calculate_sma(data_15m, 20).iloc[-1]
                ma50 = AdvancedIndicators.calculate_sma(data_15m, 50).iloc[-1]
                
                trend_mode = self._get_combined_trend_mode(data_dict)
                current_trend_dir = self._get_trend_direction(data_15m)

                signals = []

                # ----- BOUNCE 信号 -----
                bounce_rsi_limit = UltimateConfig.RSI_CONFIG['bounce_limits'].get(trend_mode, 50)
                if current_rsi < bounce_rsi_limit and volume_ratio > UltimateConfig.VOLUME_CONFIG['min_ratio']:
                    if self._is_signal_allowed('BOUNCE', trend_mode):
                        raw_score = self._calculate_bounce_score(current_rsi, volume_ratio)
                        raw_score = self._apply_penalties(raw_score, current_rsi, volume_ratio, data_15m)
                        raw_score = self._apply_success_rate_weight(symbol, 'BOUNCE', raw_score)
                        
                        dynamic_th = self._get_dynamic_threshold('BOUNCE', data_15m, current_price)
                        if raw_score >= dynamic_th:
                            signals.append(self._create_bounce_signal(
                                symbol, data_15m, current_price, current_rsi, volume_ratio, ma20, raw_score,
                                current_trend_dir, trend_mode
                            ))
                            signal_counts['BOUNCE'] += 1

                # ----- CALLBACK 信号 -----
                if current_rsi > self.params['rsi_callback_min']:
                    if self._is_signal_allowed('CALLBACK', trend_mode):
                        recent_high = data_15m['high'].iloc[-30:].max()
                        callback_pct = ((recent_high - current_price) / recent_high) * 100
                        if self.params['callback_pct_min'] <= callback_pct <= self.params['callback_pct_max']:
                            raw_score = self._calculate_callback_score(current_rsi, volume_ratio, callback_pct)
                            raw_score = self._apply_penalties(raw_score, current_rsi, volume_ratio, data_15m)
                            raw_score = self._apply_success_rate_weight(symbol, 'CALLBACK', raw_score)
                            
                            dynamic_th = self._get_dynamic_threshold('CALLBACK', data_15m, current_price)
                            if raw_score >= dynamic_th:
                                signals.append(self._create_callback_signal(
                                    symbol, data_15m, current_price, current_rsi, volume_ratio, recent_high, 
                                    callback_pct, ma20, raw_score, current_trend_dir, trend_mode
                                ))
                                signal_counts['CALLBACK'] += 1

                # ----- CALLBACK_CONFIRM_K 信号 -----
                if UltimateConfig.RSI_CONFIG['oversold'] < current_rsi < UltimateConfig.RSI_CONFIG['overbought']:
                    if self._is_signal_allowed('CALLBACK_CONFIRM_K', trend_mode):
                        lookback = UltimateConfig.VOLUME_CONFIG['lookback_period']
                        recent_volume_avg = data_15m['volume'].iloc[-lookback:].mean()
                        prior_volume_avg = data_15m['volume'].iloc[-lookback*2:-lookback].mean()
                        volume_shrink = recent_volume_avg < prior_volume_avg * UltimateConfig.VOLUME_CONFIG['shrink_threshold']
                        volume_surge = volume_ratio > UltimateConfig.VOLUME_CONFIG['surge_threshold']
                        
                        if volume_shrink and volume_surge:
                            recent_high = data_15m['high'].iloc[-30:].max()
                            callback_pct = ((recent_high - current_price) / recent_high) * 100
                            
                            if 2 <= callback_pct <= 15:
                                recent_3_closes = data_15m['close'].iloc[-3:].values
                                price_increasing = len(recent_3_closes) >= 2 and recent_3_closes[-1] > recent_3_closes[0]
                                
                                if price_increasing and ma20 > ma50 and current_price > ma20:
                                    rsi_dict = {'15m': rsi_15m, '1H': rsi_1h, '4H': rsi_4h}
                                    div_type, div_strength = self._detect_divergence_multi_tf(data_dict, rsi_dict)
                                    
                                    if div_type == 'bullish' and div_strength > 0.2:
                                        raw_score = self._calculate_callback_confirm_score(
                                            current_rsi, volume_ratio, callback_pct)
                                        raw_score = self._apply_penalties(raw_score, current_rsi, volume_ratio, data_15m)
                                        raw_score = self._apply_success_rate_weight(symbol, 'CALLBACK_CONFIRM_K', raw_score)
                                        
                                        dynamic_th = self._get_dynamic_threshold('CALLBACK_CONFIRM_K', data_15m, current_price)
                                        if raw_score >= dynamic_th:
                                            signals.append(self._create_callback_confirm_signal(
                                                symbol, data_15m, current_price, current_rsi, volume_ratio,
                                                recent_high, callback_pct, ma20, ma50, raw_score,
                                                current_trend_dir, trend_mode
                                            ))
                                            signal_counts['CALLBACK_CONFIRM_K'] += 1

                # ----- TREND_EXHAUSTION 信号 -----
                if current_rsi > self.params['trend_exhaustion_rsi_min'] and volume_ratio < 1.0:
                    if self._is_signal_allowed('TREND_EXHAUSTION', trend_mode):
                        trend_dir_1h = self._get_trend_direction(data_1h)
                        if trend_dir_1h == 1:
                            continue

                        rsi_prev = rsi_15m.iloc[-2] if len(rsi_15m) >= 2 else current_rsi
                        rsi_boost = 8 if current_rsi < rsi_prev else 0

                        if len(data_15m) >= 2:
                            curr = data_15m.iloc[-1]
                            prev = data_15m.iloc[-2]
                            curr_body = abs(curr['close'] - curr['open'])
                            prev_body = abs(prev['close'] - prev['open'])
                            curr_upper_shadow = curr['high'] - max(curr['close'], curr['open'])
                            
                            condition1 = curr_body < prev_body
                            condition2 = curr_upper_shadow > curr_body * 1.5
                            
                            if not (condition1 or condition2):
                                continue
                        else:
                            continue

                        if UltimateConfig.TREND_EXHAUSTION['require_macd_cross']:
                            macd_df = AdvancedIndicators.calculate_macd(data_15m)
                            if not self._detect_macd_cross(macd_df, 'SELL', 
                                                          UltimateConfig.TREND_EXHAUSTION['macd_lookback']):
                                continue

                        raw_score = self._calculate_trend_exhaustion_score(current_rsi, volume_ratio)
                        
                        if volume_ratio < UltimateConfig.VOLUME_CONFIG['ultra_low']:
                            raw_score -= UltimateConfig.VOLUME_CONFIG['low_penalty']
                            
                        raw_score = int(raw_score) + rsi_boost
                        raw_score = max(0, raw_score)
                        raw_score = self._apply_success_rate_weight(symbol, 'TREND_EXHAUSTION', raw_score)
                        
                        dynamic_th = self._get_dynamic_threshold('TREND_EXHAUSTION', data_15m, current_price)
                        if raw_score >= dynamic_th:
                            signals.append(self._create_trend_exhaustion_signal(
                                symbol, data_15m, current_price, current_rsi, volume_ratio, ma20, raw_score,
                                current_trend_dir, trend_mode
                            ))
                            signal_counts['TREND_EXHAUSTION'] += 1

                # ----- CONFIRMATION_K 信号 -----
                engulf_dir, engulf_strength = self._detect_engulfing(data_15m)
                if engulf_dir and self._is_signal_allowed('CONFIRMATION_K', trend_mode):
                    rsi_dict = {'15m': rsi_15m, '1H': rsi_1h, '4H': rsi_4h}
                    div_type, div_strength = self._detect_divergence_multi_tf(data_dict, rsi_dict)
                    
                    raw_score = self._calculate_confirmation_k_score(
                        engulf_dir, current_rsi, volume_ratio, engulf_strength,
                        div_type, div_strength
                    )
                    
                    raw_score = self._apply_penalties(raw_score, current_rsi, volume_ratio, data_15m)
                    raw_score = self._apply_success_rate_weight(symbol, 'CONFIRMATION_K', raw_score)
                    
                    dynamic_th = self._get_dynamic_threshold('CONFIRMATION_K', data_15m, current_price)
                    if raw_score >= dynamic_th:
                        signals.append(self._create_confirmation_k_signal(
                            symbol, data_15m, current_price, current_rsi, volume_ratio,
                            ma20, ma50, engulf_dir, engulf_strength, div_type, div_strength, raw_score,
                            current_trend_dir, trend_mode
                        ))
                        signal_counts['CONFIRMATION_K'] += 1

                if signals:
                    best_signal = max(signals, key=lambda x: x.get('score', 0))
                    
                    if not self._check_group_limit(symbol, all_signals):
                        if DEBUG:
                            print(f"⚠️ {symbol}: 板块信号数量超限，跳过")
                        continue
                        
                    time_decay = self._calculate_time_decay(symbol, best_signal['direction'])
                    best_signal['score'] = int(best_signal['score'] * time_decay)
                    best_signal['position_size'] *= time_decay
                    
                    all_signals.append(best_signal)
                    
                    self.recent_signals.append({
                        'symbol': symbol,
                        'direction': best_signal['direction'],
                        'time': current_time
                    })

            except Exception as e:
                if DEBUG:
                    print(f"⚠️ 处理 {symbol} 时出错: {e}")
                log_performance(f"Error processing {symbol}: {str(e)}")
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
                    current_trend_dir = self._get_trend_direction(data_15m)
                    direction = obs['direction']
                    
                    if (direction == 'BUY' and current_trend_dir == 1) or \
                       (direction == 'SELL' and current_trend_dir == -1):
                        new_score = obs['score'] + UltimateConfig.OBSERVATION_SCORE_BOOST
                        
                        if new_score >= UltimateConfig.HIGH_CONFIDENCE_THRESHOLD:
                            signal = obs['signal']
                            signal['score'] = new_score
                            signal['signal_time'] = current_time
                            signal['reason'] += "\n• 延迟1根K线确认趋势后增强"
                            signal['position_size'] = self._calculate_position_size(new_score, data_15m, signal['current_price'])
                            
                            if self._check_group_limit(symbol, all_signals):
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

    def send_signal(self, signal):
        if not self.bot:
            print(f"\n📨 [模拟发送] {signal['symbol']} - {signal['pattern']} ({signal['score']}分)")
            return True
            
        if signal['score'] < UltimateConfig.HIGH_CONFIDENCE_THRESHOLD:
            print(f"📝 信号 {signal['symbol']} 分数 {signal['score']} 低于高置信度阈值，仅记录不发送")
            return False
            
        message = self._format_signal_message(signal)
        
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
                    
        return False

    def _format_signal_message(self, signal):
        direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
        pattern_emoji = {
            'BOUNCE': '🔺',
            'CALLBACK': '🔄',
            'CALLBACK_CONFIRM_K': '🚀',
            'CONFIRMATION_K': '🔰',
            'TREND_EXHAUSTION': '📉'
        }.get(signal['pattern'], '💰')
        
        entry = signal['entry_points']
        confidence = "🔥 高置信度" if signal['score'] >= 80 else "⚠️ 中等置信度"
        position_pct = int(signal['position_size'] * 100)
        
        group = COIN_TO_GROUP.get(signal['symbol'], '其他')
        
        return f"""
        
 <b>🚀 交易信号</b>  {confidence}

<b>🎯 交易对:</b> {signal['symbol']}/USDT
<b>📊 板块:</b> {group}
<b>📊 模式:</b> {signal['pattern']} {pattern_emoji}
<b>📈 方向:</b> {signal['direction']} {direction_emoji}
<b>⭐ 评分:</b> {signal['score']}/100
<b>💼 建议仓位:</b> {position_pct}%
<b>📉 RSI:</b> {signal['rsi']}
<b>📊 成交量:</b> {signal['volume_ratio']:.1f}x

<b>💰 当前价格:</b> ${signal['current_price']:.4f}
<code>───────────────────────────</code>

<b>🎯 入场:</b> ${entry['main_entry']:.4f}
<b>🛑 止损:</b> ${entry['stop_loss']:.4f}
<b>🎯 止盈:</b> ${entry['take_profit2']:.4f}
<b>⚖️ 盈亏比:</b> {entry['risk_reward']}:1

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
        print("\n" + "=" * 70)
        print(f"🤖 终极智能交易系统 {UltimateConfig.VERSION}")
        print("=" * 70)
        
        self.data_fetcher = MultiSourceDataFetcher()
        self.cooldown_manager = CooldownManager()
        self.signal_checker = SignalChecker()
        self.telegram = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
        
        self.cycle_count = 0
        self.total_signals = 0
        self.start_time = datetime.now()
        
        print(f"\n✅ 系统初始化完成")
        print(f"📡 监控币种: {len(MONITOR_COINS)}个")
        print(f"📊 板块数量: {len(set(COIN_TO_GROUP.values()))}个")
        print(f"🤖 Telegram: {'✅ 已启用' if self.telegram.bot else '⚠️ 已禁用'}")
        print(f"🔧 回测模式: {'✅ 开启' if BACKTEST_MODE else '❌ 关闭'}")
        if DEBUG:
            print("🔧 调试模式: 已启用")
        print("=" * 70)

    def run_analysis(self):
        self.cycle_count += 1
        print(f"\n🔄 第 {self.cycle_count} 次实时分析开始...")
        print(f"⏰ 时间: {datetime.now().strftime('%H:%M:%S')}")

        try:
            intervals = ['15m', '1H', '4H']
            coins_data = self.data_fetcher.get_all_coins_data(MONITOR_COINS, intervals)
            
            if not coins_data or len(coins_data) < 10:
                print("❌ 数据获取失败或数据不足")
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
            log_performance(f"Critical error: {str(e)}")
            return []

    def _process_signals(self, signals):
        print(f"\n📨 准备发送 {len(signals)} 个交易信号...")
        
        signals.sort(key=lambda x: x.get('score', 0), reverse=True)
        max_to_send = min(UltimateConfig.MAX_SIGNALS_TO_SEND, len(signals))
        top_signals = signals[:max_to_send]

        sent_count = 0
        for i, signal in enumerate(top_signals, 1):
            symbol = signal['symbol']
            pattern = signal['pattern']
            score = signal['score']
            direction = signal['direction']
            trend_dir = signal['trend_direction']
            trend_mode = signal['trend_mode']
            
            print(f"\n[{i}] {symbol} {direction}: {pattern} ({score}分)")

            cooldown_ok, cooldown_reason = self.cooldown_manager.check_cooldown(
                symbol, direction, trend_dir, trend_mode, score
            )
            
            if not cooldown_ok:
                print(f"   ⚠️ 冷却阻止: {cooldown_reason}")
                continue

            success = self.telegram.send_signal(signal)
            
            if success:
                self.cooldown_manager.record_signal(symbol, direction, pattern, score, trend_dir, trend_mode)
                self.total_signals += 1
                sent_count += 1
                time.sleep(2)
                
                log_performance(f"Signal sent: {symbol} {pattern} {score}")

        print(f"\n✅ 本次成功发送 {sent_count} 个交易信号")

# ============ 主程序入口 ============
def main():
    print("=" * 70)
    print("🤖 终极智能交易系统 v37.0 - 全面优化版")
    print("=" * 70)
    print(f"📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 监控币种: {len(MONITOR_COINS)}个")
    print(f"🎯 信号模式: 动态分层 + 结构确认 + 多周期背离 + 板块风控")
    print("=" * 70)

    try:
        system = UltimateTradingSystem()
        
        # 简单模式：直接运行实时分析（忽略回测）
        print("\n🎯 运行实时分析...")
        signals = system.run_analysis()

        if signals:
            print(f"\n✅ 分析完成！发现 {len(signals)} 个交易信号")
        else:
            print("\n📊 本次分析未发现信号")

        print("\n🏁 运行完成。")
        
    except KeyboardInterrupt:
        print("\n\n🛑 系统被用户停止")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 系统运行失败: {e}")
        traceback.print_exc()
        log_performance(f"Fatal error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
