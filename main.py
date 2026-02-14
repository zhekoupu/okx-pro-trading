#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极智能交易系统 v34.0 · 动态参数增强版（全策略ATR/ADX适配）
================================================================
✅ 9种策略全部集成ATR动态止损止盈
✅ ADX市场状态过滤，避免逆势交易
✅ 内置回测引擎（最长2星期）
✅ 完整保留Telegram通知功能
✅ 适配GitHub Actions自动运行（单次分析后退出）
================================================================
"""

# ============ 自动安装依赖 ============
import subprocess
import sys
import os
import atexit

def install_packages():
    required_packages = ['pandas', 'numpy', 'requests', 'pyTelegramBotAPI']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"🔧 正在安装 {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} 安装完成")

print("🔧 检查并安装依赖...")
install_packages()

# ============ 导入库 ============
import pandas as pd
import numpy as np
import telebot
import time
import traceback
import requests
import pickle
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict, deque

# ============ 从环境变量读取敏感信息 ============
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "8455563588:AAERqF8wtcQUOojByNPPpbb0oJG-7VMpr9s")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "2004655568")
IN_GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS') == 'true'   # 检测是否在GitHub Actions中运行

OKX_API_BASE_URL = "https://www.okx.com"
OKX_CANDLE_INTERVAL = ["15m", "1H"]
OKX_CANDLE_LIMIT = 200

# 监控币种列表
MONITOR_COINS = [
    'BTC', 'ETH', 'BNB', 'XRP', 'SOL', 'ADA', 'AVAX', 'DOT',
    'DOGE', 'LTC', 'UNI', 'LINK', 'ATOM', 'XLM', 'ALGO',
    'FIL', 'TRX', 'ETC', 'XTZ', 'AAVE', 'COMP', 'YFI',
    'SUSHI', 'SNX', 'CRV', '1INCH', 'NEAR', 'GRT', 'SAND',
    'MANA', 'ENJ', 'CHZ', 'BAT', 'ZIL', 'ONE', 'IOTA',
    'DASH', 'ZEC', 'EGLD', 'CRO', 'KSM', 'DYDX', 'JUP',
    'STORJ', 'SKL', 'HYPE', 'WLD', 'ARB', 'OP', 'LDO',
    'APT', 'SUI', 'SEI', 'INJ', 'FET', 'THETA', 'AR',
    'ENS', 'PEPE', 'SHIB', 'APE', 'LIT', 'GALA', 'IMX', 'AXS'
]

# ============ 增强的系统配置类 ============
class UltimateConfig:
    VERSION = "34.0-全策略ATR/ADX适配版"
    ANALYSIS_INTERVAL = 45
    COINS_TO_MONITOR = len(MONITOR_COINS)
    MAX_SIGNALS = 10

    COOLDOWN_CONFIG = {
        'same_coin_cooldown': 120,
        'same_direction_cooldown': 60,
        'max_signals_per_coin_per_day': 3,
        'enable_cooldown': True
    }

    MULTI_TIMEFRAME_CONFIG = {
        'enabled': True,
        'timeframes': ['15m', '1H'],
        'consensus_threshold': 0.6,
        'weight_15m': 1.0,
        'weight_1H': 1.2
    }

    OKX_CONFIG = {
        'base_url': OKX_API_BASE_URL,
        'candle_endpoint': '/api/v5/market/candles',
        'intervals': OKX_CANDLE_INTERVAL,
        'limit': OKX_CANDLE_LIMIT,
        'rate_limit': 30,
        'retry_times': 3,
        'timeout': 20
    }

    # ----- ATR动态止损参数 -----
    ATR_CONFIG = {
        'period': 14,
        'stop_loss_multiplier': 1.5,
        'take_profit_multiplier': 3.0,
        'trailing_activation': 1.0,
        'trailing_stop': 1.0
    }

    # ----- ADX市场状态过滤 -----
    ADX_CONFIG = {
        'period': 14,
        'trend_threshold': 25,
        'enabled': True
    }

    # ----- 回测配置 -----
    BACKTEST_CONFIG = {
        'enabled': False,
        'start_date': (datetime.now() - timedelta(days=14)).strftime('%Y-%m-%d'),
        'end_date': datetime.now().strftime('%Y-%m-%d'),
        'symbols': ['BTC', 'ETH', 'SOL'],
        'interval': '15m',
        'initial_capital': 10000,
        'commission': 0.0005,
        'slippage': 0.0001
    }

    RISK_CONFIG = {
        'base_risk': 1.7,
        'position_size': {'min': 70, 'max': 95},
        'stop_loss': {'min': 0.5, 'max': 3.0},
        'take_profit': 'technical',
        'risk_reward': {'min': 2.2, 'max': 8.0},
        'short_config': {
            'max_position_size': 40,
            'stop_loss_tight': 0.6,
            'rsi_threshold': 65
        }
    }

    MARKET_MODES = {
        'BOUNCE': {
            'name': '反弹模式',
            'enabled': True,
            'conditions': {'max_rsi': 42, 'min_volume_ratio': 0.7, 'min_score': 35, 'risk_reward': 2.5}
        },
        'BREAKOUT': {
            'name': '突破模式',
            'enabled': True,
            'conditions': {'min_rsi': 45, 'max_rsi': 68, 'min_volume_ratio': 1.2, 'min_score': 30, 'risk_reward': 2.5}
        },
        'BREAKOUT_FAIL_SHORT': {
            'name': '突破失败做空',
            'enabled': True,
            'conditions': {'min_rsi': 65, 'breakout_failure_threshold': 0.98, 'min_score': 35, 'risk_reward': 2.2}
        },
        'TREND': {
            'name': '趋势模式',
            'enabled': True,
            'conditions': {'min_rsi': 40, 'max_rsi': 75, 'min_volume_ratio': 1.0, 'min_score': 35, 'risk_reward': 2.5}
        },
        'CALLBACK': {
            'name': '回调模式',
            'enabled': True,
            'conditions': {'min_rsi': 55, 'callback_range': {'min': 5, 'max': 15}, 'min_score': 40, 'risk_reward': 2.5}
        },
        'BOUNCE_FAIL_SHORT': {
            'name': '反弹失败做空',
            'enabled': True,
            'conditions': {'min_score': 45, 'max_bounce_pct': 2.0, 'lookback_periods': 10, 'fib_threshold': 38.2, 'risk_reward': 2.2}
        },
        'TREND_EXHAUSTION': {
            'name': '趋势衰竭做空',
            'enabled': True,
            'conditions': {'min_score': 55, 'trend_periods': 30, 'exhaustion_threshold': 0.6,
                           'volume_divergence_threshold': 0.7, 'required_confirmation': 3, 'risk_reward': 2.0}
        },
        'BOUNCE_FAIL_CONFIRM_K': {
            'name': '反弹失败·确认K做空',
            'enabled': True,
            'conditions': {
                'min_score': 50, 'max_bounce_count': 1, 'min_entity_ratio': 0.6,
                'max_lower_shadow_ratio': 0.2, 'required_confirmation': 2,
                'volume_requirement': 0.8, 'risk_reward': 2.5
            }
        },
        'CALLBACK_CONFIRM_K': {
            'name': '回调企稳·确认K做多',
            'enabled': True,
            'conditions': {
                'min_score': 50, 'max_callback_count': 1, 'min_entity_ratio': 0.6,
                'max_upper_shadow_ratio': 0.2, 'required_confirmation': 2,
                'volume_requirement': 0.8, 'risk_reward': 2.5
            }
        }
    }

    TELEGRAM_CONFIG = {
        'enabled': True,
        'parse_mode': 'HTML',
        'show_emoji': True,
        'show_details': True,
        'include_entry_exit': True,
        'include_structure_levels': True
    }

# ============ 技术指标增强（ATR/ADX） ============
class TechnicalIndicatorsMultiTF:
    @staticmethod
    def calculate_rsi(data: pd.DataFrame, period: int = 14):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_ma(data: pd.DataFrame, period: int):
        return data['close'].rolling(window=period).mean()

    @staticmethod
    def calculate_volume_ratio(data: pd.DataFrame, period: int = 20):
        current_volume = data['volume']
        avg_volume = data['volume'].rolling(window=period).mean()
        volume_ratio = current_volume / avg_volume
        return volume_ratio.fillna(0)

    @staticmethod
    def calculate_structure_levels(data: pd.DataFrame):
        if len(data) < 20:
            return {}
        recent_high = data['high'].rolling(window=20).max().iloc[-1]
        recent_low = data['low'].rolling(window=20).min().iloc[-1]
        ma20 = data['close'].rolling(window=20).mean().iloc[-1]
        ma50 = data['close'].rolling(window=50).mean().iloc[-1]
        pivot = (data['high'].iloc[-1] + data['low'].iloc[-1] + data['close'].iloc[-1]) / 3
        r1 = 2 * pivot - data['low'].iloc[-1]
        s1 = 2 * pivot - data['high'].iloc[-1]
        return {
            'ma20': ma20, 'ma50': ma50, 'recent_high': recent_high,
            'recent_low': recent_low, 'pivot': pivot, 'r1': r1, 's1': s1
        }

    @staticmethod
    def calculate_fibonacci_levels(high, low):
        diff = high - low
        return {
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50.0%': high - diff * 0.5,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786
        }

    @staticmethod
    def calculate_bollinger_bands(data: pd.DataFrame, window: int = 20, num_std: int = 2):
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        middle_band = rolling_mean
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, middle_band, lower_band

    @staticmethod
    def calculate_macd(data: pd.DataFrame):
        exp1 = data['close'].ewm(span=12, adjust=False).mean()
        exp2 = data['close'].ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd - signal

    @staticmethod
    def calculate_atr(data: pd.DataFrame, period: int = 14):
        high, low, close = data['high'], data['low'], data['close']
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    @staticmethod
    def calculate_adx(data: pd.DataFrame, period: int = 14):
        high, low, close = data['high'], data['low'], data['close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
        dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
        adx = dx.ewm(alpha=1/period, adjust=False).mean()
        return adx, plus_di, minus_di

    @staticmethod
    def get_multi_timeframe_consensus(indicators_dict: Dict[str, Dict], direction: str = 'BUY') -> Tuple[float, Dict]:
        if not UltimateConfig.MULTI_TIMEFRAME_CONFIG['enabled']:
            return 1.0, {}
        timeframes = UltimateConfig.MULTI_TIMEFRAME_CONFIG['timeframes']
        weights = {'15m': 1.0, '1H': 1.2}
        valid_indicators = {}
        for tf in timeframes:
            if tf in indicators_dict and indicators_dict[tf]:
                valid_indicators[tf] = indicators_dict[tf]
        if not valid_indicators:
            return 0.0, {}
        total_weight = sum(weights.get(tf, 1.0) for tf in valid_indicators.keys())
        weighted_score = 0
        consensus_details = {}
        for tf, indicators in valid_indicators.items():
            tf_weight = weights.get(tf, 1.0)
            if direction == 'BUY':
                rsi = indicators.get('rsi', 50)
                volume_ratio = indicators.get('volume_ratio', 1)
                rsi_score = max(0, (40 - max(20, rsi)) / 20)
                volume_score = min(1, volume_ratio / 2)
                tf_score = (rsi_score + volume_score) / 2
            else:
                rsi = indicators.get('rsi', 50)
                volume_ratio = indicators.get('volume_ratio', 1)
                rsi_score = max(0, (min(80, rsi) - 60) / 20)
                volume_score = min(1, volume_ratio / 2)
                tf_score = (rsi_score + volume_score) / 2
            weighted_score += tf_score * tf_weight
            consensus_details[tf] = {
                'score': round(tf_score * 100, 1),
                'rsi': round(rsi, 1),
                'volume_ratio': round(volume_ratio, 2)
            }
        consensus = weighted_score / total_weight
        return consensus, consensus_details

# ============ 冷却管理器（适配 GitHub Actions） ============
class CooldownManager:
    def __init__(self):
        self.config = UltimateConfig.COOLDOWN_CONFIG
        self.cooldown_db = {}
        self.signal_history = defaultdict(list)
        self.cooldown_file = 'cooldown_state.pkl'
        self.use_persistence = not IN_GITHUB_ACTIONS   # GitHub Actions 中禁用持久化
        if self.use_persistence:
            self.load_state()
            atexit.register(self.save_state)
        else:
            print("⚙️ GitHub Actions 模式: 禁用冷却持久化")

    def load_state(self):
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
        if not self.use_persistence:
            return
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
        if not self.config['enable_cooldown']:
            return True, ""
        now = datetime.now()
        if symbol in self.cooldown_db:
            last_signal_time = self.cooldown_db[symbol]['time']
            cooldown_minutes = self.config['same_coin_cooldown']
            if (now - last_signal_time).total_seconds() / 60 < cooldown_minutes:
                remaining = cooldown_minutes - (now - last_signal_time).total_seconds() / 60
                return False, f"同币种冷却中 ({remaining:.1f}分钟)"
        if symbol in self.cooldown_db:
            last_direction = self.cooldown_db[symbol]['direction']
            if last_direction == direction:
                last_signal_time = self.cooldown_db[symbol]['time']
                cooldown_minutes = self.config['same_direction_cooldown']
                if (now - last_signal_time).total_seconds() / 60 < cooldown_minutes:
                    remaining = cooldown_minutes - (now - last_signal_time).total_seconds() / 60
                    return False, f"同方向冷却中 ({remaining:.1f}分钟)"
        today_str = now.strftime('%Y-%m-%d')
        today_signals = [s for s in self.signal_history[symbol] if s['date'] == today_str]
        if len(today_signals) >= self.config['max_signals_per_coin_per_day']:
            return False, f"今日已达最大信号数 ({len(today_signals)}个)"
        return True, ""

    def record_signal(self, symbol: str, direction: str, pattern: str, score: int):
        now = datetime.now()
        self.cooldown_db[symbol] = {
            'time': now,
            'direction': direction,
            'pattern': pattern,
            'score': score
        }
        self.signal_history[symbol].append({
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'direction': direction,
            'pattern': pattern,
            'score': score
        })
        cutoff_date = (now - timedelta(days=30)).strftime('%Y-%m-%d')
        self.signal_history[symbol] = [
            s for s in self.signal_history[symbol] if s['date'] >= cutoff_date
        ]
        expired_keys = []
        for key, record in self.cooldown_db.items():
            if (now - record['time']).total_seconds() / 3600 > 24:
                expired_keys.append(key)
        for key in expired_keys:
            del self.cooldown_db[key]

    def get_cooldown_status(self, symbol: str = None):
        if symbol:
            if symbol in self.cooldown_db:
                record = self.cooldown_db[symbol]
                elapsed = (datetime.now() - record['time']).total_seconds() / 60
                remaining = self.config['same_coin_cooldown'] - elapsed
                return {
                    'symbol': symbol,
                    'direction': record['direction'],
                    'pattern': record['pattern'],
                    'elapsed_minutes': round(elapsed, 1),
                    'remaining_minutes': round(max(0, remaining), 1),
                    'score': record['score']
                }
            return None
        status = {}
        now = datetime.now()
        for sym, record in self.cooldown_db.items():
            elapsed = (now - record['time']).total_seconds() / 60
            remaining = self.config['same_coin_cooldown'] - elapsed
            status[sym] = {
                'direction': record['direction'],
                'pattern': record['pattern'],
                'elapsed': round(elapsed, 1),
                'remaining': round(max(0, remaining), 1)
            }
        return status

# ============ OKX数据获取器 ============
class OKXDataFetcher:
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
        self.cache_duration = 300
        print("✅ OKX数据获取器初始化")

    def get_candles(self, symbol: str, interval: str, limit: int = None, before: str = None):
        if limit is None:
            limit = self.limit
        cache_key = f"{symbol}_{interval}_{limit}_{before}"
        current_time = time.time()
        if cache_key in self.cache:
            if current_time - self.cache_time.get(cache_key, 0) < self.cache_duration:
                return self.cache[cache_key]

        inst_id = f"{symbol}-USDT"
        params = {'instId': inst_id, 'bar': interval, 'limit': limit}
        if before:
            params['before'] = before

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
                            df['timestamp'] = pd.to_datetime(df['timestamp'].astype(np.int64), unit='ms')
                            for col in ['open', 'high', 'low', 'close', 'volume']:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                            df.set_index('timestamp', inplace=True)
                            df.sort_index(inplace=True)
                            self.cache[cache_key] = df
                            self.cache_time[cache_key] = current_time
                            return df
                else:
                    if retry == self.retry_times - 1:
                        print(f"⚠️  {symbol} ({interval}): 请求失败 {response.status_code}")
            except Exception as e:
                if retry < self.retry_times - 1:
                    time.sleep(1)
                else:
                    print(f"⚠️  {symbol} ({interval}): 请求异常 {str(e)}")
        return None

    def get_multi_timeframe_data(self, symbol: str) -> Dict[str, pd.DataFrame]:
        data_dict = {}
        for interval in self.intervals:
            df = self.get_candles(symbol, interval)
            if df is not None and len(df) >= 30:
                data_dict[interval] = df
        return data_dict if data_dict else None

    def get_all_coins_data(self, symbols: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        print(f"\n📡 开始获取 {len(symbols)} 个币种的实时数据...")
        print(f"📊 多周期分析: {', '.join(self.intervals)}")
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
                period_counts = {k: len(v) for k, v in data_dict.items()}
                print(f"[{i}/{total}] {symbol}: ✅ 成功 ({period_counts})")
            else:
                print(f"[{i}/{total}] {symbol}: ⚠️ 数据不足")
            if i % 10 == 0:
                time.sleep(1)
        print(f"\n📊 数据获取完成: {len(coins_data)}/{total} 个币种")
        return coins_data

    def get_historical_candles(self, symbol: str, interval: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        all_dfs = []
        current_end = None
        target_start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        target_end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

        while True:
            df = self.get_candles(symbol, interval, limit=100, before=current_end)
            if df is None or len(df) == 0:
                break
            df_filtered = df[(df.index >= pd.to_datetime(target_start_ts, unit='ms')) &
                             (df.index <= pd.to_datetime(target_end_ts, unit='ms'))]
            if not df_filtered.empty:
                all_dfs.append(df_filtered)
            earliest_ts = df.index.min().value // 10**6
            if earliest_ts <= target_start_ts:
                break
            current_end = str(earliest_ts)
            time.sleep(0.2)

        if all_dfs:
            full_df = pd.concat(all_dfs)
            full_df = full_df.sort_index()
            return full_df
        return None

# ============ K线分析工具类 ============
class KLineAnalyzer:
    @staticmethod
    def analyze_candle(kline_data: pd.Series) -> Dict:
        try:
            open_price = float(kline_data['open'])
            high_price = float(kline_data['high'])
            low_price = float(kline_data['low'])
            close_price = float(kline_data['close'])
            is_bullish = close_price > open_price
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            if total_range == 0:
                return {'is_bullish': is_bullish, 'entity_ratio': 0, 'upper_shadow_ratio': 0, 'lower_shadow_ratio': 0, 'is_doji': True}
            entity_ratio = body_size / total_range if total_range > 0 else 0
            if is_bullish:
                upper_shadow = high_price - close_price
                lower_shadow = open_price - low_price
            else:
                upper_shadow = high_price - open_price
                lower_shadow = close_price - low_price
            upper_shadow_ratio = upper_shadow / total_range if total_range > 0 else 0
            lower_shadow_ratio = lower_shadow / total_range if total_range > 0 else 0
            is_doji = entity_ratio < 0.1
            is_long_upper_shadow = upper_shadow_ratio > 0.3
            is_long_lower_shadow = lower_shadow_ratio > 0.3
            return {
                'is_bullish': is_bullish, 'entity_ratio': entity_ratio,
                'upper_shadow_ratio': upper_shadow_ratio, 'lower_shadow_ratio': lower_shadow_ratio,
                'is_doji': is_doji, 'is_long_upper_shadow': is_long_upper_shadow,
                'is_long_lower_shadow': is_long_lower_shadow, 'body_size': body_size,
                'total_range': total_range, 'open': open_price, 'high': high_price,
                'low': low_price, 'close': close_price, 'volume': float(kline_data['volume'])
            }
        except:
            return None

    @staticmethod
    def is_confirmation_candle(kline_analysis: Dict, direction: str, config: Dict) -> bool:
        if not kline_analysis:
            return False
        if direction == 'SELL':
            if kline_analysis['is_bullish']:
                return False
        else:
            if not kline_analysis['is_bullish']:
                return False
        if kline_analysis['entity_ratio'] < config.get('min_entity_ratio', 0.6):
            return False
        if direction == 'SELL':
            if kline_analysis['lower_shadow_ratio'] > config.get('max_lower_shadow_ratio', 0.2):
                return False
        else:
            if kline_analysis['upper_shadow_ratio'] > config.get('max_upper_shadow_ratio', 0.2):
                return False
        if kline_analysis['is_doji']:
            return False
        return True

# ============ 基础信号检查器（增强ATR/ADX支持） ============
class BaseSignalChecker:
    def __init__(self, pattern_name: str):
        self.pattern_name = pattern_name
        self.config = UltimateConfig.MARKET_MODES.get(pattern_name, {}).get('conditions', {})
        self.risk_reward = UltimateConfig.MARKET_MODES.get(pattern_name, {}).get('conditions', {}).get('risk_reward', 2.5)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        raise NotImplementedError

    def calculate_structure_levels_multi_tf(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        structure_levels = {}
        for tf, data in data_dict.items():
            if len(data) >= 20:
                levels = TechnicalIndicatorsMultiTF.calculate_structure_levels(data)
                if levels:
                    structure_levels[tf] = levels
        merged_levels = {}
        if structure_levels:
            for tf in ['1H', '15m']:
                if tf in structure_levels:
                    merged_levels = structure_levels[tf]
                    merged_levels['timeframe'] = tf
                    break
        return merged_levels

    def get_multi_timeframe_indicators(self, data_dict: Dict[str, pd.DataFrame]) -> Dict[str, Dict]:
        indicators = {}
        for tf, data in data_dict.items():
            if len(data) < 30:
                continue
            rsi = TechnicalIndicatorsMultiTF.calculate_rsi(data, 14)
            volume_ratio = TechnicalIndicatorsMultiTF.calculate_volume_ratio(data, 20)
            ma20 = TechnicalIndicatorsMultiTF.calculate_ma(data, 20)
            ma50 = TechnicalIndicatorsMultiTF.calculate_ma(data, 50)
            indicators[tf] = {
                'rsi': rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50,
                'volume_ratio': volume_ratio.iloc[-1] if not pd.isna(volume_ratio.iloc[-1]) else 1,
                'ma20': ma20.iloc[-1] if not pd.isna(ma20.iloc[-1]) else 0,
                'ma50': ma50.iloc[-1] if not pd.isna(ma50.iloc[-1]) else 0
            }
        return indicators

    def get_atr(self, data: pd.DataFrame, period: int = None) -> Optional[float]:
        if period is None:
            period = UltimateConfig.ATR_CONFIG['period']
        atr_series = TechnicalIndicatorsMultiTF.calculate_atr(data, period)
        if atr_series is not None and len(atr_series) > 0 and not pd.isna(atr_series.iloc[-1]):
            return atr_series.iloc[-1]
        return None

    def calculate_dynamic_stop_loss(self, entry_price: float, direction: str, atr: float) -> float:
        multiplier = UltimateConfig.ATR_CONFIG['stop_loss_multiplier']
        if direction.upper() == 'BUY':
            stop = entry_price - atr * multiplier
        else:
            stop = entry_price + atr * multiplier
        return round(stop, 6)

    def calculate_dynamic_take_profit(self, entry_price: float, direction: str, atr: float, risk_reward: float = None) -> float:
        if risk_reward is None:
            risk_reward = self.risk_reward
        multiplier = UltimateConfig.ATR_CONFIG['take_profit_multiplier']
        if risk_reward > 0:
            if direction.upper() == 'BUY':
                tp = entry_price + (entry_price - self.calculate_dynamic_stop_loss(entry_price, direction, atr)) * risk_reward
            else:
                tp = entry_price - (self.calculate_dynamic_stop_loss(entry_price, direction, atr) - entry_price) * risk_reward
        else:
            if direction.upper() == 'BUY':
                tp = entry_price + atr * multiplier
            else:
                tp = entry_price - atr * multiplier
        return round(tp, 6)

    @staticmethod
    def get_market_state(data: pd.DataFrame) -> Dict:
        adx_config = UltimateConfig.ADX_CONFIG
        if not adx_config['enabled'] or len(data) < adx_config['period'] + 10:
            return {'trend_strength': 'RANGING', 'trend_direction': None,
                    'adx_value': 0, 'plus_di': 0, 'minus_di': 0}
        adx, plus_di, minus_di = TechnicalIndicatorsMultiTF.calculate_adx(data, adx_config['period'])
        if adx is None or plus_di is None or minus_di is None:
            return {'trend_strength': 'RANGING', 'trend_direction': None,
                    'adx_value': 0, 'plus_di': 0, 'minus_di': 0}
        adx_val = adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0
        pdi = plus_di.iloc[-1] if not pd.isna(plus_di.iloc[-1]) else 0
        mdi = minus_di.iloc[-1] if not pd.isna(minus_di.iloc[-1]) else 0
        threshold = adx_config['trend_threshold']
        if adx_val >= threshold:
            strength = 'STRONG_TREND'
            direction = 'UP' if pdi > mdi else 'DOWN'
        else:
            strength = 'RANGING'
            direction = None
        return {
            'trend_strength': strength,
            'trend_direction': direction,
            'adx_value': round(adx_val, 1),
            'plus_di': round(pdi, 1),
            'minus_di': round(mdi, 1)
        }

    def is_market_allowed(self, market_state: Dict, strategy_direction: str) -> bool:
        if market_state['trend_strength'] != 'STRONG_TREND':
            return True
        if strategy_direction.upper() == 'BUY' and market_state['trend_direction'] == 'DOWN':
            return False
        if strategy_direction.upper() == 'SELL' and market_state['trend_direction'] == 'UP':
            return False
        return True

# ============ 策略具体实现（全9种策略ATR/ADX改造） ============

# ---------- 1. BOUNCE 反弹模式 ----------
class BounceSignalChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('BOUNCE')
        self.min_score = self.config.get('min_score', 35)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data = data_dict['15m']
            if len(data) < 50:
                return None

            # ADX过滤
            market_state = self.get_market_state(data)
            if not self.is_market_allowed(market_state, 'BUY'):
                return None

            # 获取ATR
            atr = self.get_atr(data)
            if atr is None or atr <= 0:
                return None

            # 指标检查
            indicators = self.get_multi_timeframe_indicators({'15m': data})['15m']
            current_rsi = indicators['rsi']
            if pd.isna(current_rsi) or current_rsi > self.config.get('max_rsi', 42):
                return None
            current_volume_ratio = indicators['volume_ratio']
            if pd.isna(current_volume_ratio) or current_volume_ratio < self.config.get('min_volume_ratio', 0.7):
                return None

            # 多时间框架共识
            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(
                self.get_multi_timeframe_indicators(data_dict), 'BUY')
            if consensus < 0.6:
                return None

            # 评分计算
            score = 45
            if current_rsi < 30:
                score += 25
            elif current_rsi < 35:
                score += 20
            elif current_rsi < 40:
                score += 15
            if current_volume_ratio > 1.5:
                score += 20
            elif current_volume_ratio > 1.2:
                score += 15
            elif current_volume_ratio > 0.8:
                score += 10
            score = min(100, score + (consensus * 20))
            if score < self.min_score:
                return None

            entry_price = data['close'].iloc[-1]
            # 技术止损：近期低点下方
            recent_low = data['low'].iloc[-20:].min()
            tech_stop = recent_low * 0.995
            # ATR动态止损
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'BUY', atr)
            # 取更严格（更低）的止损
            stop_loss = min(tech_stop, atr_stop)

            # 止盈：基于风险回报比
            take_profit = self.calculate_dynamic_take_profit(entry_price, 'BUY', atr, self.risk_reward)
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
            if risk_reward < 1.5:
                return None

            return {
                'symbol': symbol, 'pattern': 'BOUNCE', 'direction': 'BUY',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(current_volume_ratio, 2), 'current_price': entry_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round(risk_reward, 1),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'signal_time': datetime.now(), 'signal_type': 'BUY',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"RSI超卖({current_rsi:.1f}) + 放量反弹，ATR动态止损"
            }
        except Exception:
            return None

# ---------- 2. BREAKOUT 突破模式 ----------
class BreakoutSignalChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('BREAKOUT')
        self.min_score = self.config.get('min_score', 30)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data = data_dict['15m']
            if len(data) < 50:
                return None

            market_state = self.get_market_state(data)
            if not self.is_market_allowed(market_state, 'BUY'):
                return None

            atr = self.get_atr(data)
            if atr is None or atr <= 0:
                return None

            indicators = self.get_multi_timeframe_indicators({'15m': data})['15m']
            current_rsi = indicators['rsi']
            conditions = self.config
            if pd.isna(current_rsi) or not (conditions.get('min_rsi', 45) <= current_rsi <= conditions.get('max_rsi', 68)):
                return None
            current_volume_ratio = indicators['volume_ratio']
            if pd.isna(current_volume_ratio) or current_volume_ratio < conditions.get('min_volume_ratio', 1.2):
                return None

            # 突破确认：价格高于前20周期高点
            resistance = data['high'].iloc[-21:-1].max()  # 不包含当前K线
            if data['close'].iloc[-1] < resistance * 1.01:
                return None

            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(
                self.get_multi_timeframe_indicators(data_dict), 'BUY')
            if consensus < 0.6:
                return None

            # 评分
            score = 50
            if 55 <= current_rsi <= 65:
                score += 25
            elif 45 <= current_rsi <= 55:
                score += 20
            elif 65 <= current_rsi <= 68:
                score += 15
            if current_volume_ratio > 2.0:
                score += 25
            elif current_volume_ratio > 1.5:
                score += 20
            elif current_volume_ratio > 1.2:
                score += 15
            breakout_strength = (data['close'].iloc[-1] - resistance) / resistance * 100
            if breakout_strength > 2:
                score += 20
            elif breakout_strength > 1:
                score += 15
            score = min(100, score + (consensus * 20))
            if score < self.min_score:
                return None

            entry_price = data['close'].iloc[-1]
            # 技术止损：突破前高下方
            tech_stop = resistance * 0.99
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'BUY', atr)
            stop_loss = min(tech_stop, atr_stop)

            take_profit = self.calculate_dynamic_take_profit(entry_price, 'BUY', atr, self.risk_reward)
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
            if risk_reward < 1.5:
                return None

            return {
                'symbol': symbol, 'pattern': 'BREAKOUT', 'direction': 'BUY',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(current_volume_ratio, 2), 'current_price': entry_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round(risk_reward, 1),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'signal_time': datetime.now(), 'signal_type': 'BUY',
                'confidence': 'HIGH' if score > 70 else 'MEDIUM',
                'reason': f"突破阻力{resistance:.4f}，涨幅{breakout_strength:.1f}%，ATR动态止损"
            }
        except Exception:
            return None

# ---------- 3. BREAKOUT_FAIL_SHORT 突破失败做空 ----------
class BreakoutFailShortChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('BREAKOUT_FAIL_SHORT')
        self.min_score = self.config.get('min_score', 35)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data = data_dict['15m']
            if len(data) < 50:
                return None

            market_state = self.get_market_state(data)
            if not self.is_market_allowed(market_state, 'SELL'):
                return None

            atr = self.get_atr(data)
            if atr is None or atr <= 0:
                return None

            indicators = self.get_multi_timeframe_indicators({'15m': data})['15m']
            current_rsi = indicators['rsi']
            if pd.isna(current_rsi) or current_rsi < self.config.get('min_rsi', 65):
                return None

            # 突破失败判定：价格未能站稳前高
            recent_high = data['high'].iloc[-11:-1].max()  # 前10根高点
            if data['close'].iloc[-1] > recent_high * 0.99:
                return None
            # 跌破短期支撑
            support = data['low'].iloc[-11:-1].min()
            if data['close'].iloc[-1] > support * 1.02:
                return None

            current_volume_ratio = indicators['volume_ratio']
            if pd.isna(current_volume_ratio) or current_volume_ratio < 0.8:
                return None

            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(
                self.get_multi_timeframe_indicators(data_dict), 'SELL')
            if consensus < 0.6:
                return None

            score = 40
            if current_rsi > 75:
                score += 30
            elif current_rsi > 70:
                score += 25
            elif current_rsi > 65:
                score += 20
            if current_volume_ratio > 1.2:
                score += 20
            elif current_volume_ratio > 1.0:
                score += 15
            elif current_volume_ratio > 0.8:
                score += 10
            price_change = (data['close'].iloc[-1] - data['close'].iloc[-5]) / data['close'].iloc[-5] * 100
            if price_change < -2:
                score += 15
            elif price_change < -1:
                score += 10
            ma20 = TechnicalIndicatorsMultiTF.calculate_ma(data, 20).iloc[-1]
            if data['close'].iloc[-1] < ma20:
                score += 10
            score = min(100, score + (consensus * 20))
            if score < self.min_score:
                return None

            entry_price = data['close'].iloc[-1]
            # 技术止损：前高上方
            tech_stop = recent_high * 1.01
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'SELL', atr)
            # 做空止损取较高者（更严格）
            stop_loss = max(tech_stop, atr_stop)

            take_profit = self.calculate_dynamic_take_profit(entry_price, 'SELL', atr, self.risk_reward)
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
            if risk_reward < 1.5:
                return None

            return {
                'symbol': symbol, 'pattern': 'BREAKOUT_FAIL_SHORT', 'direction': 'SELL',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(current_volume_ratio, 2), 'current_price': entry_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round(risk_reward, 1),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'signal_time': datetime.now(), 'signal_type': 'SELL',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"高RSI({current_rsi:.1f}) + 突破失败 + 跌破支撑，ATR动态止损"
            }
        except Exception:
            return None

# ---------- 4. TREND 趋势模式 ----------
class TrendSignalChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('TREND')
        self.min_score = self.config.get('min_score', 35)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data = data_dict['15m']
            if len(data) < 50:
                return None

            market_state = self.get_market_state(data)
            if not self.is_market_allowed(market_state, 'BUY'):
                return None

            atr = self.get_atr(data)
            if atr is None or atr <= 0:
                return None

            indicators = self.get_multi_timeframe_indicators({'15m': data})['15m']
            current_rsi = indicators['rsi']
            conditions = self.config
            if pd.isna(current_rsi) or not (conditions.get('min_rsi', 40) <= current_rsi <= conditions.get('max_rsi', 75)):
                return None
            current_volume_ratio = indicators['volume_ratio']
            if pd.isna(current_volume_ratio) or current_volume_ratio < conditions.get('min_volume_ratio', 1.0):
                return None

            # 多时间框架趋势确认
            uptrend_count = 0
            for tf, df in data_dict.items():
                if len(df) >= 30:
                    ma20_tf = TechnicalIndicatorsMultiTF.calculate_ma(df, 20).iloc[-1]
                    ma50_tf = TechnicalIndicatorsMultiTF.calculate_ma(df, 50).iloc[-1]
                    if df['close'].iloc[-1] > ma20_tf > ma50_tf:
                        uptrend_count += 1
            if uptrend_count < max(1, len(data_dict) * 0.5):
                return None

            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(
                self.get_multi_timeframe_indicators(data_dict), 'BUY')
            if consensus < 0.6:
                return None

            score = 60
            if 50 <= current_rsi <= 70:
                score += 20
            elif 40 <= current_rsi <= 50:
                score += 15
            if current_volume_ratio > 1.5:
                score += 20
            elif current_volume_ratio > 1.2:
                score += 15
            ma20 = indicators['ma20']
            ma50 = indicators['ma50']
            trend_strength = (ma20 - ma50) / ma50 * 100
            if trend_strength > 5:
                score += 15
            elif trend_strength > 3:
                score += 10
            score = min(100, score + (consensus * 20) + (uptrend_count * 5))
            if score < self.min_score:
                return None

            entry_price = data['close'].iloc[-1]
            # 技术止损：MA20下方
            tech_stop = ma20 * 0.98
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'BUY', atr)
            stop_loss = min(tech_stop, atr_stop)

            take_profit = self.calculate_dynamic_take_profit(entry_price, 'BUY', atr, self.risk_reward)
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
            if risk_reward < 1.5:
                return None

            return {
                'symbol': symbol, 'pattern': 'TREND', 'direction': 'BUY',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(current_volume_ratio, 2), 'current_price': entry_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round(risk_reward, 1),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'signal_time': datetime.now(), 'signal_type': 'BUY',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"多周期上升趋势，ADX:{market_state['adx_value']}，ATR动态止损"
            }
        except Exception:
            return None

# ---------- 5. CALLBACK 回调模式 ----------
class CallbackSignalChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('CALLBACK')
        self.min_score = self.config.get('min_score', 40)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data = data_dict['15m']
            if len(data) < 50:
                return None

            market_state = self.get_market_state(data)
            if not self.is_market_allowed(market_state, 'BUY'):
                return None

            atr = self.get_atr(data)
            if atr is None or atr <= 0:
                return None

            indicators = self.get_multi_timeframe_indicators({'15m': data})['15m']
            current_rsi = indicators['rsi']
            if pd.isna(current_rsi) or current_rsi < self.config.get('min_rsi', 55):
                return None

            # 计算回调幅度
            recent_high = data['high'].iloc[-30:].max()
            current_price = data['close'].iloc[-1]
            callback_pct = ((recent_high - current_price) / recent_high) * 100
            callback_range = self.config.get('callback_range', {'min': 5, 'max': 15})
            if not (callback_range['min'] <= callback_pct <= callback_range['max']):
                return None

            ma20 = indicators['ma20']
            if current_price < ma20 * 0.95:
                return None

            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(
                self.get_multi_timeframe_indicators(data_dict), 'BUY')
            if consensus < 0.6:
                return None

            score = 50
            if 60 <= current_rsi <= 70:
                score += 20
            elif current_rsi > 70:
                score += 15
            if 8 <= callback_pct <= 12:
                score += 25
            elif 5 <= callback_pct <= 15:
                score += 20
            volume_ratio = indicators['volume_ratio']
            if volume_ratio > 0.8:
                score += 15
            if current_price > ma20:
                score += 20
            score = min(100, score + (consensus * 20))
            if score < self.min_score:
                return None

            entry_price = current_price
            # 技术止损：回调低点下方
            recent_low = data['low'].iloc[-20:].min()
            tech_stop = recent_low * 0.995
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'BUY', atr)
            stop_loss = min(tech_stop, atr_stop)

            take_profit = self.calculate_dynamic_take_profit(entry_price, 'BUY', atr, self.risk_reward)
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
            if risk_reward < 1.5:
                return None

            return {
                'symbol': symbol, 'pattern': 'CALLBACK', 'direction': 'BUY',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'callback_pct': round(callback_pct, 1), 'current_price': entry_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round(risk_reward, 1),
                'volume_ratio': round(volume_ratio, 2),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'signal_time': datetime.now(), 'signal_type': 'BUY',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"回调至{callback_pct:.1f}%，RSI:{current_rsi:.1f}，ATR动态止损"
            }
        except Exception:
            return None

# ---------- 6. BOUNCE_FAIL_SHORT 反弹失败做空 ----------
class BounceFailShortChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('BOUNCE_FAIL_SHORT')
        self.min_score = self.config.get('min_score', 45)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data = data_dict['15m']
            if len(data) < 50:
                return None

            market_state = self.get_market_state(data)
            if not self.is_market_allowed(market_state, 'SELL'):
                return None

            atr = self.get_atr(data)
            if atr is None or atr <= 0:
                return None

            indicators = self.get_multi_timeframe_indicators({'15m': data})['15m']
            current_rsi = indicators['rsi']
            rsi_history = TechnicalIndicatorsMultiTF.calculate_rsi(data, 14)
            rsi_below_30 = rsi_history < 30
            if not rsi_below_30.any():
                return None
            last_rsi_below_30_idx = None
            for i in range(len(rsi_history) - 1, -1, -1):
                if rsi_below_30.iloc[i]:
                    last_rsi_below_30_idx = i
                    break
            if last_rsi_below_30_idx is None or len(data) - last_rsi_below_30_idx > self.config.get('lookback_periods', 10):
                return None

            low_price = data['low'].iloc[last_rsi_below_30_idx]
            high_after_low = data['high'].iloc[last_rsi_below_30_idx:].max()
            fib_levels = TechnicalIndicatorsMultiTF.calculate_fibonacci_levels(high_after_low, low_price)
            fib_38_2 = fib_levels['38.2%']
            bounce_pct = (high_after_low - low_price) / low_price * 100
            max_bounce = self.config.get('max_bounce_pct', 2.0)
            condition_a = bounce_pct < max_bounce
            condition_b = data['close'].iloc[-1] < fib_38_2
            if not (condition_a or condition_b):
                return None
            if data['close'].iloc[-1] > high_after_low * 0.99:
                return None
            volume_ratio = indicators['volume_ratio']
            if volume_ratio > 1.5:
                return None

            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(
                self.get_multi_timeframe_indicators(data_dict), 'SELL')
            if consensus < 0.7:
                return None

            score = 50
            if bounce_pct < 1.0:
                score += 30
            elif bounce_pct < 1.5:
                score += 25
            elif bounce_pct < 2.0:
                score += 20
            if current_rsi < 35:
                score += 20
            elif current_rsi < 40:
                score += 15
            elif current_rsi < 45:
                score += 10
            if volume_ratio < 0.7:
                score += 20
            elif volume_ratio < 0.9:
                score += 15
            elif volume_ratio < 1.1:
                score += 10
            distance_to_fib = abs(data['close'].iloc[-1] - fib_38_2) / fib_38_2 * 100
            if distance_to_fib > 2:
                score += 15
            elif distance_to_fib > 1:
                score += 10
            if condition_a and condition_b:
                score += 25
            elif condition_a or condition_b:
                score += 15
            score = min(100, score + (consensus * 30))
            if score < self.min_score:
                return None

            entry_price = data['close'].iloc[-1]
            # 技术止损：反弹高点上方
            tech_stop = high_after_low * 1.01
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'SELL', atr)
            stop_loss = max(tech_stop, atr_stop)

            take_profit = self.calculate_dynamic_take_profit(entry_price, 'SELL', atr, self.risk_reward)
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
            if risk_reward < 1.5:
                return None

            return {
                'symbol': symbol, 'pattern': 'BOUNCE_FAIL_SHORT', 'direction': 'SELL',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(volume_ratio, 2), 'bounce_pct': round(bounce_pct, 2),
                'current_price': entry_price, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round(risk_reward, 1),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'signal_time': datetime.now(), 'signal_type': 'SELL',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"反弹乏力({bounce_pct:.1f}%) + 未触38.2%回撤，ATR动态止损"
            }
        except Exception:
            return None

# ---------- 7. TREND_EXHAUSTION 趋势衰竭做空 ----------
class TrendExhaustionShortChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('TREND_EXHAUSTION')
        self.min_score = self.config.get('min_score', 55)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict or '1H' not in data_dict:
                return None
            data_15m = data_dict['15m']
            data_1h = data_dict['1H']
            if len(data_15m) < 50 or len(data_1h) < 30:
                return None

            market_state = self.get_market_state(data_15m)
            if not self.is_market_allowed(market_state, 'SELL'):
                return None

            atr = self.get_atr(data_15m)
            if atr is None or atr <= 0:
                return None

            current_price = data_15m['close'].iloc[-1]
            price_30_periods_ago = data_15m['close'].iloc[-30]
            price_increase = (current_price - price_30_periods_ago) / price_30_periods_ago * 100
            if price_increase < 15:
                return None

            conditions = self.config
            confirmation_signals = 0
            exhaustion_signals = []

            # 信号1：RSI顶背离
            rsi_15m = TechnicalIndicatorsMultiTF.calculate_rsi(data_15m, 14)
            recent_high_idx = data_15m['high'].iloc[-20:].idxmax()
            recent_rsi_high = rsi_15m.loc[recent_high_idx]
            current_rsi = rsi_15m.iloc[-1]
            if current_price > data_15m['high'].loc[recent_high_idx] and current_rsi < recent_rsi_high:
                confirmation_signals += 1
                exhaustion_signals.append("RSI顶背离")

            # 信号2：成交量递减
            volume_ma10 = data_15m['volume'].rolling(window=10).mean()
            volume_ma20 = data_15m['volume'].rolling(window=20).mean()
            if volume_ma10.iloc[-1] < volume_ma20.iloc[-1] * 0.8:
                confirmation_signals += 1
                exhaustion_signals.append("成交量递减")

            # 信号3：涨幅递减
            recent_highs = data_15m['high'].iloc[-30:].values
            highs_increase = []
            for i in range(1, len(recent_highs)):
                if recent_highs[i] > recent_highs[i-1]:
                    increase = (recent_highs[i] - recent_highs[i-1]) / recent_highs[i-1] * 100
                    highs_increase.append(increase)
            if len(highs_increase) >= 3:
                last_3_increases = highs_increase[-3:]
                if last_3_increases[2] < last_3_increases[1] < last_3_increases[0]:
                    confirmation_signals += 1
                    exhaustion_signals.append("涨幅递减")

            # 信号4：MACD背离
            macd_diff = TechnicalIndicatorsMultiTF.calculate_macd(data_15m)
            if len(macd_diff) > 10:
                recent_macd_high_idx = macd_diff.iloc[-20:].idxmax()
                recent_price_high_idx = data_15m['high'].iloc[-20:].idxmax()
                if (current_price > data_15m['high'].loc[recent_price_high_idx] and
                    macd_diff.iloc[-1] < macd_diff.loc[recent_macd_high_idx]):
                    confirmation_signals += 1
                    exhaustion_signals.append("MACD背离")

            # 信号5：长上影线
            recent_kline = data_15m.iloc[-1]
            if (recent_kline['high'] - max(recent_kline['open'], recent_kline['close'])) > (
                abs(recent_kline['close'] - recent_kline['open']) * 1.5):
                confirmation_signals += 1
                exhaustion_signals.append("长上影线")

            # 1H周期超买
            rsi_1h = TechnicalIndicatorsMultiTF.calculate_rsi(data_1h, 14)
            if rsi_1h.iloc[-1] > 70:
                confirmation_signals += 1
                exhaustion_signals.append("1H周期超买")

            if confirmation_signals < conditions.get('required_confirmation', 3):
                return None

            score = 40 + (confirmation_signals * 15)
            if score < self.min_score:
                return None

            entry_price = current_price
            # 技术止损：近期高点上方
            recent_high = data_15m['high'].iloc[-20:].max()
            tech_stop = recent_high * 1.01
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'SELL', atr)
            stop_loss = max(tech_stop, atr_stop)

            take_profit = self.calculate_dynamic_take_profit(entry_price, 'SELL', atr, self.risk_reward)
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
            if risk_reward < 2.0:
                return None

            return {
                'symbol': symbol, 'pattern': 'TREND_EXHAUSTION', 'direction': 'SELL',
                'rsi': round(float(current_rsi), 1), 'price_increase': round(price_increase, 1),
                'confirmation_signals': confirmation_signals, 'exhaustion_signals': exhaustion_signals,
                'score': int(score), 'current_price': entry_price, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': round(risk_reward, 1),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'signal_time': datetime.now(), 'signal_type': 'SELL',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"趋势衰竭({confirmation_signals}个确认信号): {', '.join(exhaustion_signals)}，ATR动态止损"
            }
        except Exception:
            return None

# ---------- 8. BOUNCE_FAIL_CONFIRM_K 反弹失败·确认K做空 ----------
class BounceFailConfirmKShortChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('BOUNCE_FAIL_CONFIRM_K')
        self.kline_analyzer = KLineAnalyzer()
        self.min_score = self.config.get('min_score', 50)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data_15m = data_dict['15m']
            if len(data_15m) < 50:
                return None

            # ADX过滤
            market_state = self.get_market_state(data_15m)
            if not self.is_market_allowed(market_state, 'SELL'):
                return None

            # 获取ATR
            atr = self.get_atr(data_15m)
            if atr is None or atr <= 0:
                return None

            bounce_structure = self._identify_bounce_structure(data_15m)
            if not bounce_structure:
                return None
            if not self._check_bounce_failure(data_15m, bounce_structure):
                return None
            current_kline = data_15m.iloc[-1]
            kline_analysis = self.kline_analyzer.analyze_candle(current_kline)
            if not kline_analysis or kline_analysis['is_bullish']:
                return None
            if not self.kline_analyzer.is_confirmation_candle(kline_analysis, 'SELL', self.config):
                return None
            if not self._is_first_confirmation_k(data_15m, bounce_structure):
                return None
            if not self._check_volume_confirmation(data_15m):
                return None
            indicators_ok, indicators_info = self._check_technical_indicators(data_15m)
            if not indicators_ok:
                return None

            score = self._calculate_signal_score(data_15m, bounce_structure, kline_analysis, indicators_info)
            if score < self.min_score:
                return None

            entry_price = kline_analysis['close']
            # 技术止损：反弹高点上方
            tech_stop = bounce_structure['bounce_high'] * 1.01
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'SELL', atr)
            stop_loss = max(tech_stop, atr_stop)

            take_profit = self.calculate_dynamic_take_profit(entry_price, 'SELL', atr, self.risk_reward)
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
            if risk_reward < 1.5:
                return None

            return {
                'symbol': symbol, 'pattern': 'BOUNCE_FAIL_CONFIRM_K', 'direction': 'SELL',
                'score': int(score), 'current_price': entry_price, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': round(risk_reward, 2),
                'rsi': indicators_info.get('rsi', 50), 'volume_ratio': indicators_info.get('volume_ratio', 1),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'confirmation_k_info': {
                    'is_bullish': kline_analysis['is_bullish'],
                    'entity_ratio': round(kline_analysis['entity_ratio'], 2),
                    'lower_shadow_ratio': round(kline_analysis['lower_shadow_ratio'], 2),
                    'body_size': round(kline_analysis['body_size'], 4)
                },
                'is_first_confirmation_k': True, 'signal_time': datetime.now(),
                'signal_type': 'SELL', 'confidence': 'HIGH' if score > 70 else 'MEDIUM',
                'reason': f"反弹失败确认K做空 | 反弹幅度:{bounce_structure['bounce_pct']:.1f}% | 确认K实体:{kline_analysis['entity_ratio']:.0%} | ATR动态止损"
            }
        except Exception:
            return None

    def _identify_bounce_structure(self, data: pd.DataFrame) -> Dict:
        if len(data) < 30:
            return None
        recent_low_idx = data['low'].iloc[-30:].idxmin()
        recent_low = data['low'].loc[recent_low_idx]
        after_low_data = data.loc[recent_low_idx:]
        if len(after_low_data) < 3:
            return None
        bounce_high_idx = after_low_data['high'].idxmax()
        bounce_high = after_low_data['high'].loc[bounce_high_idx]
        bounce_pct = (bounce_high - recent_low) / recent_low * 100
        if bounce_pct < 1.0:
            return None
        fib_levels = TechnicalIndicatorsMultiTF.calculate_fibonacci_levels(bounce_high, recent_low)
        return {
            'bounce_low': recent_low, 'bounce_high': bounce_high, 'bounce_pct': bounce_pct,
            'fib_levels': fib_levels,
            'bounce_peak_index': data.index.get_loc(bounce_high_idx)
        }

    def _check_bounce_failure(self, data: pd.DataFrame, bounce_structure: Dict) -> bool:
        current_price = data['close'].iloc[-1]
        bounce_high = bounce_structure['bounce_high']
        fib_38_2 = bounce_structure['fib_levels']['38.2%']
        if current_price > bounce_high * 0.99:
            return False
        if current_price > fib_38_2:
            return False
        if bounce_structure['bounce_pct'] > 8:
            return False
        return True

    def _is_first_confirmation_k(self, data: pd.DataFrame, bounce_structure: Dict) -> bool:
        bounce_peak_idx = bounce_structure['bounce_peak_index']
        current_idx = len(data) - 1
        bearish_count = 0
        for i in range(bounce_peak_idx + 1, current_idx + 1):
            if i >= len(data):
                break
            kline = data.iloc[i]
            kline_analysis = self.kline_analyzer.analyze_candle(kline)
            if not kline_analysis:
                continue
            if kline_analysis['is_bullish']:
                bearish_count = 0
            else:
                bearish_count += 1
                if kline_analysis['entity_ratio'] >= self.config.get('min_entity_ratio', 0.6):
                    if i == current_idx and bearish_count == 1:
                        return True
        return False

    def _check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        if len(data) < 20:
            return False
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        return volume_ratio >= 0.5

    def _check_technical_indicators(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        info = {}
        rsi = TechnicalIndicatorsMultiTF.calculate_rsi(data, 14)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        info['rsi'] = round(current_rsi, 1)
        if current_rsi < 35:
            return False, info
        volume_ratio = TechnicalIndicatorsMultiTF.calculate_volume_ratio(data, 20)
        current_volume_ratio = volume_ratio.iloc[-1] if not pd.isna(volume_ratio.iloc[-1]) else 1
        info['volume_ratio'] = round(current_volume_ratio, 2)
        ma20 = TechnicalIndicatorsMultiTF.calculate_ma(data, 20).iloc[-1]
        if data['close'].iloc[-1] > ma20 * 1.05:
            return False, info
        return True, info

    def _calculate_signal_score(self, data: pd.DataFrame, bounce_structure: Dict,
                               kline_analysis: Dict, indicators_info: Dict) -> int:
        score = 50
        bounce_pct = bounce_structure['bounce_pct']
        if bounce_pct < 2.0:
            score += 20
        elif bounce_pct < 3.0:
            score += 15
        elif bounce_pct < 4.0:
            score += 10
        entity_ratio = kline_analysis['entity_ratio']
        if entity_ratio > 0.8:
            score += 20
        elif entity_ratio > 0.7:
            score += 15
        elif entity_ratio > 0.6:
            score += 10
        rsi = indicators_info.get('rsi', 50)
        if rsi > 60:
            score += 15
        elif rsi > 55:
            score += 10
        volume_ratio = indicators_info.get('volume_ratio', 1)
        if volume_ratio > 1.2:
            score += 10
        elif volume_ratio > 0.8:
            score += 5
        return min(score, 100)

# ---------- 9. CALLBACK_CONFIRM_K 回调企稳·确认K做多 ----------
class CallbackConfirmKBuyChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('CALLBACK_CONFIRM_K')
        self.kline_analyzer = KLineAnalyzer()
        self.min_score = self.config.get('min_score', 50)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data_15m = data_dict['15m']
            if len(data_15m) < 50:
                return None

            market_state = self.get_market_state(data_15m)
            if not self.is_market_allowed(market_state, 'BUY'):
                return None

            atr = self.get_atr(data_15m)
            if atr is None or atr <= 0:
                return None

            callback_structure = self._identify_callback_structure(data_15m)
            if not callback_structure:
                return None
            if not self._check_callback_stabilization(data_15m, callback_structure):
                return None
            current_kline = data_15m.iloc[-1]
            kline_analysis = self.kline_analyzer.analyze_candle(current_kline)
            if not kline_analysis or not kline_analysis['is_bullish']:
                return None
            if not self.kline_analyzer.is_confirmation_candle(kline_analysis, 'BUY', self.config):
                return None
            if not self._is_first_confirmation_k(data_15m, callback_structure):
                return None
            if not self._check_volume_confirmation(data_15m):
                return None
            indicators_ok, indicators_info = self._check_technical_indicators(data_15m)
            if not indicators_ok:
                return None

            score = self._calculate_signal_score(data_15m, callback_structure, kline_analysis, indicators_info)
            if score < self.min_score:
                return None

            entry_price = kline_analysis['close']
            # 技术止损：回调低点下方
            tech_stop = callback_structure['callback_low'] * 0.995
            atr_stop = self.calculate_dynamic_stop_loss(entry_price, 'BUY', atr)
            stop_loss = min(tech_stop, atr_stop)

            take_profit = self.calculate_dynamic_take_profit(entry_price, 'BUY', atr, self.risk_reward)
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
            if risk_reward < 1.5:
                return None

            return {
                'symbol': symbol, 'pattern': 'CALLBACK_CONFIRM_K', 'direction': 'BUY',
                'score': int(score), 'current_price': entry_price, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': round(risk_reward, 2),
                'rsi': indicators_info.get('rsi', 50), 'volume_ratio': indicators_info.get('volume_ratio', 1),
                'atr': round(atr, 4), 'adx': market_state['adx_value'],
                'market_state': market_state['trend_strength'],
                'confirmation_k_info': {
                    'is_bullish': kline_analysis['is_bullish'],
                    'entity_ratio': round(kline_analysis['entity_ratio'], 2),
                    'upper_shadow_ratio': round(kline_analysis['upper_shadow_ratio'], 2),
                    'body_size': round(kline_analysis['body_size'], 4)
                },
                'is_first_confirmation_k': True, 'signal_time': datetime.now(),
                'signal_type': 'BUY', 'confidence': 'HIGH' if score > 70 else 'MEDIUM',
                'reason': f"回调企稳确认K做多 | 回调幅度:{callback_structure['callback_pct']:.1f}% | 确认K实体:{kline_analysis['entity_ratio']:.0%} | ATR动态止损"
            }
        except Exception:
            return None

    def _identify_callback_structure(self, data: pd.DataFrame) -> Dict:
        if len(data) < 30:
            return None
        recent_high_idx = data['high'].iloc[-30:].idxmax()
        recent_high = data['high'].loc[recent_high_idx]
        after_high_data = data.loc[recent_high_idx:]
        if len(after_high_data) < 3:
            return None
        callback_low_idx = after_high_data['low'].idxmin()
        callback_low = after_high_data['low'].loc[callback_low_idx]
        callback_pct = (recent_high - callback_low) / recent_high * 100
        if callback_pct < 3.0:
            return None
        fib_levels = TechnicalIndicatorsMultiTF.calculate_fibonacci_levels(recent_high, callback_low)
        return {
            'callback_high': recent_high, 'callback_low': callback_low, 'callback_pct': callback_pct,
            'fib_levels': fib_levels,
            'callback_low_index': data.index.get_loc(callback_low_idx)
        }

    def _check_callback_stabilization(self, data: pd.DataFrame, callback_structure: Dict) -> bool:
        current_price = data['close'].iloc[-1]
        callback_low = callback_structure['callback_low']
        if current_price < callback_low * 0.995:
            return False
        fib_61_8 = callback_structure['fib_levels']['61.8%']
        if current_price < fib_61_8 * 0.99:
            return False
        callback_pct = callback_structure['callback_pct']
        if callback_pct < 5 or callback_pct > 20:
            return False
        return True

    def _is_first_confirmation_k(self, data: pd.DataFrame, callback_structure: Dict) -> bool:
        callback_low_idx = callback_structure['callback_low_index']
        current_idx = len(data) - 1
        bullish_count = 0
        for i in range(callback_low_idx + 1, current_idx + 1):
            if i >= len(data):
                break
            kline = data.iloc[i]
            kline_analysis = self.kline_analyzer.analyze_candle(kline)
            if not kline_analysis:
                continue
            if not kline_analysis['is_bullish']:
                bullish_count = 0
            else:
                bullish_count += 1
                if kline_analysis['entity_ratio'] >= self.config.get('min_entity_ratio', 0.6):
                    if i == current_idx and bullish_count == 1:
                        return True
        return False

    def _check_volume_confirmation(self, data: pd.DataFrame) -> bool:
        if len(data) < 20:
            return False
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        return volume_ratio >= 0.8

    def _check_technical_indicators(self, data: pd.DataFrame) -> Tuple[bool, Dict]:
        info = {}
        rsi = TechnicalIndicatorsMultiTF.calculate_rsi(data, 14)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        info['rsi'] = round(current_rsi, 1)
        if current_rsi > 70:
            return False, info
        volume_ratio = TechnicalIndicatorsMultiTF.calculate_volume_ratio(data, 20)
        current_volume_ratio = volume_ratio.iloc[-1] if not pd.isna(volume_ratio.iloc[-1]) else 1
        info['volume_ratio'] = round(current_volume_ratio, 2)
        ma20 = TechnicalIndicatorsMultiTF.calculate_ma(data, 20).iloc[-1]
        if data['close'].iloc[-1] < ma20 * 0.95:
            return False, info
        return True, info

    def _calculate_signal_score(self, data: pd.DataFrame, callback_structure: Dict,
                               kline_analysis: Dict, indicators_info: Dict) -> int:
        score = 50
        callback_pct = callback_structure['callback_pct']
        if 5 <= callback_pct <= 10:
            score += 20
        elif 10 < callback_pct <= 15:
            score += 15
        elif 3 <= callback_pct <= 20:
            score += 10
        entity_ratio = kline_analysis['entity_ratio']
        if entity_ratio > 0.8:
            score += 20
        elif entity_ratio > 0.7:
            score += 15
        elif entity_ratio > 0.6:
            score += 10
        rsi = indicators_info.get('rsi', 50)
        if rsi < 40:
            score += 15
        elif rsi < 45:
            score += 10
        volume_ratio = indicators_info.get('volume_ratio', 1)
        if volume_ratio > 1.5:
            score += 15
        elif volume_ratio > 1.2:
            score += 10
        return min(score, 100)

# ============ 回测引擎 ============
class BacktestEngine:
    def __init__(self, config: Dict, data_fetcher: OKXDataFetcher):
        self.config = config
        self.data_fetcher = data_fetcher
        self.signal_checkers = self._init_checkers()

    def _init_checkers(self):
        checkers = {}
        for mode in UltimateConfig.MARKET_MODES:
            if UltimateConfig.MARKET_MODES[mode]['enabled']:
                try:
                    if mode == 'BOUNCE':
                        checkers[mode] = BounceSignalChecker()
                    elif mode == 'BREAKOUT':
                        checkers[mode] = BreakoutSignalChecker()
                    elif mode == 'BREAKOUT_FAIL_SHORT':
                        checkers[mode] = BreakoutFailShortChecker()
                    elif mode == 'TREND':
                        checkers[mode] = TrendSignalChecker()
                    elif mode == 'CALLBACK':
                        checkers[mode] = CallbackSignalChecker()
                    elif mode == 'BOUNCE_FAIL_SHORT':
                        checkers[mode] = BounceFailShortChecker()
                    elif mode == 'TREND_EXHAUSTION':
                        checkers[mode] = TrendExhaustionShortChecker()
                    elif mode == 'BOUNCE_FAIL_CONFIRM_K':
                        checkers[mode] = BounceFailConfirmKShortChecker()
                    elif mode == 'CALLBACK_CONFIRM_K':
                        checkers[mode] = CallbackConfirmKBuyChecker()
                except Exception as e:
                    print(f"⚠️ 回测加载策略 {mode} 失败: {e}")
        return checkers

    def run(self):
        print("\n" + "="*70)
        print("🚀 开始回测 (最长2星期)")
        print(f"📅 时间范围: {self.config['start_date']} 至 {self.config['end_date']}")
        print(f"📊 回测币种: {', '.join(self.config['symbols'])}")
        print(f"💵 初始资金: ${self.config['initial_capital']:,.2f}")
        print(f"💸 手续费: {self.config['commission']*100:.2f}% | 滑点: {self.config['slippage']*100:.2f}%")
        print("="*70)

        total_pnl = 0
        total_trades = 0
        winning_trades = 0
        equity = self.config['initial_capital']
        trades = []

        for symbol in self.config['symbols']:
            print(f"\n🔍 回测 {symbol} ...")
            df = self.data_fetcher.get_historical_candles(
                symbol, self.config['interval'],
                self.config['start_date'], self.config['end_date']
            )
            if df is None or len(df) < 50:
                print(f"⚠️ {symbol} 数据不足，跳过")
                continue

            for i in range(50, len(df)):
                current_data = df.iloc[:i+1]
                data_dict = {self.config['interval']: current_data}
                for name, checker in self.signal_checkers.items():
                    signal = checker.check_coin_multi_tf(symbol, data_dict)
                    if signal:
                        entry_price = signal['entry_price']
                        stop_loss = signal['stop_loss']
                        take_profit = signal['take_profit']
                        direction = signal['direction']

                        commission = self.config['commission']
                        slippage = self.config['slippage']
                        if direction == 'BUY':
                            exec_price = entry_price * (1 + slippage)
                        else:
                            exec_price = entry_price * (1 - slippage)

                        exit_price = None
                        exit_reason = None
                        for j in range(i+1, len(df)):
                            bar = df.iloc[j]
                            if direction == 'BUY':
                                if bar['low'] <= stop_loss:
                                    exit_price = stop_loss * (1 - slippage)
                                    exit_reason = 'STOP_LOSS'
                                    break
                                elif bar['high'] >= take_profit:
                                    exit_price = take_profit * (1 + slippage)
                                    exit_reason = 'TAKE_PROFIT'
                                    break
                            else:
                                if bar['high'] >= stop_loss:
                                    exit_price = stop_loss * (1 + slippage)
                                    exit_reason = 'STOP_LOSS'
                                    break
                                elif bar['low'] <= take_profit:
                                    exit_price = take_profit * (1 - slippage)
                                    exit_reason = 'TAKE_PROFIT'
                                    break

                        if exit_price is None:
                            last_close = df['close'].iloc[-1]
                            if direction == 'BUY':
                                exit_price = last_close * (1 - slippage)
                            else:
                                exit_price = last_close * (1 + slippage)
                            exit_reason = 'EXPIRED'

                        if direction == 'BUY':
                            pnl = (exit_price - exec_price) * (1 - commission)
                        else:
                            pnl = (exec_price - exit_price) * (1 - commission)

                        equity += pnl
                        total_pnl += pnl
                        total_trades += 1
                        if pnl > 0:
                            winning_trades += 1

                        trades.append({
                            'time': df.index[i],
                            'symbol': symbol,
                            'direction': direction,
                            'entry': exec_price,
                            'exit': exit_price,
                            'pnl': pnl,
                            'reason': exit_reason
                        })

                        print(f"   {df.index[i].strftime('%m-%d %H:%M')} {symbol} {direction} "
                              f"入场:{exec_price:.4f} 离场:{exit_price:.4f} "
                              f"盈亏:${pnl:.2f} ({exit_reason})")

        print("\n" + "="*70)
        print("📊 回测统计")
        print("="*70)
        print(f"初始资金: ${self.config['initial_capital']:,.2f}")
        print(f"最终权益: ${equity:,.2f}")
        print(f"总盈亏: ${total_pnl:,.2f}  ({total_pnl/self.config['initial_capital']*100:+.2f}%)")
        print(f"总交易次数: {total_trades}")
        if total_trades > 0:
            win_rate = (winning_trades / total_trades) * 100
            print(f"胜率: {win_rate:.2f}% ({winning_trades}/{total_trades})")
            avg_pnl = total_pnl / total_trades
            print(f"平均盈亏: ${avg_pnl:.2f}")
        print("="*70)
        return {
            'total_pnl': total_pnl,
            'total_trades': total_trades,
            'win_rate': win_rate if total_trades > 0 else 0,
            'equity': equity,
            'trades': trades
        }

# ============ 主系统 ============
class UltimateTradingSystem:
    def __init__(self, telegram_bot_token=None, telegram_chat_id=None):
        print("🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
        print("🚀 终极智能交易系统 v34.0 全策略ATR/ADX适配版")
        print("🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
        self.config = UltimateConfig
        self.coins = MONITOR_COINS[:self.config.COINS_TO_MONITOR]
        self.analysis_cycle_count = 0
        self.no_signal_count = 0
        self.start_time = datetime.now()
        self.cooldown_manager = CooldownManager()
        self.data_fetcher = OKXDataFetcher()
        self.signal_checkers = {}
        self.init_signal_checkers()
        if telegram_bot_token and telegram_chat_id:
            try:
                self.telegram = UltimateTelegramNotifier(telegram_bot_token, telegram_chat_id)
            except Exception as e:
                print(f"⚠️ Telegram通知器初始化失败: {e}")
                self.telegram = None
        else:
            self.telegram = None
        self.stats = {
            'total_signals': 0, 'buy_signals': 0, 'sell_signals': 0,
            'signals_today': defaultdict(int)
        }
        print("\n✅ 系统初始化完成")
        print("=" * 70)

    def init_signal_checkers(self):
        print("\n🔄 初始化信号检查器...")
        checkers_map = {
            'BOUNCE': BounceSignalChecker,
            'BREAKOUT': BreakoutSignalChecker,
            'BREAKOUT_FAIL_SHORT': BreakoutFailShortChecker,
            'TREND': TrendSignalChecker,
            'CALLBACK': CallbackSignalChecker,
            'BOUNCE_FAIL_SHORT': BounceFailShortChecker,
            'TREND_EXHAUSTION': TrendExhaustionShortChecker,
            'BOUNCE_FAIL_CONFIRM_K': BounceFailConfirmKShortChecker,
            'CALLBACK_CONFIRM_K': CallbackConfirmKBuyChecker
        }
        for mode, checker_class in checkers_map.items():
            try:
                if self.config.MARKET_MODES.get(mode, {}).get('enabled', False):
                    self.signal_checkers[mode] = checker_class()
                    print(f"   ✅ {mode}: {checker_class.__name__}")
                else:
                    print(f"   ⚠️ {mode}: 已禁用")
            except Exception as e:
                print(f"   ❌ {mode}: 初始化失败 - {str(e)}")
        print(f"✅ 已初始化 {len(self.signal_checkers)} 个信号检查器")

    def get_coins_data(self):
        print("📊 获取实时市场数据...")
        return self.data_fetcher.get_all_coins_data(self.coins)

    def run_multi_mode_analysis(self, coins_data):
        if not coins_data:
            return []
        print(f"🔄 运行多模式分析 ({len(coins_data)}个币种)...")
        all_signals = []
        for mode, checker in self.signal_checkers.items():
            print(f"🤖 检查{mode}模式...")
            mode_signals = []
            for symbol, data_dict in coins_data.items():
                direction = 'SELL' if mode in ['BREAKOUT_FAIL_SHORT', 'BOUNCE_FAIL_SHORT',
                                              'TREND_EXHAUSTION', 'BOUNCE_FAIL_CONFIRM_K'] else 'BUY'
                cooldown_ok, _ = self.cooldown_manager.check_cooldown(symbol, direction)
                if not cooldown_ok:
                    continue
                signal = checker.check_coin_multi_tf(symbol, data_dict)
                if signal:
                    mode_signals.append(signal)
            if mode_signals:
                print(f"✅ {mode}模式发现 {len(mode_signals)} 个信号")
                all_signals.extend(mode_signals)
            else:
                print(f"📊 {mode}模式未发现信号")
        unique = {}
        for s in all_signals:
            sym = s['symbol']
            if sym not in unique or s['score'] > unique[sym]['score']:
                unique[sym] = s
        final = list(unique.values())
        final.sort(key=lambda x: x.get('score', 0), reverse=True)
        return final[:self.config.MAX_SIGNALS]

    def process_signals(self, signals):
        if not signals:
            if self.telegram:
                try:
                    no_signal_msg = f"""
📊 <b>本轮分析结果</b>
━━━━━━━━━━━━━━━━━━━━
🔍 分析时间: {datetime.now().strftime('%H:%M:%S')}
📊 扫描币种: {len(self.coins)}个
❌ 未发现符合条件的交易信号
━━━━━━━━━━━━━━━━━━━━
💡 <i>市场可能没有明显机会，建议观望</i>
⏰ 下次分析: {self.config.ANALYSIS_INTERVAL}分钟后
"""
                    self.telegram.bot.send_message(self.telegram.chat_id, no_signal_msg, parse_mode='HTML')
                except:
                    pass
            return
        sorted_signals = sorted(signals, key=lambda x: x.get('score', 0), reverse=True)
        if self.telegram:
            self.telegram.send_top_3_signals(sorted_signals)
        for signal in sorted_signals:
            cooldown_ok, _ = self.cooldown_manager.check_cooldown(
                signal['symbol'], signal['direction']
            )
            if cooldown_ok:
                self.cooldown_manager.record_signal(
                    signal['symbol'], signal['direction'],
                    signal['pattern'], signal['score']
                )
                self.stats['total_signals'] += 1
                if signal['direction'] == 'BUY':
                    self.stats['buy_signals'] += 1
                else:
                    self.stats['sell_signals'] += 1
                today = datetime.now().strftime('%Y-%m-%d')
                self.stats['signals_today'][today] += 1
        print(f"\n✅ 已处理 {len(sorted_signals)} 个信号")

    def run_analysis_cycle_enhanced(self):
        try:
            self.analysis_cycle_count += 1
            print("\n" + "="*70)
            print(f"🤖 第 {self.analysis_cycle_count} 次智能分析周期 (v{self.config.VERSION})")
            print("="*70)
            coins_data = self.get_coins_data()
            if not coins_data or len(coins_data) < 10:
                print("❌ 数据获取失败或数据不足，跳过本次分析")
                return []
            signals = self.run_multi_mode_analysis(coins_data)
            if signals:
                self.process_signals(signals)
                self.no_signal_count = 0
            else:
                print("\n📊 本次分析未发现符合条件的交易信号")
                self.no_signal_count += 1
            next_time = datetime.now() + timedelta(minutes=self.config.ANALYSIS_INTERVAL)
            print(f"\n⏳ 下次分析: {self.config.ANALYSIS_INTERVAL}分钟后 ({next_time.strftime('%H:%M:%S')})")
            return signals
        except Exception as e:
            print(f"\n❌ 分析周期失败: {str(e)}")
            traceback.print_exc()
            return []

    def run_single_cycle(self):
        print("\n🚀 立即运行分析周期...")
        return self.run_analysis_cycle_enhanced()

    def run_continuous(self):
        print("\n🚀 系统将在3秒后自动启动连续监控模式...")
        for i in range(3, 0, -1):
            print(f"  {i}...")
            time.sleep(1)
        print("\n🚀 终极系统开始自动连续运行...")
        print("💡 按 Ctrl+C 停止\n")
        self.run_analysis_cycle_enhanced()
        while True:
            try:
                time.sleep(self.config.ANALYSIS_INTERVAL * 60)
                self.run_analysis_cycle_enhanced()
            except KeyboardInterrupt:
                print("\n\n🛑 系统被用户中断")
                break
            except Exception as e:
                print(f"\n❌ 运行出错: {e}")
                time.sleep(60)

# ============ Telegram通知器 ============
class UltimateTelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot = telebot.TeleBot(bot_token)
        self.chat_id = chat_id
        self.config = UltimateConfig.TELEGRAM_CONFIG
        self.message_history = deque(maxlen=100)
        self.test_connection()
        if not IN_GITHUB_ACTIONS:
            self.send_startup_message()
        else:
            # GitHub Actions 中只发送一个简短运行通知（可选）
            try:
                self.bot.send_message(self.chat_id, "🔁 GitHub Actions 自动分析已启动", parse_mode='HTML')
            except:
                pass

    def send_startup_message(self):
        try:
            startup_msg = f"""
🚀 <b>终极智能交易系统 v{UltimateConfig.VERSION} 已启动！</b>
━━━━━━━━━━━━━━━━━━━━
📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 监控币种: {len(MONITOR_COINS)}个
⏰ 分析间隔: {UltimateConfig.ANALYSIS_INTERVAL}分钟
🔍 分析周期: {', '.join(UltimateConfig.MULTI_TIMEFRAME_CONFIG['timeframes'])}
⚙️ 动态止损: ATR×{UltimateConfig.ATR_CONFIG['stop_loss_multiplier']}
⚙️ 市场过滤: ADX≥{UltimateConfig.ADX_CONFIG['trend_threshold']}
⚙️ 全策略ATR/ADX适配
━━━━━━━━━━━━━━━━━━━━
<code>系统状态: ✅ 运行中</code>
"""
            self.bot.send_message(self.chat_id, startup_msg, parse_mode='HTML', disable_web_page_preview=True)
        except Exception as e:
            print(f"❌ 发送启动消息失败: {e}")

    def test_connection(self):
        try:
            self.bot.get_me()
            print("✅ Telegram连接测试成功")
        except Exception as e:
            print(f"❌ Telegram连接失败: {e}")
            self.config['enabled'] = False

    def format_price(self, price):
        try:
            if price >= 100:
                return f"${price:,.2f}"
            elif price >= 10:
                return f"${price:,.3f}"
            elif price >= 1:
                return f"${price:,.4f}"
            elif price >= 0.1:
                return f"${price:,.5f}"
            elif price >= 0.01:
                return f"${price:,.6f}"
            elif price >= 0.001:
                return f"${price:,.7f}"
            else:
                return f"${price:,.8f}"
        except:
            return f"${price}"

    def format_percentage(self, value):
        try:
            return f"{value:+.1f}%" if value >= 0 else f"{value:.1f}%"
        except:
            return f"{value}%"

    def get_emoji_for_pattern(self, pattern):
        emojis = {
            'BOUNCE': '🔺', 'BREAKOUT': '🚀', 'BREAKOUT_FAIL_SHORT': '🔻',
            'TREND': '📈', 'CALLBACK': '📉', 'BOUNCE_FAIL_SHORT': '⚡',
            'TREND_EXHAUSTION': '🔥', 'BOUNCE_FAIL_CONFIRM_K': '🎯',
            'CALLBACK_CONFIRM_K': '🎯'
        }
        return emojis.get(pattern, '💰')

    def get_emoji_for_direction(self, direction):
        return '🟢' if direction == 'BUY' else '🔴'

    def get_emoji_for_rsi(self, rsi):
        try:
            rsi_value = float(rsi)
            if rsi_value < 30:
                return '🟢'
            elif rsi_value > 70:
                return '🔴'
            elif rsi_value > 60:
                return '🟡'
            else:
                return '⚪'
        except:
            return '⚪'

    def get_emoji_for_score(self, score):
        try:
            score_value = int(score)
            if score_value >= 90:
                return '🔥🔥'
            elif score_value >= 80:
                return '🔥'
            elif score_value >= 70:
                return '⭐'
            elif score_value >= 60:
                return '✨'
            else:
                return '📊'
        except:
            return '📊'

    def create_detailed_signal_message(self, signal):
        try:
            symbol = signal.get('symbol', 'UNKNOWN')
            pattern = signal.get('pattern', 'UNKNOWN')
            direction = signal.get('direction', 'BUY')
            score = signal.get('score', 0)
            rsi = signal.get('rsi', 0)

            pattern_emoji = self.get_emoji_for_pattern(pattern)
            direction_emoji = self.get_emoji_for_direction(direction)
            score_emoji = self.get_emoji_for_score(score)
            rsi_emoji = self.get_emoji_for_rsi(rsi)

            current_price = self.format_price(signal.get('current_price', 0))
            entry_price = self.format_price(signal.get('entry_price', 0))
            stop_loss = self.format_price(signal.get('stop_loss', 0))
            take_profit = self.format_price(signal.get('take_profit', 0))

            entry = float(signal.get('entry_price', 1))
            sl = float(signal.get('stop_loss', 1))
            tp = float(signal.get('take_profit', 1))

            if direction == 'BUY':
                risk_pct = (entry - sl) / entry * 100
                reward_pct = (tp - entry) / entry * 100
            else:
                risk_pct = (sl - entry) / entry * 100
                reward_pct = (entry - tp) / entry * 100

            message = f"""
{pattern_emoji} <b>{symbol}/USDT - {pattern}</b> {direction_emoji}
━━━━━━━━━━━━━━━━━━━━
📊 <b>信号强度:</b> {score}/100 {score_emoji}
├ 交易方向: {'买入做多' if direction == 'BUY' else '卖出做空'}
├ RSI指标: {rsi:.1f} {rsi_emoji}
└ 风险回报比: {signal.get('risk_reward', 0):.1f}:1

🎯 <b>交易点位 (ATR动态):</b>
├ 当前价格: {current_price}
├ <b>入场价格: {entry_price}</b>
├ <b>止损价格: {stop_loss} ({self.format_percentage(-risk_pct) if direction == 'BUY' else self.format_percentage(risk_pct)})</b>
└ <b>止盈价格: {take_profit} ({self.format_percentage(reward_pct) if direction == 'BUY' else self.format_percentage(-reward_pct)})</b>

📈 <b>技术分析:</b>
"""
            if 'volume_ratio' in signal:
                vol_ratio = signal['volume_ratio']
                vol_emoji = '📈' if vol_ratio > 1.2 else '📊' if vol_ratio > 0.8 else '📉'
                message += f"├ 成交量比率: {vol_ratio:.1f}x {vol_emoji}\n"
            if 'atr' in signal:
                message += f"├ ATR: {signal['atr']:.4f}\n"
            if 'adx' in signal:
                message += f"├ ADX: {signal['adx']:.1f} ({signal['market_state']})\n"
            if 'bounce_pct' in signal:
                message += f"├ 反弹幅度: {signal['bounce_pct']:.1f}%\n"
            if 'callback_pct' in signal:
                message += f"├ 回调幅度: {signal['callback_pct']:.1f}%\n"
            if 'confirmation_k_info' in signal:
                conf = signal['confirmation_k_info']
                if 'entity_ratio' in conf:
                    message += f"├ 确认K实体: {conf['entity_ratio']:.0%}\n"
            message += "└ 动态止损基于ATR计算\n"

            reason = signal.get('reason', '动态ATR止损 + ADX市场过滤')
            message += f"""
📋 <b>交易理由:</b>
{reason}

⚠️ <b>风险管理:</b>
├ 止损幅度: {abs(risk_pct):.1f}%
├ 止盈幅度: {abs(reward_pct):.1f}%
└ 建议仓位: {'20-30%' if score >= 80 else '15-20%' if score >= 70 else '10-15%'}

⏰ <b>信号时效:</b>
└ 生成时间: {signal.get('signal_time', datetime.now()).strftime('%H:%M:%S')}
"""
            return message
        except Exception as e:
            return f"❌ 生成信号消息失败: {str(e)}"

    def send_top_3_signals(self, signals):
        if not signals or len(signals) == 0:
            return
        try:
            header_msg = f"""
📊 <b>本轮分析发现 {len(signals)} 个交易信号</b>
━━━━━━━━━━━━━━━━━━━━
🎯 将为您展示评分最高的前{min(3, len(signals))}个信号
⏰ 分析时间: {datetime.now().strftime('%H:%M:%S')}
━━━━━━━━━━━━━━━━━━━━
"""
            self.bot.send_message(self.chat_id, header_msg, parse_mode='HTML', disable_web_page_preview=True)
            time.sleep(1)
            for i, signal in enumerate(signals[:3]):
                detailed_message = self.create_detailed_signal_message(signal)
                if detailed_message:
                    self.bot.send_message(self.chat_id, detailed_message, parse_mode='HTML', disable_web_page_preview=True)
                    time.sleep(2)
            summary_msg = f"""
📈 <b>本轮分析完成</b>
━━━━━━━━━━━━━━━━━━━━
✅ 已发送{min(3, len(signals))}个详细交易信号
⏰ 下次分析: {UltimateConfig.ANALYSIS_INTERVAL}分钟后
━━━━━━━━━━━━━━━━━━━━
💡 <i>市场有风险，投资需谨慎</i>
"""
            self.bot.send_message(self.chat_id, summary_msg, parse_mode='HTML', disable_web_page_preview=True)
        except Exception as e:
            print(f"❌ 发送详细信号失败: {e}")

    def send_signal_message(self, signal, cooldown_status: str = ""):
        try:
            if not self.config['enabled']:
                return
            message = self.create_detailed_signal_message(signal)
            if cooldown_status:
                message += f"\n⏳ {cooldown_status}"
            self.bot.send_message(self.chat_id, message, parse_mode='HTML', disable_web_page_preview=True, disable_notification=False)
            self.message_history.append({
                'time': datetime.now(),
                'symbol': signal.get('symbol', 'UNKNOWN'),
                'pattern': signal.get('pattern', 'UNKNOWN'),
                'direction': signal.get('direction', 'BUY')
            })
            print(f"✅ Telegram信号发送成功: {signal.get('symbol', 'UNKNOWN')}")
        except Exception as e:
            print(f"❌ 发送Telegram消息失败: {e}")

    def send_batch_summary(self, signals):
        if not signals or not self.config['enabled']:
            return
        try:
            sorted_signals = sorted(signals, key=lambda x: x.get('score', 0), reverse=True)
            total_count = len(signals)
            buy_count = sum(1 for s in signals if s.get('direction') == 'BUY')
            sell_count = total_count - buy_count
            summary = f"""
📊 <b>分析周期总结</b>
<b>发现 {total_count} 个交易信号</b>
━━━━━━━━━━━━━━━━━━━━
├ 做多信号: {buy_count}个
├ 做空信号: {sell_count}个
└ 发送时间: {datetime.now().strftime('%H:%M:%S')}

🏆 <b>最佳信号 TOP 5</b>
"""
            for i, signal in enumerate(sorted_signals[:5], 1):
                symbol = signal.get('symbol', 'UNKNOWN')
                pattern = signal.get('pattern', 'UNKNOWN')
                direction = signal.get('direction', 'BUY')
                score = signal.get('score', 0)
                rsi = signal.get('rsi', 0)
                price = self.format_price(signal.get('current_price', 0))
                direction_emoji = self.get_emoji_for_direction(direction)
                pattern_emoji = self.get_emoji_for_pattern(pattern)
                score_emoji = self.get_emoji_for_score(score)
                summary += f"\n{i}. {direction_emoji} <b>{symbol}</b>\n"
                summary += f"   {pattern_emoji} {pattern} | 评分: {score} {score_emoji}\n"
                summary += f"   RSI: {rsi:.1f} | 价格: {price}\n"
            summary += "\n" + "─" * 30
            summary += f"\n⏰ 下次分析: {UltimateConfig.ANALYSIS_INTERVAL}分钟后"
            summary += f"\n📱 详细信号已单独发送"
            self.bot.send_message(self.chat_id, summary, parse_mode='HTML', disable_web_page_preview=True)
        except Exception as e:
            print(f"❌ 发送批量总结失败: {e}")

# ============ 主入口 ============
def main():
    print("="*70)
    print("🚀 终极智能交易系统 v34.0 全策略ATR/ADX适配版")
    print("="*70)

    if UltimateConfig.BACKTEST_CONFIG['enabled']:
        print("\n🔧 回测模式已启用，将运行回测，不发送Telegram通知")
        fetcher = OKXDataFetcher()
        engine = BacktestEngine(UltimateConfig.BACKTEST_CONFIG, fetcher)
        engine.run()
        print("\n✅ 回测完成")
        return

    system = UltimateTradingSystem(
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )
    if system:
        print("\n✅ 系统初始化完成！")
        if IN_GITHUB_ACTIONS:
            print("\n🔧 运行在 GitHub Actions 环境，执行单次分析后退出...")
            system.run_single_cycle()
            print("✅ 分析完成，退出。")
        else:
            print("\n🚀 自动启动连续监控模式...")
            system.run_continuous()

if __name__ == "__main__":
    main()