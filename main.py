#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极智能交易系统 v33.6 完整正式版（GitHub Actions 优化版）
功能特性：
✅ 1. 趋势衰竭做空检测器
✅ 2. HYPE暴涨原因分析器
✅ 3. 智能币种分类器
✅ 4. 反弹失败·确认K做空策略（防追尾）
✅ 5. 回调企稳·确认K做多策略（防追尾）
✅ 6. 增强Telegram通知：前3个信号详细分析
✅ 7. GitHub Actions 适配：自动从环境变量读取Telegram配置，单次运行后退出
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

# ============ 用户配置区（优先从环境变量读取）============
# Telegram 配置：如果环境变量不存在，则设为 None（Telegram 通知器会禁用）
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
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

# ============ 系统配置类 ============
class UltimateConfig:
    VERSION = "33.6-正式版-GitHub优化"
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
            'conditions': {'max_rsi': 42, 'min_volume_ratio': 0.7, 'min_score': 35}
        },
        'BREAKOUT': {
            'name': '突破模式',
            'enabled': True,
            'conditions': {'min_rsi': 45, 'max_rsi': 68, 'min_volume_ratio': 1.2, 'min_score': 30}
        },
        'BREAKOUT_FAIL_SHORT': {
            'name': '突破失败做空',
            'enabled': True,
            'conditions': {'min_rsi': 65, 'breakout_failure_threshold': 0.98, 'min_score': 35}
        },
        'TREND': {
            'name': '趋势模式',
            'enabled': True,
            'conditions': {'min_rsi': 40, 'max_rsi': 75, 'min_volume_ratio': 1.0, 'min_score': 35}
        },
        'CALLBACK': {
            'name': '回调模式',
            'enabled': True,
            'conditions': {'min_rsi': 55, 'callback_range': {'min': 5, 'max': 15}, 'min_score': 40}
        },
        'BOUNCE_FAIL_SHORT': {
            'name': '反弹失败做空',
            'enabled': True,
            'conditions': {'min_score': 45, 'max_bounce_pct': 2.0, 'lookback_periods': 10, 'fib_threshold': 38.2}
        },
        'TREND_EXHAUSTION': {
            'name': '趋势衰竭做空',
            'enabled': True,
            'conditions': {'min_score': 55, 'trend_periods': 30, 'exhaustion_threshold': 0.6, 'volume_divergence_threshold': 0.7, 'required_confirmation': 3}
        },
        'BOUNCE_FAIL_CONFIRM_K': {
            'name': '反弹失败·确认K做空',
            'enabled': True,
            'conditions': {
                'min_score': 50, 'max_bounce_count': 1, 'min_entity_ratio': 0.6,
                'max_lower_shadow_ratio': 0.2, 'required_confirmation': 2,
                'volume_requirement': 0.8, 'stop_loss_tight': 1.5, 'take_profit_ratio': 2.5
            }
        },
        'CALLBACK_CONFIRM_K': {
            'name': '回调企稳·确认K做多',
            'enabled': True,
            'conditions': {
                'min_score': 50, 'max_callback_count': 1, 'min_entity_ratio': 0.6,
                'max_upper_shadow_ratio': 0.2, 'required_confirmation': 2,
                'volume_requirement': 0.8, 'stop_loss_tight': 1.5, 'take_profit_ratio': 2.5
            }
        }
    }

    TELEGRAM_CONFIG = {
        'enabled': True,  # 将在运行时根据是否有token决定是否启用
        'parse_mode': 'HTML',
        'show_emoji': True,
        'show_details': True,
        'include_entry_exit': True,
        'include_structure_levels': True
    }

# ============ 冷却管理器 ============
class CooldownManager:
    def __init__(self):
        self.config = UltimateConfig.COOLDOWN_CONFIG
        self.cooldown_db = {}
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
                    self.signal_history = defaultdict(list, data.get('signal_history', {}))
                print(f"✅ 冷却状态已加载: {len(self.cooldown_db)}个币种记录")
        except Exception as e:
            print(f"❌ 加载冷却状态失败: {e}")
            self.cooldown_db = {}
            self.signal_history = defaultdict(list)

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

    def get_candles(self, symbol: str, interval: str):
        cache_key = f"{symbol}_{interval}"
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

# ============ 技术指标计算器 ============
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

# ============ 信号检查器基类 ============
class BaseSignalChecker:
    def __init__(self, pattern_name: str):
        self.pattern_name = pattern_name
        self.config = UltimateConfig.MARKET_MODES.get(pattern_name, {})

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

# ============ 改进的Telegram通知器（支持凭证缺失时禁用）============
class UltimateTelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.config = UltimateConfig.TELEGRAM_CONFIG.copy()
        # 如果凭证缺失，直接禁用
        if not bot_token or not chat_id:
            print("⚠️ Telegram 凭证未提供，通知器已禁用")
            self.config['enabled'] = False
            return
        try:
            self.bot = telebot.TeleBot(bot_token)
            self.chat_id = chat_id
            self.message_history = deque(maxlen=100)
            self.test_connection()
            self.send_startup_message()
        except Exception as e:
            print(f"❌ Telegram 初始化失败: {e}")
            self.config['enabled'] = False

    def send_startup_message(self):
        if not self.config.get('enabled', False):
            return
        try:
            startup_msg = f"""
🚀 <b>终极智能交易系统 v{UltimateConfig.VERSION} 已启动！</b>
━━━━━━━━━━━━━━━━━━━━
📅 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
📊 监控币种: {len(MONITOR_COINS)}个
⏰ 分析间隔: {UltimateConfig.ANALYSIS_INTERVAL}分钟
🔍 分析周期: {', '.join(UltimateConfig.MULTI_TIMEFRAME_CONFIG['timeframes'])}
🤖 交易模式: 9种（含新增确认K策略）

🎯 <b>系统已开始扫描市场机会...</b>
首次分析将在3秒后开始
━━━━━━━━━━━━━━━━━━━━
<code>系统状态: ✅ 运行中</code>
"""
            self.bot.send_message(self.chat_id, startup_msg, parse_mode='HTML', disable_web_page_preview=True)
            print("✅ 系统启动消息已发送到Telegram")
        except Exception as e:
            print(f"❌ 发送启动消息失败: {e}")
            self.config['enabled'] = False

    def test_connection(self):
        if not self.config.get('enabled', False):
            return
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
            'BOUNCE': '🔺',
            'BREAKOUT': '🚀',
            'BREAKOUT_FAIL_SHORT': '🔻',
            'TREND': '📈',
            'CALLBACK': '📉',
            'BOUNCE_FAIL_SHORT': '⚡',
            'TREND_EXHAUSTION': '🔥',
            'BOUNCE_FAIL_CONFIRM_K': '🎯',
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

🎯 <b>交易点位:</b>
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

            if 'callback_pct' in signal:
                callback_pct = signal['callback_pct']
                message += f"├ 回调幅度: {callback_pct:.1f}%\n"

            if 'bounce_pct' in signal:
                bounce_pct = signal['bounce_pct']
                message += f"├ 反弹幅度: {bounce_pct:.1f}%\n"

            if 'confirmation_k_info' in signal:
                conf_info = signal['confirmation_k_info']
                if pattern in ['BOUNCE_FAIL_CONFIRM_K', 'CALLBACK_CONFIRM_K']:
                    message += f"├ 确认K实体: {conf_info.get('entity_ratio', 0):.0%}\n"
                    if direction == 'SELL':
                        message += f"└ 下影线比例: {conf_info.get('lower_shadow_ratio', 0):.0%}\n"
                    else:
                        message += f"└ 上影线比例: {conf_info.get('upper_shadow_ratio', 0):.0%}\n"
                else:
                    message += "└ 确认K信号有效\n"
            else:
                message += "└ 技术指标支持信号\n"

            reason = signal.get('reason', '')
            if not reason:
                if pattern == 'BOUNCE':
                    reason = f"RSI超卖({rsi:.1f})后的反弹机会，当前价格处于支撑位附近"
                elif pattern == 'BREAKOUT':
                    reason = f"价格突破关键阻力位，成交量放大{vol_ratio:.1f}倍，突破有效性高"
                elif pattern == 'BOUNCE_FAIL_CONFIRM_K':
                    reason = f"反弹失败后出现确认K线，实体比例{conf_info.get('entity_ratio', 0):.0%}，做空信号明确"
                elif pattern == 'CALLBACK_CONFIRM_K':
                    reason = f"回调至支撑位企稳，出现确认K线，实体比例{conf_info.get('entity_ratio', 0):.0%}，做多信号明确"
                else:
                    reason = "技术分析显示明确的交易机会"

            message += f"""
📋 <b>交易理由:</b>
{reason}

⚠️ <b>风险管理:</b>
├ 止损幅度: {abs(risk_pct):.1f}%
├ 止盈幅度: {abs(reward_pct):.1f}%
└ 建议仓位: {'20-30%' if score >= 80 else '15-20%' if score >= 70 else '10-15%'}

🚀 <b>操作建议:</b>
{'1️⃣ 立即买入做多，严格止损' if direction == 'BUY' else '1️⃣ 立即卖出做空，严格止损'}
2️⃣ 入场价格: {entry_price}
3️⃣ 止损位置: {stop_loss}
4️⃣ 止盈位置: {take_profit}
5️⃣ 分批入场，控制风险

⏰ <b>信号时效:</b>
└ 生成时间: {signal.get('signal_time', datetime.now()).strftime('%H:%M:%S')}
"""
            return message
        except Exception as e:
            return f"❌ 生成信号消息失败: {str(e)}"

    def send_top_3_signals(self, signals):
        if not signals or len(signals) == 0 or not self.config.get('enabled', False):
            return
        try:
            print(f"📤 准备发送前{min(3, len(signals))}个详细信号到Telegram...")
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
                    try:
                        self.bot.send_message(self.chat_id, detailed_message, parse_mode='HTML', disable_web_page_preview=True)
                        print(f"✅ 已发送第{i+1}个信号: {signal.get('symbol')}")
                        time.sleep(2)
                    except Exception as e:
                        print(f"❌ 发送第{i+1}个信号失败: {e}")

            summary_msg = f"""
📈 <b>本轮分析完成</b>
━━━━━━━━━━━━━━━━━━━━
✅ 已发送{min(3, len(signals))}个详细交易信号
⏰ 下次分析: {UltimateConfig.ANALYSIS_INTERVAL}分钟后
📊 系统状态: ✅ 持续监控中
━━━━━━━━━━━━━━━━━━━━
💡 <i>温馨提示: 市场有风险，投资需谨慎</i>
"""
            self.bot.send_message(self.chat_id, summary_msg, parse_mode='HTML', disable_web_page_preview=True)
            print("✅ 前3个详细信号已成功发送到Telegram")
        except Exception as e:
            print(f"❌ 发送详细信号失败: {e}")

    def send_signal_message(self, signal, cooldown_status: str = ""):
        if not self.config.get('enabled', False):
            return
        try:
            message = self.create_detailed_signal_message(signal)
            if cooldown_status:
                message += f"\n⏳ {cooldown_status}"
            self.bot.send_message(self.chat_id, message, parse_mode='HTML', disable_web_page_preview=True, disable_notification=False)
            self.message_history.append({'time': datetime.now(), 'symbol': signal.get('symbol', 'UNKNOWN'), 'pattern': signal.get('pattern', 'UNKNOWN'), 'direction': signal.get('direction', 'BUY')})
            print(f"✅ Telegram信号发送成功: {signal.get('symbol', 'UNKNOWN')}")
        except Exception as e:
            print(f"❌ 发送Telegram消息失败: {e}")

    def send_batch_summary(self, signals):
        if not signals or not self.config.get('enabled', False):
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
            print("✅ 批量总结发送成功")
        except Exception as e:
            print(f"❌ 发送批量总结失败: {e}")

# ============ 反弹失败·确认K做空策略检查器 ============
class BounceFailConfirmKShortChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('BOUNCE_FAIL_CONFIRM_K')
        self.conditions = self.config.get('conditions', {})
        self.kline_analyzer = KLineAnalyzer()
        self.min_score = self.conditions.get('min_score', 50)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data_15m = data_dict['15m']
            if len(data_15m) < 50:
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
            if not self.kline_analyzer.is_confirmation_candle(kline_analysis, 'SELL', self.conditions):
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
            stop_loss = self._calculate_stop_loss(data_15m, bounce_structure, entry_price)
            take_profit = self._calculate_take_profit(entry_price, stop_loss)
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
            if risk_reward < 1.5:
                return None
            return {
                'symbol': symbol, 'pattern': 'BOUNCE_FAIL_CONFIRM_K', 'direction': 'SELL',
                'score': int(score), 'current_price': entry_price, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': round(risk_reward, 2),
                'rsi': indicators_info.get('rsi', 50), 'volume_ratio': indicators_info.get('volume_ratio', 1),
                'confirmation_k_info': {
                    'is_bullish': kline_analysis['is_bullish'],
                    'entity_ratio': round(kline_analysis['entity_ratio'], 2),
                    'lower_shadow_ratio': round(kline_analysis['lower_shadow_ratio'], 2),
                    'body_size': round(kline_analysis['body_size'], 4)
                },
                'is_first_confirmation_k': True, 'signal_time': datetime.now(),
                'signal_type': 'SELL', 'confidence': 'HIGH' if score > 70 else 'MEDIUM',
                'reason': f"反弹失败确认K做空 | 反弹幅度:{bounce_structure['bounce_pct']:.1f}% | 确认K实体:{kline_analysis['entity_ratio']:.0%} | 第1根确认K"
            }
        except Exception as e:
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
        current_price = data['close'].iloc[-1]
        return {
            'bounce_low': recent_low, 'bounce_high': bounce_high, 'bounce_pct': bounce_pct,
            'current_price': current_price, 'fib_levels': fib_levels,
            'bounce_start_index': data.index.get_loc(recent_low_idx),
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
        bounce_peak_idx = bounce_structure['bounce_peak_index']
        if bounce_peak_idx < len(data) - 1:
            peak_kline = data.iloc[bounce_peak_idx]
            kline_analysis = self.kline_analyzer.analyze_candle(peak_kline)
            if kline_analysis and kline_analysis['upper_shadow_ratio'] > 0.3:
                return True
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
                if kline_analysis['entity_ratio'] >= self.conditions.get('min_entity_ratio', 0.6):
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
        current_price = data['close'].iloc[-1]
        if current_price > ma20 * 1.05:
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

    def _calculate_stop_loss(self, data: pd.DataFrame, bounce_structure: Dict, entry_price: float) -> float:
        bounce_high = bounce_structure['bounce_high']
        stop_loss = bounce_high * 1.01
        stop_loss_pct = (stop_loss - entry_price) / entry_price * 100
        if stop_loss_pct > 3:
            stop_loss = entry_price * 1.03
        return round(stop_loss, 6)

    def _calculate_take_profit(self, entry_price: float, stop_loss: float) -> float:
        risk = stop_loss - entry_price
        take_profit = entry_price - risk * self.conditions.get('take_profit_ratio', 2.5)
        return round(take_profit, 6)

# ============ 回调企稳·确认K做多策略检查器 ============
class CallbackConfirmKBuyChecker(BaseSignalChecker):
    def __init__(self):
        super().__init__('CALLBACK_CONFIRM_K')
        self.conditions = self.config.get('conditions', {})
        self.kline_analyzer = KLineAnalyzer()
        self.min_score = self.conditions.get('min_score', 50)

    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict:
                return None
            data_15m = data_dict['15m']
            if len(data_15m) < 50:
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
            if not self.kline_analyzer.is_confirmation_candle(kline_analysis, 'BUY', self.conditions):
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
            stop_loss = self._calculate_stop_loss(data_15m, callback_structure, entry_price)
            take_profit = self._calculate_take_profit(entry_price, stop_loss)
            risk_reward = (take_profit - entry_price) / (entry_price - stop_loss)
            if risk_reward < 1.5:
                return None
            return {
                'symbol': symbol, 'pattern': 'CALLBACK_CONFIRM_K', 'direction': 'BUY',
                'score': int(score), 'current_price': entry_price, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': round(risk_reward, 2),
                'rsi': indicators_info.get('rsi', 50), 'volume_ratio': indicators_info.get('volume_ratio', 1),
                'confirmation_k_info': {
                    'is_bullish': kline_analysis['is_bullish'],
                    'entity_ratio': round(kline_analysis['entity_ratio'], 2),
                    'upper_shadow_ratio': round(kline_analysis['upper_shadow_ratio'], 2),
                    'body_size': round(kline_analysis['body_size'], 4)
                },
                'is_first_confirmation_k': True, 'signal_time': datetime.now(),
                'signal_type': 'BUY', 'confidence': 'HIGH' if score > 70 else 'MEDIUM',
                'reason': f"回调企稳确认K做多 | 回调幅度:{callback_structure['callback_pct']:.1f}% | 确认K实体:{kline_analysis['entity_ratio']:.0%} | 第1根确认K"
            }
        except Exception as e:
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
        current_price = data['close'].iloc[-1]
        support_levels = self._find_support_levels(data, callback_low_idx)
        return {
            'callback_high': recent_high, 'callback_low': callback_low, 'callback_pct': callback_pct,
            'current_price': current_price, 'fib_levels': fib_levels, 'support_levels': support_levels,
            'callback_start_index': data.index.get_loc(recent_high_idx),
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
        callback_low_idx = callback_structure['callback_low_index']
        if callback_low_idx < len(data) - 1:
            low_kline = data.iloc[callback_low_idx]
            kline_analysis = self.kline_analyzer.analyze_candle(low_kline)
            if kline_analysis:
                if kline_analysis['lower_shadow_ratio'] > 0.3 or kline_analysis['is_doji']:
                    return True
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
                if kline_analysis['entity_ratio'] >= self.conditions.get('min_entity_ratio', 0.6):
                    if i == current_idx and bullish_count == 1:
                        return True
        return False

    def _find_support_levels(self, data: pd.DataFrame, callback_low_idx: int) -> List[float]:
        support_levels = []
        lookback_data = data.iloc[:callback_low_idx]
        if len(lookback_data) > 10:
            prev_lows = lookback_data['low'].rolling(window=10).min()
            significant_lows = prev_lows[prev_lows < data['low'].iloc[callback_low_idx] * 1.05]
            if not significant_lows.empty:
                support_levels.append(float(significant_lows.iloc[-1]))
        if len(data) >= 20:
            ma20 = TechnicalIndicatorsMultiTF.calculate_ma(data, 20).iloc[callback_low_idx]
            support_levels.append(float(ma20))
        if len(data) >= 50:
            ma50 = TechnicalIndicatorsMultiTF.calculate_ma(data, 50).iloc[callback_low_idx]
            support_levels.append(float(ma50))
        return sorted(support_levels)

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
        current_price = data['close'].iloc[-1]
        if current_price < ma20 * 0.95:
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

    def _calculate_stop_loss(self, data: pd.DataFrame, callback_structure: Dict, entry_price: float) -> float:
        callback_low = callback_structure['callback_low']
        stop_loss = callback_low * 0.99
        stop_loss_pct = (entry_price - stop_loss) / entry_price * 100
        if stop_loss_pct > 3:
            stop_loss = entry_price * 0.97
        return round(stop_loss, 6)

    def _calculate_take_profit(self, entry_price: float, stop_loss: float) -> float:
        risk = entry_price - stop_loss
        take_profit = entry_price + risk * self.conditions.get('take_profit_ratio', 2.5)
        return round(take_profit, 6)

# ============ 其他策略检查器（简化版）============
class BounceSignalChecker(BaseSignalChecker):
    def __init__(self): super().__init__('BOUNCE')
    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            indicators_dict = self.get_multi_timeframe_indicators(data_dict)
            if not indicators_dict or '15m' not in indicators_dict:
                return None
            indicators = indicators_dict['15m']
            data_15m = data_dict['15m']
            current_price = data_15m['close'].iloc[-1]
            current_rsi = indicators['rsi']
            if pd.isna(current_rsi) or current_rsi > self.config.get('conditions', {}).get('max_rsi', 42):
                return None
            current_volume_ratio = indicators['volume_ratio']
            if pd.isna(current_volume_ratio) or current_volume_ratio < self.config.get('conditions', {}).get('min_volume_ratio', 0.7):
                return None
            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(indicators_dict, 'BUY')
            if consensus < 0.6:
                return None
            score = 45
            if current_rsi < 30: score += 25
            elif current_rsi < 35: score += 20
            elif current_rsi < 40: score += 15
            if current_volume_ratio > 1.5: score += 20
            elif current_volume_ratio > 1.2: score += 15
            elif current_volume_ratio > 0.8: score += 10
            score = min(100, score + (consensus * 20))
            if score < self.config.get('conditions', {}).get('min_score', 35):
                return None
            entry_price = current_price
            stop_loss = current_price * 0.96
            take_profit = current_price * 1.08
            return {
                'symbol': symbol, 'pattern': 'BOUNCE', 'direction': 'BUY',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(current_volume_ratio, 2), 'current_price': current_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round((take_profit - entry_price) / (entry_price - stop_loss), 1),
                'signal_time': datetime.now(), 'signal_type': 'BUY',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM'
            }
        except: return None

class BreakoutSignalChecker(BaseSignalChecker):
    def __init__(self): super().__init__('BREAKOUT')
    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            indicators_dict = self.get_multi_timeframe_indicators(data_dict)
            if not indicators_dict or '15m' not in indicators_dict:
                return None
            indicators = indicators_dict['15m']
            data_15m = data_dict['15m']
            current_price = data_15m['close'].iloc[-1]
            current_rsi = indicators['rsi']
            conditions = self.config.get('conditions', {})
            if pd.isna(current_rsi) or not (conditions.get('min_rsi', 45) <= current_rsi <= conditions.get('max_rsi', 68)):
                return None
            current_volume_ratio = indicators['volume_ratio']
            if pd.isna(current_volume_ratio) or current_volume_ratio < conditions.get('min_volume_ratio', 1.2):
                return None
            resistance = data_15m['high'].rolling(window=20).max().iloc[-2]
            if current_price < resistance * 1.015:
                return None
            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(indicators_dict, 'BUY')
            if consensus < 0.6:
                return None
            score = 50
            if 55 <= current_rsi <= 65: score += 25
            elif 45 <= current_rsi <= 55: score += 20
            elif 65 <= current_rsi <= 68: score += 15
            if current_volume_ratio > 2.0: score += 25
            elif current_volume_ratio > 1.5: score += 20
            elif current_volume_ratio > 1.2: score += 15
            breakout_strength = (current_price - resistance) / resistance * 100
            if breakout_strength > 2: score += 20
            elif breakout_strength > 1: score += 15
            score = min(100, score + (consensus * 20))
            if score < conditions.get('min_score', 30):
                return None
            entry_price = current_price
            stop_loss = resistance * 0.985
            take_profit = entry_price + (entry_price - stop_loss) * 3
            return {
                'symbol': symbol, 'pattern': 'BREAKOUT', 'direction': 'BUY',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(current_volume_ratio, 2), 'current_price': current_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round((take_profit - entry_price) / (entry_price - stop_loss), 1),
                'signal_time': datetime.now(), 'signal_type': 'BUY',
                'confidence': 'HIGH' if score > 70 else 'MEDIUM'
            }
        except: return None

class BreakoutFailShortChecker(BaseSignalChecker):
    def __init__(self): super().__init__('BREAKOUT_FAIL_SHORT')
    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            indicators_dict = self.get_multi_timeframe_indicators(data_dict)
            if not indicators_dict or '15m' not in indicators_dict:
                return None
            indicators = indicators_dict['15m']
            data_15m = data_dict['15m']
            current_price = data_15m['close'].iloc[-1]
            current_rsi = indicators['rsi']
            conditions = self.config.get('conditions', {})
            if pd.isna(current_rsi) or current_rsi < conditions.get('min_rsi', 65):
                return None
            recent_high = data_15m['high'].rolling(window=10).max().iloc[-2]
            if current_price > recent_high * 0.99:
                return None
            support = data_15m['low'].rolling(window=10).min().iloc[-2]
            if current_price > support * 1.02:
                return None
            current_volume_ratio = indicators['volume_ratio']
            if pd.isna(current_volume_ratio) or current_volume_ratio < 0.8:
                return None
            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(indicators_dict, 'SELL')
            if consensus < 0.6:
                return None
            score = 40
            if current_rsi > 75: score += 30
            elif current_rsi > 70: score += 25
            elif current_rsi > 65: score += 20
            if current_volume_ratio > 1.2: score += 20
            elif current_volume_ratio > 1.0: score += 15
            elif current_volume_ratio > 0.8: score += 10
            price_change = (data_15m['close'].iloc[-1] - data_15m['close'].iloc[-5]) / data_15m['close'].iloc[-5] * 100
            if price_change < -2: score += 15
            elif price_change < -1: score += 10
            ma20 = TechnicalIndicatorsMultiTF.calculate_ma(data_15m, 20).iloc[-1]
            if current_price < ma20: score += 10
            score = min(100, score + (consensus * 20))
            if score < conditions.get('min_score', 35):
                return None
            entry_price = current_price
            stop_loss = min(recent_high * 1.01, entry_price * 1.02)
            take_profit = entry_price * 0.96
            return {
                'symbol': symbol, 'pattern': 'BREAKOUT_FAIL_SHORT', 'direction': 'SELL',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(current_volume_ratio, 2), 'current_price': current_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round((entry_price - take_profit) / (stop_loss - entry_price), 1),
                'signal_time': datetime.now(), 'signal_type': 'SELL',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"高RSI({current_rsi:.1f}) + 突破失败 + 跌破支撑"
            }
        except: return None

class TrendSignalChecker(BaseSignalChecker):
    def __init__(self): super().__init__('TREND')
    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            indicators_dict = self.get_multi_timeframe_indicators(data_dict)
            if not indicators_dict:
                return None
            trend_scores = {}
            for tf, indicators in indicators_dict.items():
                if 'rsi' in indicators and 'ma20' in indicators and 'ma50' in indicators:
                    current_price = data_dict[tf]['close'].iloc[-1]
                    rsi = indicators['rsi']
                    ma20 = indicators['ma20']
                    ma50 = indicators['ma50']
                    conditions = self.config.get('conditions', {})
                    is_uptrend = (current_price > ma20 > ma50 and
                                 conditions.get('min_rsi', 40) <= rsi <= conditions.get('max_rsi', 75))
                    trend_scores[tf] = {'is_uptrend': is_uptrend}
            uptrend_count = sum(1 for tf_score in trend_scores.values() if tf_score['is_uptrend'])
            if uptrend_count < max(1, len(trend_scores) * 0.5):
                return None
            if '15m' not in indicators_dict:
                return None
            indicators = indicators_dict['15m']
            data_15m = data_dict['15m']
            current_price = data_15m['close'].iloc[-1]
            current_rsi = indicators['rsi']
            current_volume_ratio = indicators['volume_ratio']
            conditions = self.config.get('conditions', {})
            if pd.isna(current_rsi) or not (conditions.get('min_rsi', 40) <= current_rsi <= conditions.get('max_rsi', 75)):
                return None
            if pd.isna(current_volume_ratio) or current_volume_ratio < conditions.get('min_volume_ratio', 1.0):
                return None
            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(indicators_dict, 'BUY')
            if consensus < 0.6:
                return None
            score = 60
            if 50 <= current_rsi <= 70: score += 20
            elif 40 <= current_rsi <= 50: score += 15
            if current_volume_ratio > 1.5: score += 20
            elif current_volume_ratio > 1.2: score += 15
            ma20 = indicators['ma20']
            ma50 = indicators['ma50']
            trend_strength = (ma20 - ma50) / ma50 * 100
            if trend_strength > 5: score += 15
            elif trend_strength > 3: score += 10
            score = min(100, score + (consensus * 20) + (uptrend_count * 5))
            if score < conditions.get('min_score', 35):
                return None
            entry_price = current_price
            stop_loss = ma20 * 0.97
            take_profit = entry_price + (entry_price - stop_loss) * 2.5
            return {
                'symbol': symbol, 'pattern': 'TREND', 'direction': 'BUY',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(current_volume_ratio, 2), 'current_price': current_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round((take_profit - entry_price) / (entry_price - stop_loss), 1),
                'signal_time': datetime.now(), 'signal_type': 'BUY',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM'
            }
        except: return None

class CallbackSignalChecker(BaseSignalChecker):
    def __init__(self): super().__init__('CALLBACK')
    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            indicators_dict = self.get_multi_timeframe_indicators(data_dict)
            if not indicators_dict or '15m' not in indicators_dict:
                return None
            indicators = indicators_dict['15m']
            data_15m = data_dict['15m']
            current_price = data_15m['close'].iloc[-1]
            current_rsi = indicators['rsi']
            conditions = self.config.get('conditions', {})
            if pd.isna(current_rsi) or current_rsi < conditions.get('min_rsi', 55):
                return None
            lookback_periods = [20, 30, 50]
            recent_highs = []
            for period in lookback_periods:
                if len(data_15m) >= period:
                    recent_high = data_15m['high'].iloc[-period:].max()
                    if recent_high > current_price:
                        recent_highs.append(recent_high)
            if not recent_highs:
                return None
            recent_high = max(recent_highs)
            callback_pct = ((recent_high - current_price) / recent_high) * 100
            callback_range = conditions.get('callback_range', {'min': 5, 'max': 15})
            if not (callback_range['min'] <= callback_pct <= callback_range['max']):
                return None
            ma20 = indicators['ma20']
            if current_price < ma20 * 0.95:
                return None
            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(indicators_dict, 'BUY')
            if consensus < 0.6:
                return None
            score = 50
            if 60 <= current_rsi <= 70: score += 20
            elif current_rsi > 70: score += 15
            if 8 <= callback_pct <= 12: score += 25
            elif 5 <= callback_pct <= 15: score += 20
            current_volume = data_15m['volume'].iloc[-1]
            avg_volume = data_15m['volume'].rolling(window=20).mean().iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0
            if volume_ratio > 0.8: score += 15
            if current_price > ma20: score += 20
            score = min(max(score, 0), 100)
            score = min(100, score + (consensus * 20))
            if score < conditions.get('min_score', 40):
                return None
            entry_price = current_price
            stop_loss = current_price * 0.95
            take_profit = current_price * 1.10
            return {
                'symbol': symbol, 'pattern': 'CALLBACK', 'direction': 'BUY',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'callback_pct': round(callback_pct, 1), 'current_price': current_price,
                'entry_price': entry_price, 'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': round((take_profit - entry_price) / (entry_price - stop_loss), 1),
                'volume_ratio': round(volume_ratio, 2), 'signal_time': datetime.now(),
                'signal_type': 'BUY', 'confidence': 'HIGH' if score > 75 else 'MEDIUM'
            }
        except: return None

class BounceFailShortChecker(BaseSignalChecker):
    def __init__(self): super().__init__('BOUNCE_FAIL_SHORT')
    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            indicators_dict = self.get_multi_timeframe_indicators(data_dict)
            if not indicators_dict or '15m' not in indicators_dict:
                return None
            indicators = indicators_dict['15m']
            data_15m = data_dict['15m']
            current_price = data_15m['close'].iloc[-1]
            current_rsi = indicators['rsi']
            rsi_history = TechnicalIndicatorsMultiTF.calculate_rsi(data_15m, 14)
            rsi_below_30 = rsi_history < 30
            if not rsi_below_30.any():
                return None
            last_rsi_below_30_idx = None
            for i in range(len(rsi_history) - 1, -1, -1):
                if rsi_below_30.iloc[i]:
                    last_rsi_below_30_idx = i
                    break
            conditions = self.config.get('conditions', {})
            if last_rsi_below_30_idx is None or len(data_15m) - last_rsi_below_30_idx > conditions.get('lookback_periods', 10):
                return None
            low_price = data_15m['low'].iloc[last_rsi_below_30_idx]
            high_after_low = data_15m['high'].iloc[last_rsi_below_30_idx:].max()
            fib_levels = TechnicalIndicatorsMultiTF.calculate_fibonacci_levels(high_after_low, low_price)
            fib_38_2 = fib_levels['38.2%']
            bounce_pct = (high_after_low - low_price) / low_price * 100
            condition_a = bounce_pct < conditions.get('max_bounce_pct', 2.0)
            condition_b = current_price < fib_38_2
            if not (condition_a or condition_b):
                return None
            if current_price > high_after_low * 0.99:
                return None
            volume_ratio = indicators['volume_ratio']
            if volume_ratio > 1.5:
                return None
            consensus, _ = TechnicalIndicatorsMultiTF.get_multi_timeframe_consensus(indicators_dict, 'SELL')
            if consensus < 0.7:
                return None
            score = 50
            if bounce_pct < 1.0: score += 30
            elif bounce_pct < 1.5: score += 25
            elif bounce_pct < 2.0: score += 20
            if current_rsi < 35: score += 20
            elif current_rsi < 40: score += 15
            elif current_rsi < 45: score += 10
            if volume_ratio < 0.7: score += 20
            elif volume_ratio < 0.9: score += 15
            elif volume_ratio < 1.1: score += 10
            distance_to_fib = abs(current_price - fib_38_2) / fib_38_2 * 100
            if distance_to_fib > 2: score += 15
            elif distance_to_fib > 1: score += 10
            if current_price < fib_38_2: score += 15
            if condition_a and condition_b: score += 25
            elif condition_a or condition_b: score += 15
            score = min(100, score + (consensus * 30))
            if score < conditions.get('min_score', 45):
                return None
            entry_price = current_price
            stop_loss = min(high_after_low * 1.015, entry_price * 1.025)
            take_profit = entry_price * 0.95
            risk_reward = round((entry_price - take_profit) / (stop_loss - entry_price), 1)
            if risk_reward < 1.5:
                return None
            return {
                'symbol': symbol, 'pattern': 'BOUNCE_FAIL_SHORT', 'direction': 'SELL',
                'score': int(score), 'rsi': round(float(current_rsi), 1),
                'volume_ratio': round(volume_ratio, 2), 'bounce_pct': round(bounce_pct, 2),
                'current_price': current_price, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit,
                'risk_reward': risk_reward, 'signal_time': datetime.now(),
                'signal_type': 'SELL', 'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"反弹乏力({bounce_pct:.1f}%) + 未触38.2%回撤 + 双失败共振"
            }
        except: return None

class TrendExhaustionShortChecker(BaseSignalChecker):
    def __init__(self): super().__init__('TREND_EXHAUSTION')
    def check_coin_multi_tf(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        try:
            if '15m' not in data_dict or '1H' not in data_dict:
                return None
            data_15m = data_dict['15m']
            data_1h = data_dict['1H']
            if len(data_15m) < 50:
                return None
            current_price = data_15m['close'].iloc[-1]
            price_30_periods_ago = data_15m['close'].iloc[-30]
            price_increase = (current_price - price_30_periods_ago) / price_30_periods_ago * 100
            if price_increase < 15:
                return None
            conditions = self.config.get('conditions', {})
            confirmation_signals = 0
            exhaustion_signals = []
            rsi_15m = TechnicalIndicatorsMultiTF.calculate_rsi(data_15m, 14)
            recent_high_idx = data_15m['high'].iloc[-20:].idxmax()
            recent_rsi_high = rsi_15m.loc[recent_high_idx]
            current_rsi = rsi_15m.iloc[-1]
            if current_price > data_15m['high'].loc[recent_high_idx] and current_rsi < recent_rsi_high:
                confirmation_signals += 1
                exhaustion_signals.append("RSI顶背离")
            volume_ma10 = data_15m['volume'].rolling(window=10).mean()
            volume_ma20 = data_15m['volume'].rolling(window=20).mean()
            if volume_ma10.iloc[-1] < volume_ma20.iloc[-1] * 0.8:
                confirmation_signals += 1
                exhaustion_signals.append("成交量递减")
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
            macd_diff = TechnicalIndicatorsMultiTF.calculate_macd(data_15m)
            if len(macd_diff) > 10:
                recent_macd_high_idx = macd_diff.iloc[-20:].idxmax()
                recent_price_high_idx = data_15m['high'].iloc[-20:].idxmax()
                if (current_price > data_15m['high'].loc[recent_price_high_idx] and
                    macd_diff.iloc[-1] < macd_diff.loc[recent_macd_high_idx]):
                    confirmation_signals += 1
                    exhaustion_signals.append("MACD背离")
            recent_kline = data_15m.iloc[-1]
            if (recent_kline['high'] - max(recent_kline['open'], recent_kline['close'])) > (
                abs(recent_kline['close'] - recent_kline['open']) * 1.5):
                confirmation_signals += 1
                exhaustion_signals.append("长上影线")
            if confirmation_signals < conditions.get('required_confirmation', 3):
                return None
            rsi_1h = TechnicalIndicatorsMultiTF.calculate_rsi(data_1h, 14)
            if rsi_1h.iloc[-1] > 70:
                confirmation_signals += 1
                exhaustion_signals.append("1H周期超买")
            score = 40 + (confirmation_signals * 15)
            if score < conditions.get('min_score', 55):
                return None
            entry_price = current_price
            recent_high = data_15m['high'].iloc[-20:].max()
            stop_loss = recent_high * 1.01
            take_profit = entry_price * 0.93
            risk_reward = (entry_price - take_profit) / (stop_loss - entry_price)
            if risk_reward < 2:
                return None
            return {
                'symbol': symbol, 'pattern': 'TREND_EXHAUSTION', 'direction': 'SELL',
                'rsi': round(float(current_rsi), 1), 'price_increase': round(price_increase, 1),
                'confirmation_signals': confirmation_signals, 'exhaustion_signals': exhaustion_signals,
                'score': int(score), 'current_price': current_price, 'entry_price': entry_price,
                'stop_loss': stop_loss, 'take_profit': take_profit, 'risk_reward': round(risk_reward, 1),
                'signal_time': datetime.now(), 'signal_type': 'SELL',
                'confidence': 'HIGH' if score > 75 else 'MEDIUM',
                'reason': f"趋势衰竭({confirmation_signals}个确认信号): {', '.join(exhaustion_signals)}"
            }
        except: return None

# ============ HYPE分析器 ============
class HypeAnalyzer:
    def __init__(self, data_fetcher):
        self.hype_symbol = 'hype'
        self.data_fetcher = data_fetcher

    def analyze_hype_surge(self):
        print(f"\n🔍 开始分析 {self.hype_symbol} 暴涨原因...")
        data_dict = self.data_fetcher.get_multi_timeframe_data(self.hype_symbol)
        if not data_dict:
            return None
        analysis_result = {
            'symbol': self.hype_symbol, 'analysis_time': datetime.now(),
            'factors': [], 'strength_score': 0,
            'sustainability': '未知', 'risk_level': '未知'
        }
        if '15m' in data_dict:
            data_15m = data_dict['15m']
            if len(data_15m) >= 24:
                price_6h_ago = data_15m['close'].iloc[-24]
                current_price = data_15m['close'].iloc[-1]
                increase_6h = (current_price - price_6h_ago) / price_6h_ago * 100
                analysis_result['price_6h_ago'] = price_6h_ago
                analysis_result['current_price'] = current_price
                analysis_result['increase_6h'] = round(increase_6h, 2)
                if increase_6h > 20:
                    analysis_result['factors'].append(f"6小时暴涨{increase_6h:.1f}%")
                    analysis_result['strength_score'] += 30
        if '1H' in data_dict:
            data_1h = data_dict['1H']
            if len(data_1h) >= 24:
                current_volume = data_1h['volume'].iloc[-1]
                avg_volume_24h = data_1h['volume'].rolling(window=24).mean().iloc[-1]
                volume_ratio = current_volume / avg_volume_24h if avg_volume_24h > 0 else 0
                analysis_result['volume_ratio'] = round(volume_ratio, 2)
                if volume_ratio > 5:
                    analysis_result['factors'].append(f"成交量放大{volume_ratio:.1f}倍")
                    analysis_result['strength_score'] += 25
                elif volume_ratio > 3:
                    analysis_result['factors'].append(f"成交量放大{volume_ratio:.1f}倍")
                    analysis_result['strength_score'] += 15
        rsi_15m = TechnicalIndicatorsMultiTF.calculate_rsi(data_15m, 14)
        current_rsi = rsi_15m.iloc[-1] if not pd.isna(rsi_15m.iloc[-1]) else 50
        analysis_result['rsi'] = round(float(current_rsi), 1)
        if current_rsi > 80:
            analysis_result['factors'].append(f"RSI严重超买({current_rsi:.1f})")
            analysis_result['strength_score'] -= 20
            analysis_result['risk_level'] = '极高'
        elif current_rsi > 70:
            analysis_result['factors'].append(f"RSI超买({current_rsi:.1f})")
            analysis_result['strength_score'] -= 10
            analysis_result['risk_level'] = '高'
        if len(data_15m) >= 48:
            returns = data_15m['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(24 * 365) * 100
            analysis_result['volatility'] = round(volatility, 2)
            if volatility > 200:
                analysis_result['factors'].append(f"极端高波动({volatility:.0f}%)")
                analysis_result['risk_level'] = '极高'
            elif volatility > 150:
                analysis_result['factors'].append(f"高波动({volatility:.0f}%)")
        recent_kline = data_15m.iloc[-1]
        upper_shadow = recent_kline['high'] - max(recent_kline['open'], recent_kline['close'])
        body = abs(recent_kline['close'] - recent_kline['open'])
        if body > 0 and upper_shadow / body > 2:
            analysis_result['factors'].append("出现长上影线（抛压迹象）")
            analysis_result['strength_score'] -= 15
            analysis_result['sustainability'] = '低'
        if analysis_result['strength_score'] >= 40:
            analysis_result['sustainability'] = '高'
            analysis_result['risk_level'] = '中'
        elif analysis_result['strength_score'] >= 20:
            analysis_result['sustainability'] = '中'
            analysis_result['risk_level'] = '中高'
        else:
            analysis_result['sustainability'] = '低'
            analysis_result['risk_level'] = '高'
        return analysis_result

    def generate_hype_report(self):
        analysis = self.analyze_hype_surge()
        if not analysis:
            return None
        report = f"""
🔴🟡🟢 HYPE暴涨深度分析报告 🔴🟡🟢
📊 基础数据
├ 币种: {analysis['symbol']}/USDT
├ 当前价格: ${analysis.get('current_price', 'N/A'):.6f}
├ 6小时涨幅: {analysis.get('increase_6h', 0):.1f}%
├ RSI: {analysis.get('rsi', 0):.1f}
├ 成交量倍数: {analysis.get('volume_ratio', 0):.1f}x
├ 波动率: {analysis.get('volatility', 0):.1f}%
└ 分析时间: {analysis['analysis_time'].strftime('%H:%M:%S')}
🎯 暴涨因素分析
"""
        for factor in analysis.get('factors', []):
            report += f"├ {factor}\n"
        report += f"""
📈 综合评估
├ 强度评分: {analysis['strength_score']}/100
├ 可持续性: {analysis['sustainability']}
├ 风险等级: {analysis['risk_level']}
├ 适合操作: {'追涨（谨慎）' if analysis['strength_score'] > 30 else '观望/等待回调'}
└ 建议仓位: {'<20%' if analysis['strength_score'] > 40 else '<10%' if analysis['strength_score'] > 20 else '不参与'}
⚠️ 风险提示
• 暴涨币种波动极大，需严格止损
• RSI > 70 时需警惕回调风险
• 成交量放大是双刃剑，既是动力也是风险
• 建议分批入场，控制仓位
🤖 交易建议
"""
        if analysis.get('rsi', 0) > 75:
            report += "• RSI严重超买，等待回调至RSI<65再考虑入场\n"
        elif analysis.get('increase_6h', 0) > 30:
            report += "• 短期涨幅过大，等待1-2小时回调再考虑\n"
        else:
            report += "• 可小仓位试探，严格设置5%止损\n"
        report += f"""
═══════════════════════════════
📅 报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        return report

# ============ 币种分类器 ============
class CoinClassifier:
    def __init__(self):
        self.classification_cache = {}
        self.cache_duration = 300

    def classify_coin(self, symbol: str, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        cache_key = f"{symbol}_{datetime.now().strftime('%H')}"
        if cache_key in self.classification_cache:
            cache_time, result = self.classification_cache[cache_key]
            if (datetime.now() - cache_time).total_seconds() < self.cache_duration:
                return result
        if '15m' not in data_dict:
            return {'symbol': symbol, 'category': 'UNKNOWN', 'confidence': 0}
        data_15m = data_dict['15m']
        rsi = TechnicalIndicatorsMultiTF.calculate_rsi(data_15m, 14)
        current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        ma20 = TechnicalIndicatorsMultiTF.calculate_ma(data_15m, 20)
        ma50 = TechnicalIndicatorsMultiTF.calculate_ma(data_15m, 50)
        current_price = data_15m['close'].iloc[-1]
        price_1h_ago = data_15m['close'].iloc[-4] if len(data_15m) >= 4 else current_price
        price_6h_ago = data_15m['close'].iloc[-24] if len(data_15m) >= 24 else current_price
        is_trend_coin = self._check_trend_coin(data_15m, current_price, ma20, ma50, current_rsi)
        is_bounce_coin = self._check_bounce_coin(data_15m, current_price, current_rsi, price_6h_ago)
        is_bounce_fail_coin = self._check_bounce_fail_coin(data_15m, current_price, current_rsi, price_1h_ago)
        categories = []
        confidences = []
        if is_trend_coin['is_coin']:
            categories.append('TREND')
            confidences.append(is_trend_coin['confidence'])
        if is_bounce_coin['is_coin']:
            categories.append('BOUNCE')
            confidences.append(is_bounce_coin['confidence'])
        if is_bounce_fail_coin['is_coin']:
            categories.append('BOUNCE_FAIL')
            confidences.append(is_bounce_fail_coin['confidence'])
        if not categories:
            final_category = 'RANGING'
            final_confidence = 0.5
        else:
            max_confidence_idx = confidences.index(max(confidences))
            final_category = categories[max_confidence_idx]
            final_confidence = confidences[max_confidence_idx]
        momentum = self._calculate_momentum(data_15m)
        result = {
            'symbol': symbol, 'category': final_category, 'confidence': round(final_confidence, 2),
            'rsi': round(float(current_rsi), 1), 'price': current_price, 'momentum': momentum,
            'price_change_1h': round((current_price - price_1h_ago) / price_1h_ago * 100, 2),
            'price_change_6h': round((current_price - price_6h_ago) / price_6h_ago * 100, 2)
        }
        self.classification_cache[cache_key] = (datetime.now(), result)
        return result

    def _check_trend_coin(self, data, current_price, ma20, ma50, rsi):
        result = {'is_coin': False, 'confidence': 0, 'trend_type': None, 'strength': 0}
        if len(data) < 50:
            return result
        if current_price > ma20.iloc[-1] > ma50.iloc[-1]:
            result['trend_type'] = 'UPTREND'
            result['strength'] += 30
        elif current_price < ma20.iloc[-1] < ma50.iloc[-1]:
            result['trend_type'] = 'DOWNTREND'
            result['strength'] += 30
        if rsi > 55 and result['trend_type'] == 'UPTREND':
            result['strength'] += 20
        elif rsi < 45 and result['trend_type'] == 'DOWNTREND':
            result['strength'] += 20
        bb_upper, bb_middle, bb_lower = TechnicalIndicatorsMultiTF.calculate_bollinger_bands(data)
        if result['trend_type'] == 'UPTREND' and current_price > bb_middle.iloc[-1]:
            result['strength'] += 15
        elif result['trend_type'] == 'DOWNTREND' and current_price < bb_middle.iloc[-1]:
            result['strength'] += 15
        result['confidence'] = min(100, result['strength']) / 100
        result['is_coin'] = result['confidence'] > 0.6
        return result

    def _check_bounce_coin(self, data, current_price, rsi, price_6h_ago):
        result = {'is_coin': False, 'confidence': 0, 'bounce_strength': 0, 'from_level': None}
        if len(data) < 24:
            return result
        recent_low = data['low'].iloc[-24:].min()
        recent_high = data['high'].iloc[-24:].max()
        bounce_from_low = (current_price - recent_low) / recent_low * 100
        decline_from_high = (recent_high - recent_low) / recent_high * 100
        if bounce_from_low > 5 and decline_from_high > 10:
            result['bounce_strength'] += 30
            result['from_level'] = 'LOW'
        if rsi < 40:
            result['bounce_strength'] += 20
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]
        if current_volume > avg_volume * 1.2:
            result['bounce_strength'] += 15
        price_change_6h = (current_price - price_6h_ago) / price_6h_ago * 100
        if price_change_6h > 3:
            result['bounce_strength'] += 15
        result['confidence'] = min(100, result['bounce_strength']) / 100
        result['is_coin'] = result['confidence'] > 0.5
        return result

    def _check_bounce_fail_coin(self, data, current_price, rsi, price_1h_ago):
        result = {'is_coin': False, 'confidence': 0, 'failure_signals': []}
        if len(data) < 30:
            return result
        recent_low_idx = data['low'].iloc[-30:].idxmin()
        recent_low = data['low'].loc[recent_low_idx]
        bounce_high = data['high'].iloc[-30:].max()
        bounce_pct = (bounce_high - recent_low) / recent_low * 100
        if bounce_pct < 3:
            result['failure_signals'].append('反弹幅度小')
            result['confidence'] += 0.2
        if abs(current_price - recent_low) / recent_low * 100 < 2:
            result['failure_signals'].append('接近前低')
            result['confidence'] += 0.2
        rsi_1h_ago = TechnicalIndicatorsMultiTF.calculate_rsi(
            data.iloc[:-4] if len(data) > 4 else data, 14
        ).iloc[-1] if len(data) > 4 else rsi
        if rsi < rsi_1h_ago:
            result['failure_signals'].append('RSI再次走弱')
            result['confidence'] += 0.15
        price_change_1h = (current_price - price_1h_ago) / price_1h_ago * 100
        if price_change_1h < -1:
            result['failure_signals'].append('1小时下跌')
            result['confidence'] += 0.15
        current_volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(window=10).mean().iloc[-1]
        if current_volume < avg_volume * 0.8:
            result['failure_signals'].append('成交量萎缩')
            result['confidence'] += 0.1
        result['is_coin'] = len(result['failure_signals']) >= 2 and result['confidence'] > 0.3
        return result

    def _calculate_momentum(self, data):
        if len(data) < 10:
            return 'NEUTRAL'
        price_5_periods_ago = data['close'].iloc[-5]
        current_price = data['close'].iloc[-1]
        change = (current_price - price_5_periods_ago) / price_5_periods_ago * 100
        if change > 2:
            return 'STRONG_UP'
        elif change > 0.5:
            return 'UP'
        elif change < -2:
            return 'STRONG_DOWN'
        elif change < -0.5:
            return 'DOWN'
        else:
            return 'NEUTRAL'

    def classify_all_coins(self, coins_data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict]:
        print(f"\n🤖 开始自动分类 {len(coins_data)} 个币种...")
        classifications = {}
        for symbol, data_dict in coins_data.items():
            try:
                classification = self.classify_coin(symbol, data_dict)
                classifications[symbol] = classification
            except Exception as e:
                print(f"⚠️ {symbol} 分类失败: {e}")
        category_counts = defaultdict(int)
        for class_result in classifications.values():
            category_counts[class_result['category']] += 1
        print(f"\n📊 分类统计结果:")
        for category, count in sorted(category_counts.items()):
            print(f"  {category}: {count}个币种")
        return classifications

# ============ 主系统类 ============
class UltimateTradingSystem:
    def __init__(self, telegram_bot_token=None, telegram_chat_id=None):
        print("🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
        print("🚀 终极智能交易系统 v33.6 完整正式版")
        print("🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀🚀")
        self.config = UltimateConfig
        self.coins = MONITOR_COINS[:self.config.COINS_TO_MONITOR]
        self.current_mode = 'BOUNCE'
        self.analysis_cycle_count = 0
        self.no_signal_count = 0
        self.start_time = datetime.now()
        self.cooldown_manager = CooldownManager()
        print("✅ 冷却管理器初始化成功")
        self.data_fetcher = OKXDataFetcher()
        print("✅ OKX数据获取器初始化成功")
        self.signal_checkers = {}
        self.init_signal_checkers()
        self.hype_analyzer = HypeAnalyzer(self.data_fetcher)
        self.coin_classifier = CoinClassifier()
        print("✅ 新增模块初始化完成")
        if telegram_bot_token and telegram_chat_id:
            try:
                self.telegram = UltimateTelegramNotifier(telegram_bot_token, telegram_chat_id)
                print("✅ Telegram通知器初始化成功")
            except Exception as e:
                print(f"⚠️ Telegram通知器初始化失败: {e}")
                self.telegram = None
        else:
            self.telegram = None
        self.stats = {
            'total_signals': 0, 'buy_signals': 0, 'sell_signals': 0,
            'signals_today': defaultdict(int), 'successful_trades': 0, 'failed_trades': 0
        }
        print("\n🤖 系统特性:")
        print(f"   • 监控币种: {len(self.coins)}个")
        print(f"   • 分析间隔: {self.config.ANALYSIS_INTERVAL}分钟")
        print(f"   • 多周期分析: {', '.join(self.config.MULTI_TIMEFRAME_CONFIG['timeframes'])}")
        print(f"   • 冷却机制: 同币种{self.config.COOLDOWN_CONFIG['same_coin_cooldown']}分钟")
        print(f"   • 交易模式: 9种（含新增确认K策略）")
        print(f"   • 实时Telegram通知: {'已启用' if self.telegram else '已禁用'}")
        print("\n✨ 新增对称策略:")
        print("   1. 反弹失败·确认K做空（防止追尾）")
        print("   2. 回调企稳·确认K做多（防止追尾）")
        print("\n✅ 系统初始化完成")
        print("=" * 70)

    def init_signal_checkers(self):
        print("\n🔄 初始化信号检查器...")
        signal_checkers_config = {
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
        for mode, checker_class in signal_checkers_config.items():
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
        coins_data = self.data_fetcher.get_all_coins_data(self.coins)
        return coins_data

    def run_multi_mode_analysis(self, coins_data):
        if not coins_data:
            return []
        print(f"🔄 运行多模式分析 ({len(coins_data)}个币种)...")
        all_signals = []
        for mode, checker in self.signal_checkers.items():
            if not self.config.MARKET_MODES.get(mode, {}).get('enabled', False):
                continue
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
                print(f"📊 {mode}模式未发现符合条件的信号")
        unique_signals = {}
        for signal in all_signals:
            symbol = signal['symbol']
            if symbol not in unique_signals or signal['score'] > unique_signals[symbol]['score']:
                unique_signals[symbol] = signal
        final_signals = list(unique_signals.values())
        final_signals.sort(key=lambda x: x.get('score', 0), reverse=True)
        return final_signals[:self.config.MAX_SIGNALS]

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
                    print("📊 无信号通知已发送到Telegram")
                except Exception as e:
                    print(f"❌ 发送无信号通知失败: {e}")
            return

        print(f"\n🔔 发现 {len(signals)} 个交易信号")
        sorted_signals = sorted(signals, key=lambda x: x.get('score', 0), reverse=True)

        if self.telegram:
            self.telegram.send_top_3_signals(sorted_signals)

        for signal in sorted_signals:
            cooldown_ok, cooldown_reason = self.cooldown_manager.check_cooldown(
                signal.get('symbol', 'UNKNOWN'),
                signal.get('direction', 'BUY')
            )
            if cooldown_ok:
                self.cooldown_manager.record_signal(
                    signal.get('symbol', 'UNKNOWN'),
                    signal.get('direction', 'BUY'),
                    signal.get('pattern', 'UNKNOWN'),
                    signal.get('score', 0)
                )
                self.stats['total_signals'] += 1
                if signal.get('direction') == 'BUY':
                    self.stats['buy_signals'] += 1
                else:
                    self.stats['sell_signals'] += 1
                today_str = datetime.now().strftime('%Y-%m-%d')
                self.stats['signals_today'][today_str] += 1

        print(f"\n✅ 已处理 {len(sorted_signals)} 个信号")

    def run_analysis_cycle_enhanced(self):
        try:
            self.analysis_cycle_count += 1
            print("\n" + "=" * 70)
            print(f"🤖 第 {self.analysis_cycle_count} 次智能分析周期 (v33.6正式版)")
            print("=" * 70)
            print("\n📊 获取市场数据...")
            coins_data = self.get_coins_data()
            if not coins_data or len(coins_data) < 10:
                print("❌ 数据获取失败或数据不足，跳过本次分析")
                return []
            cooldown_status = self.cooldown_manager.get_cooldown_status()
            print("\n🔍 运行全面市场诊断...")
            if 'hype' in [c.lower() for c in coins_data.keys()]:
                print("\n🔍 分析HYPE暴涨原因...")
                hype_report = self.hype_analyzer.generate_hype_report()
                if hype_report:
                    print(hype_report)
            print("\n🎯 运行智能币种分类...")
            classifications = self.coin_classifier.classify_all_coins(coins_data)
            print("\n🔍 分析最佳机会...")
            all_signals = []
            for mode, checker in self.signal_checkers.items():
                try:
                    for symbol, data_dict in coins_data.items():
                        direction = 'SELL' if mode in ['BREAKOUT_FAIL_SHORT', 'BOUNCE_FAIL_SHORT',
                                                      'TREND_EXHAUSTION', 'BOUNCE_FAIL_CONFIRM_K'] else 'BUY'
                        cooldown_ok, _ = self.cooldown_manager.check_cooldown(symbol, direction)
                        if not cooldown_ok:
                            continue
                        signal = checker.check_coin_multi_tf(symbol, data_dict)
                        if signal:
                            all_signals.append(signal)
                except Exception as e:
                    print(f"❌ {mode}模式收集信号失败: {e}")
            all_signals.sort(key=lambda x: x.get('score', 0), reverse=True)
            best_coins = all_signals[:10]
            if cooldown_status:
                print("\n🧊 冷却状态:")
                for symbol, status in list(cooldown_status.items())[:5]:
                    print(f"  {symbol}: {status['direction']} ({status['pattern']}) - 剩余{status['remaining']:.0f}分钟")
            else:
                print("\n🧊 冷却状态: 无冷却中的币种")
            if best_coins:
                print(f"\n🏆 最佳机会币种 (TOP {min(5, len(best_coins))}):")
                pattern_emojis = {
                    'BOUNCE': '🔺', 'BREAKOUT': '⚡', 'BREAKOUT_FAIL_SHORT': '🔻',
                    'TREND': '📈', 'CALLBACK': '🔄', 'BOUNCE_FAIL_SHORT': '⚡🔴',
                    'TREND_EXHAUSTION': '📉⚡', 'BOUNCE_FAIL_CONFIRM_K': '🎯🔻',
                    'CALLBACK_CONFIRM_K': '🎯🟢'
                }
                for i, coin in enumerate(best_coins[:5], 1):
                    pattern_emoji = pattern_emojis.get(coin['pattern'], '💰')
                    direction_emoji = '🟢' if coin.get('direction') == 'BUY' else '🔴'
                    bounce_info = f" 反弹:{coin.get('bounce_pct', '')}%" if 'bounce_pct' in coin else ""
                    callback_info = f" 回调:{coin.get('callback_pct', '')}%" if 'callback_pct' in coin else ""
                    confirm_k_info = f" 确认K" if 'confirmation_k_info' in coin else ""
                    print(f"  {i}. {pattern_emoji} {coin['symbol']:6s} | "
                          f"RSI: {coin.get('rsi', 0):5.1f} | "
                          f"评分: {coin.get('score', 0):3d} | "
                          f"方向: {direction_emoji} | "
                          f"模式: {coin['pattern']}{bounce_info}{callback_info}{confirm_k_info}")
            else:
                print("\n📊 未发现符合条件的交易机会")
            print("\n🔍 运行多模式智能分析...")
            signals = self.run_multi_mode_analysis(coins_data)
            if signals:
                self.process_signals(signals)
                self.no_signal_count = 0
            else:
                print("\n📊 本次分析未发现符合条件的交易信号")
                self.no_signal_count += 1
            next_time = datetime.now() + timedelta(minutes=self.config.ANALYSIS_INTERVAL)
            print(f"\n⏳ 下次分析: {self.config.ANALYSIS_INTERVAL}分钟后 ({next_time.strftime('%H:%M:%S')})")
            print(f"🎯 当前模式: {self.current_mode}")
            print(f"📊 无信号计数: {self.no_signal_count}")
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
        print("🎯 立即运行首次分析...")
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

# ============ 主程序（优化版，支持 GitHub Actions 一次性运行）============
def main():
    print("=" * 70)
    print("🚀 终极智能交易系统 v33.6 完整正式版（GitHub Actions 优化版）")
    print("=" * 70)
    print("📅 版本: 33.6-正式版-GitHub优化")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 初始监控币种: {len(MONITOR_COINS)}个")
    print("🎯 智能模式: 9种（新增2个确认K策略）")
    print(f"📈 多周期分析: {', '.join(UltimateConfig.MULTI_TIMEFRAME_CONFIG['timeframes'])}")
    print(f"⏰ 分析间隔: {UltimateConfig.ANALYSIS_INTERVAL}分钟")
    print(f"🧊 冷却机制: 同币种{UltimateConfig.COOLDOWN_CONFIG['same_coin_cooldown']}分钟")
    print(f"📈 数据源: OKX公共API")
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        print(f"🤖 Telegram通知: ✅ 已启用（从环境变量读取）")
    else:
        print(f"🤖 Telegram通知: ⚠️ 未配置，已禁用（如需启用请在 Secrets 中设置 TELEGRAM_BOT_TOKEN 和 TELEGRAM_CHAT_ID）")
    print("=" * 70)

    # 创建系统实例
    system = UltimateTradingSystem(
        telegram_bot_token=TELEGRAM_BOT_TOKEN,
        telegram_chat_id=TELEGRAM_CHAT_ID
    )

    if not system:
        print("❌ 系统初始化失败，请检查错误日志")
        return

    # 发送额外的启动消息（如果 Telegram 可用）
    if system.telegram:
        try:
            extra_startup_msg = f"""
🔔 <b>系统配置详情</b>
━━━━━━━━━━━━━━━━━━━━
📅 系统版本: {UltimateConfig.VERSION}
📊 监控列表: {', '.join(MONITOR_COINS[:10])}等共{len(MONITOR_COINS)}个币种
🎯 新增策略:
├ 反弹失败·确认K做空
└ 回调企稳·确认K做多
━━━━━━━━━━━━━━━━━━━━
💡 <i>您将收到前3个最佳信号的详细分析</i>
"""
            system.telegram.bot.send_message(
                system.telegram.chat_id,
                extra_startup_msg,
                parse_mode='HTML'
            )
        except Exception as e:
            print(f"⚠️ 发送额外启动消息失败: {e}")

    # 检测是否在 GitHub Actions 环境中
    if os.getenv('GITHUB_ACTIONS') == 'true':
        print("\n🔧 检测到 GitHub Actions 环境，将以一次性模式运行单次分析")
        signals = system.run_single_cycle()
        print(f"\n✅ 本次分析完成，共发现 {len(signals) if signals else 0} 个信号。")
        # 可选：发送运行状态消息（如果 Telegram 可用）
        if system.telegram and signals:
            status_msg = f"""
📈 <b>GitHub Actions 定时分析完成</b>
━━━━━━━━━━━━━━━━━━━━
📊 发现 {len(signals)} 个交易信号
⏰ 分析时间: {datetime.now().strftime('%H:%M:%S')}
🔄 下次分析将由 GitHub Actions 定时触发
━━━━━━━━━━━━━━━━━━━━
🤖 <i>系统已进入自动监控模式</i>
"""
            try:
                system.telegram.bot.send_message(
                    system.telegram.chat_id,
                    status_msg,
                    parse_mode='HTML'
                )
            except Exception as e:
                print(f"⚠️ 发送状态消息失败: {e}")
        print("\n🏁 GitHub Actions 任务结束，退出。")
        return  # 退出程序，不进入连续循环

    # 非 GitHub Actions 环境：进入连续监控模式
    print("\n🚀 检测到本地运行，启动连续监控模式...")
    system.run_continuous()

if __name__ == "__main__":
    main()