#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终极智能交易系统 v33.9 GitHub Actions适配版
适用于定时运行，单次分析后退出
"""

import os
import sys
import time
import pickle
import atexit
import requests
import traceback
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Tuple

# ============ 导入库（需提前安装）============
import pandas as pd
import numpy as np
import telebot

print("🔧 检查TA-Lib依赖...")
try:
    import talib
    TALIB_AVAILABLE = True
    print("✅ TA-Lib已安装，启用高级技术指标")
except ImportError:
    TALIB_AVAILABLE = False
    print("⚠️ TA-Lib未安装，将使用备用技术指标")

# ============ 配置 ============
# 从环境变量读取Telegram配置（GitHub Secrets传入）
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")

if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
    print("❌ 错误：未设置TELEGRAM_BOT_TOKEN或TELEGRAM_CHAT_ID环境变量")
    sys.exit(1)

print(f"🤖 Telegram配置已加载（令牌: {TELEGRAM_BOT_TOKEN[:10]}...）")

# OKX API配置
OKX_API_BASE_URL = "https://www.okx.com"
OKX_CANDLE_INTERVAL = ["15m", "1H"]
OKX_CANDLE_LIMIT = 100

# 监控币种列表（保持完整）
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

print(f"📊 监控币种列表: {len(MONITOR_COINS)}个币种")

# ============ 配置类 ============
class UltimateConfig:
    VERSION = "33.9-GitHubActions版"
    ANALYSIS_INTERVAL = 45  # 保留，但单次运行不使用
    COINS_TO_MONITOR = len(MONITOR_COINS)
    MAX_SIGNALS = 8

    COOLDOWN_CONFIG = {
        'same_coin_cooldown': 90,
        'same_direction_cooldown': 45,
        'max_signals_per_coin_per_day': 5,
        'enable_cooldown': True
    }

    SIGNAL_THRESHOLDS = {
        'BOUNCE': 25,
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
        'rate_limit': 20,
        'retry_times': 2,
        'timeout': 15
    }

    TELEGRAM_CONFIG = {
        'enabled': True,
        'parse_mode': 'HTML',
        'always_send_signals': True,
        'send_market_reports': False,
        'send_classification_reports': False
    }

# ============ 冷却管理器（保留文件持久化）============
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
        self.cache_duration = 120

    def get_candles(self, symbol: str, interval: str):
        cache_key = f"{symbol}_{interval}"
        current_time = time.time()
        if cache_key in self.cache and current_time - self.cache_time.get(cache_key, 0) < self.cache_duration:
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
    def calculate_volume_ratio(data: pd.DataFrame, period: int = 20):
        if len(data) < period:
            return pd.Series([1.0] * len(data), index=data.index)
        current_volume = data['volume']
        avg_volume = data['volume'].rolling(window=period).mean()
        volume_ratio = current_volume / avg_volume
        return volume_ratio.fillna(1.0)

# ============ 信号检查器 ============
class SignalChecker:
    def __init__(self):
        self.thresholds = UltimateConfig.SIGNAL_THRESHOLDS
        self.params = UltimateConfig.OPTIMIZATION_PARAMS

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

                signals = []

                # 反弹信号
                if rsi < self.params['rsi_bounce_max'] and volume_ratio > self.params['volume_ratio_min']:
                    score = self._calculate_bounce_score(rsi, volume_ratio)
                    if score >= self.thresholds['BOUNCE']:
                        signal = self._create_bounce_signal(symbol, data_15m, current_price, rsi, volume_ratio, ma20, score)
                        signals.append(signal)
                        signal_counts['BOUNCE'] += 1

                # 回调信号
                if rsi > self.params['rsi_callback_min']:
                    recent_high = data_15m['high'].iloc[-30:].max()
                    callback_pct = ((recent_high - current_price) / recent_high) * 100
                    if self.params['callback_pct_min'] <= callback_pct <= self.params['callback_pct_max']:
                        score = self._calculate_callback_score(rsi, volume_ratio, callback_pct)
                        if score >= self.thresholds['CALLBACK']:
                            signal = self._create_callback_signal(symbol, data_15m, current_price, rsi, volume_ratio, recent_high, callback_pct, ma20, score)
                            signals.append(signal)
                            signal_counts['CALLBACK'] += 1

                # 回调确认转强信号
                if 48 <= rsi <= 72 and volume_ratio > 1.2:
                    recent_high = data_15m['high'].iloc[-30:].max()
                    callback_pct = ((recent_high - current_price) / recent_high) * 100
                    if 2 <= callback_pct <= 15:
                        recent_3_closes = data_15m['close'].iloc[-3:].values
                        price_increasing = len(recent_3_closes) >= 2 and recent_3_closes[-1] > recent_3_closes[0]
                        if price_increasing and ma20 > ma50 and current_price > ma20:
                            score = self._calculate_callback_confirm_score(rsi, volume_ratio, callback_pct)
                            if score >= self.thresholds['CALLBACK_CONFIRM_K']:
                                signal = self._create_callback_confirm_signal(symbol, data_15m, current_price, rsi, volume_ratio, recent_high, callback_pct, ma20, ma50, score)
                                signals.append(signal)
                                signal_counts['CALLBACK_CONFIRM_K'] += 1

                # 趋势衰竭做空信号
                if rsi > self.params['trend_exhaustion_rsi_min'] and volume_ratio < 1.0:
                    score = self._calculate_trend_exhaustion_score(rsi, volume_ratio)
                    if score >= self.thresholds['TREND_EXHAUSTION']:
                        signal = self._create_trend_exhaustion_signal(symbol, data_15m, current_price, rsi, volume_ratio, ma20, score)
                        signals.append(signal)
                        signal_counts['TREND_EXHAUSTION'] += 1

                if signals:
                    best_signal = max(signals, key=lambda x: x.get('score', 0))
                    all_signals.append(best_signal)

            except Exception as e:
                continue

        self._print_statistics(signal_counts, len(coins_data))
        print(f"✅ 扫描完成: 发现 {len(all_signals)} 个交易信号")
        return all_signals

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

# ============ Telegram通知器 ============
class TelegramNotifier:
    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.bot = None
        try:
            self.bot = telebot.TeleBot(bot_token, parse_mode='HTML')
            bot_info = self.bot.get_me()
            print(f"✅ Telegram连接成功: @{bot_info.username}")
        except Exception as e:
            print(f"❌ Telegram连接失败: {e}")
            self.bot = None

    def send_signal(self, signal, cooldown_reason=""):
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

# ============ 交易系统主类（单次运行）============
class UltimateTradingSystem:
    def __init__(self):
        print("\n" + "="*60)
        print("🚀 终极智能交易系统 v33.9 - GitHub Actions版")
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
        print(f"🤖 Telegram通知: {'✅ 已启用' if self.telegram.bot else '⚠️ 已禁用'}")
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
        max_signals_to_send = min(5, len(signals))
        top_signals = signals[:max_signals_to_send]

        sent_count = 0
        for i, signal in enumerate(top_signals, 1):
            symbol = signal.get('symbol', 'UNKNOWN')
            pattern = signal.get('pattern', 'UNKNOWN')
            score = signal.get('score', 0)
            print(f"\n[{i}] {symbol}: {pattern} ({score}分)")

            cooldown_ok, cooldown_reason = self.cooldown_manager.check_cooldown(
                symbol, signal.get('direction', 'BUY')
            )
            if not cooldown_ok:
                print(f"   ⚠️ 冷却阻止: {cooldown_reason}")
                continue

            if self.telegram and self.telegram.bot:
                success = self.telegram.send_signal(signal, cooldown_reason)
                if success:
                    self.cooldown_manager.record_signal(
                        symbol,
                        signal.get('direction', 'BUY'),
                        pattern,
                        score
                    )
                    self.total_signals += 1
                    sent_count += 1
                    time.sleep(2)
                else:
                    print(f"   ⚠️ 信号发送失败，跳过")
            else:
                print(f"   ⚠️ Telegram未启用，跳过发送")

        print(f"\n✅ 本次成功发送 {sent_count} 个交易信号")

# ============ 主程序入口 ============
def main():
    print("="*60)
    print("🤖 终极智能交易系统 v33.9 - GitHub Actions版")
    print("="*60)
    print(f"📅 版本: {UltimateConfig.VERSION}")
    print(f"⏰ 启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 监控币种: {len(MONITOR_COINS)}个")
    print(f"🎯 信号模式: 4种优化策略（包含CALLBACK_CONFIRM_K）")
    print("="*60)

    try:
        system = UltimateTradingSystem()
        print("\n🎯 运行实时分析...")
        signals = system.run_analysis()

        if signals:
            print(f"\n✅ 分析完成！发现 {len(signals)} 个交易信号")
        else:
            print("\n📊 本次分析未发现信号")

        # 在GitHub Actions环境下，直接退出（无需等待）
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