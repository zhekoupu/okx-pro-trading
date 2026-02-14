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

# ... 此处省略了中间所有的类定义（CooldownManager, OKXDataFetcher, TechnicalIndicatorsMultiTF, BaseSignalChecker, KLineAnalyzer, UltimateTelegramNotifier, 各策略检查器, HypeAnalyzer, CoinClassifier, UltimateTradingSystem），它们与您提供的原代码完全一致，没有改动。由于篇幅限制，不再重复粘贴。您可以直接使用原代码中的这些类定义，只需确保将文件开头的环境变量读取和最后的 main 函数替换为下面的版本即可。

# ============ 主程序（优化版） ============
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

    # 创建系统实例（如果 Telegram 凭证缺失，内部会处理禁用）
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