name: Ultimate Trading Bot v36.12 (Production)

on:
  schedule:
    # 每 45 分钟运行一次，平衡实时性与额度消耗
    - cron: '*/45 * * * *'
  workflow_dispatch:

# 并发锁：防止同时运行多个实例，避免冷却文件冲突
concurrency:
  group: ultimate-bot-v36
  cancel-in-progress: true

env:
  PYTHON_VERSION: '3.10'

jobs:
  analyze:
    runs-on: ubuntu-latest
    timeout-minutes: 15   # 防止卡死，保护免费额度

    steps:
      # 1️⃣ 检出代码
      - name: Checkout repository
        uses: actions/checkout@v4

      # 2️⃣ 设置 Python 环境并启用依赖缓存（加速安装，节约时间）
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
          cache: 'pip'

      # 3️⃣ 恢复冷却状态缓存（使用动态主键，确保每次更新后保存）
      - name: Restore cooldown state
        uses: actions/cache@v4
        id: cache-cooldown
        with:
          path: cooldown_state.pkl   # 您的冷却文件路径，请确保脚本使用相对路径
          # 主键包含 run_id，保证每次运行唯一，从而强制在结束后保存新缓存
          key: cooldown-v36-${{ github.ref_name }}-${{ github.run_id }}
          # 回退键：当没有精确匹配时，尝试加载最近一次的历史缓存
          restore-keys: |
            cooldown-v36-${{ github.ref_name }}-
            cooldown-v36-

      # 4️⃣ 安装依赖（带重试，应对网络波动）
      - name: Install dependencies
        uses: nick-fields/retry@v3
        with:
          timeout_minutes: 5
          max_attempts: 3
          retry_wait_seconds: 20
          command: |
            python -m pip install --upgrade pip
            # 如果有 requirements.txt，强烈建议使用
            pip install -r requirements.txt
            # 如果没有，可以手动列出依赖，例如：
            # pip install pandas numpy requests pyTelegramBotAPI scipy

      # 5️⃣ 运行主程序（内部已实现多币种并发）
      - name: Run Ultimate Trading System v36.12
        id: run_bot
        uses: nick-fields/retry@v3
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
          PYTHONUNBUFFERED: 1   # 实时输出日志
        with:
          timeout_minutes: 10
          max_attempts: 2        # 失败后重试一次
          retry_wait_seconds: 60
          retry_on: error
          command: python main.py   # 请确保您的入口文件名为 main.py

      # 6️⃣ （可选）上传日志文件，便于复盘；保留1天以节约存储
      - name: Upload logs
        if: always() && hashFiles('*.log') != ''
        uses: actions/upload-artifact@v4
        with:
          name: logs-${{ github.run_id }}
          path: '*.log'
          retention-days: 1

      # 7️⃣ （可选）计算耗时并输出到日志
      - name: Show execution time
        if: always()
        run: |
          echo "✅ 本次运行完成于 $(date)"