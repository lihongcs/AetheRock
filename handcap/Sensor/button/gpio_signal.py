#!/usr/bin/env python3
"""
Orange Pi 5 Pro GPIO 信号读取 - 1/0 输出
GPIO1_A6 -> Pin 31 (排针31号)

用法:
  python3 gpio_signal.py              # 简单模式: 输出 0/1
  python3 gpio_signal.py --continuous # 持续监控模式
  python3 gpio_signal.py --poll 0.1  # 自定义轮询间隔(秒)
"""

import os
import sys
import time
import argparse
from signal import signal, SIGINT

# ============== 配置 ==============
GPIO_NUM = 38  # GPIO1_A6 对应的 sysfs 编号
GPIO_PATH = f"/sys/class/gpio/gpio{GPIO_NUM}"

# ==========================================

def init_gpio():
    """初始化GPIO"""
    if not os.path.exists(GPIO_PATH):
        with open("/sys/class/gpio/export", "w") as f:
            f.write(str(GPIO_NUM))
        time.sleep(0.1)
    
    with open(f"{GPIO_PATH}/direction", "w") as f:
        f.write("in")
    print(f"GPIO {GPIO_NUM} 初始化完成", file=sys.stderr)


def read_gpio() -> int:
    """读取GPIO值，返回 0 或 1"""
    with open(f"{GPIO_PATH}/value", "r") as f:
        return int(f.read().strip())


def cleanup(signum, frame):
    """清理退出"""
    try:
        with open("/sys/class/gpio/unexport", "w") as f:
            f.write(str(GPIO_NUM))
    except:
        pass
    print("\n程序退出")
    sys.exit(0)


def mode_simple():
    """简单模式: 读取一次当前状态"""
    init_gpio()
    
    value = read_gpio()
    print(value)  # 直接输出 0 或 1
    print(value, end="")  # 方便脚本调用
    cleanup(None, None)


def mode_continuous(poll_interval: float = 0.05):
    """持续监控模式: 实时显示信号变化"""
    init_gpio()
    
    print("\n" + "=" * 40)
    print("GPIO 实时监控 (1/0 数字输出)")
    print("=" * 40)
    print("按 Ctrl+C 退出\n")
    
    signal(SIGINT, cleanup)
    
    last_value = read_gpio()
    print(f"初始状态: {last_value}")
    print("-" * 40)
    
    while True:
        value = read_gpio()
        
        if value != last_value:
            # 信号变化时输出
            print(f"[{time.strftime('%H:%M:%S')}] GPIO = {value}")
            last_value = value
        
        time.sleep(poll_interval)


def mode_loop(poll_interval: float = 0.1):
    """循环模式: 持续输出当前值"""
    init_gpio()
    
    print("\n持续读取 GPIO 信号...")
    print("按 Ctrl+C 退出\n")
    
    signal(SIGINT, cleanup)
    
    while True:
        value = read_gpio()
        print(f"\rGPIO: {value}", end="", flush=True)
        time.sleep(poll_interval)


def mode_event():
    """事件模式: 检测到变化时输出"""
    init_gpio()
    
    print("\n事件检测模式 - 等待信号变化...")
    print("-" * 40)
    
    signal(SIGINT, cleanup)
    
    fd = os.open(GPIO_PATH + "/value", os.O_RDONLY)
    last_value = int(os.read(fd, 1))
    
    print(f"初始状态: {last_value}\n")
    
    while True:
        os.lseek(fd, 0, os.SEEK_SET)
        value = int(os.read(fd, 1))
        
        if value != last_value:
            timestamp = time.strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] {last_value} -> {value}")
            last_value = value
        
        time.sleep(0.01)
    
    os.close(fd)


def mode_polling(count: int = 10, interval: float = 0.5):
    """轮询模式: 读取指定次数"""
    init_gpio()
    
    print(f"\n轮询读取 {count} 次 (间隔 {interval}s)")
    print("-" * 40)
    
    signal(SIGINT, cleanup)
    
    for i in range(count):
        value = read_gpio()
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] #{i+1:3d} -> {value}")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPIO 1/0 信号读取")
    parser.add_argument("-c", "--continuous", action="store_true",
                        help="持续监控模式")
    parser.add_argument("-e", "--event", action="store_true",
                        help="事件检测模式 (检测变化时输出)")
    parser.add_argument("-l", "--loop", action="store_true",
                        help="循环输出模式")
    parser.add_argument("-p", "--poll", type=float, default=0.1,
                        help="轮询间隔(秒), 默认0.1")
    parser.add_argument("-n", "--count", type=int, default=10,
                        help="轮询次数, 默认10")
    parser.add_argument("-i", "--interval", type=float, default=0.5,
                        help="轮询间隔(秒), 默认0.5")
    
    args = parser.parse_args()
    
    if args.continuous:
        mode_continuous(args.poll)
    elif args.event:
        mode_event()
    elif args.loop:
        mode_loop(args.poll)
    else:
        mode_polling(args.count, args.interval)
