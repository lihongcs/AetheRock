#!/usr/bin/env python3
"""
Orange Pi 5 Pro GPIO 实时信号捕获测试
GPIO1_A6 -> Pin 31 (排针31号)
"""

import os
import sys
import time
import argparse
from datetime import datetime

# GPIO1_A6 在 RK3588 上的sysfs编号
GPIO_NUM = 38


class SignalCapture:
    """实时信号捕获器"""
    
    def __init__(self, gpio_num: int = GPIO_NUM, debounce_ms: int = 50):
        self.gpio_num = gpio_num
        self.gpio_path = f"/sys/class/gpio/gpio{gpio_num}"
        self.debounce_ms = debounce_ms
        self.last_state = None
        self.last_change_time = 0
        self.event_count = 0
        
        # 统计
        self.press_count = 0
        self.release_count = 0
        self.start_time = None
        
        self._setup()
    
    def _setup(self):
        """初始化GPIO"""
        if not os.path.exists(self.gpio_path):
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(self.gpio_num))
            time.sleep(0.1)
        
        with open(f"{self.gpio_path}/direction", "w") as f:
            f.write("in")
        
        # 获取初始状态
        self.last_state = self.read()
        self.last_change_time = time.time()
        self.start_time = time.time()
        
        print(f"GPIO {self.gpio_num} 初始化完成")
        print(f"初始状态: {self._state_str(self.last_state)}")
        print("-" * 50)
    
    def read(self) -> int:
        """读取GPIO值"""
        with open(f"{self.gpio_path}/value", "r") as f:
            return int(f.read().strip())
    
    def _state_str(self, state: int) -> str:
        return "按下" if state == 0 else "松开  "
    
    def _timestamp(self) -> str:
        """获取带时间戳的字符串"""
        return datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    def _elapsed(self) -> str:
        """获取经过的时间"""
        elapsed = time.time() - self.start_time
        return f"{elapsed:.1f}s"
    
    def capture_loop(self, poll_interval: float = 0.005, verbose: bool = True):
        """
        实时捕获循环
        
        Args:
            poll_interval: 轮询间隔(秒)，默认5ms
            verbose: 是否打印详细信息
        """
        print(f"开始实时捕获 (轮询间隔: {poll_interval*1000:.1f}ms, 防抖: {self.debounce_ms}ms)")
        print("按 Ctrl+C 停止\n")
        
        if verbose:
            print(f"{'时间戳':<15} {'电平':<6} {'状态':<8} {'事件':<12} {'间隔':<10} {'总计数'}")
            print("-" * 70)
        
        try:
            while True:
                current_state = self.read()
                current_time = time.time()
                
                # 防抖检查
                time_diff = (current_time - self.last_change_time) * 1000
                
                if current_state != self.last_state:
                    if time_diff >= self.debounce_ms:
                        # 状态确实改变
                        self.event_count += 1
                        
                        if current_state == 0:
                            self.press_count += 1
                            event = "下降沿(按下)"
                        else:
                            self.release_count += 1
                            event = "上升沿(释放)"
                        
                        interval_ms = time_diff
                        
                        if verbose:
                            print(f"{self._timestamp():<15} {current_state:<6} {self._state_str(current_state):<8} "
                                  f"{event:<12} {interval_ms:>6.1f}ms  #{self.event_count}")
                        
                        self.last_state = current_state
                        self.last_change_time = current_time
                
                time.sleep(poll_interval)
                
        except KeyboardInterrupt:
            print("\n" + "-" * 50)
            self.print_stats()
    
    def print_stats(self):
        """打印统计信息"""
        total_time = time.time() - self.start_time
        print("\n=== 捕获统计 ===")
        print(f"总运行时间: {total_time:.2f} 秒")
        print(f"总事件数:   {self.event_count}")
        print(f"按下次数:   {self.press_count}")
        print(f"释放次数:   {self.release_count}")
        if total_time > 0:
            print(f"事件频率:   {self.event_count/total_time:.2f} 次/秒")
    
    def cleanup(self):
        """清理GPIO"""
        try:
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(self.gpio_num))
        except:
            pass


def visual_monitor():
    """可视化实时监控（屏幕刷新模式）"""
    print("=" * 60)
    print("Orange Pi 5 Pro - GPIO 实时监控")
    print(f"GPIO: GPIO1_A6 (sysfs #{GPIO_NUM}, 排针31号)")
    print("=" * 60)
    
    capture = SignalCapture(gpio_num=GPIO_NUM)
    
    try:
        # 使用更快的轮询
        capture.capture_loop(poll_interval=0.01, verbose=True)
    except KeyboardInterrupt:
        pass
    finally:
        capture.cleanup()


def waveform_mode():
    """波形模式 - 显示信号变化"""
    print("\n" + "=" * 60)
    print("波形模式 - 显示最近100次信号变化")
    print("=" * 60)
    
    capture = SignalCapture(gpio_num=GPIO_NUM, debounce_ms=30)
    
    # 使用GPIO字符设备获取更精确的边沿触发
    try:
        fd = os.open(f"/sys/class/gpio/gpio{GPIO_NUM}/value", os.O_RDONLY)
        
        print(f"\n{'#':<4} {'时间':<15} {'状态':<10} {'波形'}")
        print("-" * 50)
        
        event_num = 0
        waveform = []
        
        while event_num < 100:
            # 读取并等待变化
            os.lseek(fd, 0, os.SEEK_SET)
            value = int(os.read(fd, 1))
            
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            
            # 更新波形显示
            waveform.append("█" if value == 0 else "░")
            if len(waveform) > 40:
                waveform.pop(0)
            
            state = "按下" if value == 0 else "松开"
            waveform_str = "".join(waveform[-40:])
            
            event_num += 1
            print(f"{event_num:<4} {timestamp:<15} {state:<10} {waveform_str}")
            
            time.sleep(0.1)
        
        os.close(fd)
        
    except KeyboardInterrupt:
        print("\n波形捕获中断")
    finally:
        capture.cleanup()


def simple_test(duration: int = 10):
    """简单测试模式"""
    print(f"\n简单测试模式 - 运行 {duration} 秒")
    print("-" * 50)
    
    capture = SignalCapture(gpio_num=GPIO_NUM)
    
    try:
        capture.capture_loop(poll_interval=0.01, verbose=True)
    except KeyboardInterrupt:
        pass
    finally:
        capture.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Orange Pi 5 Pro GPIO 信号捕获")
    parser.add_argument("-m", "--mode", choices=["monitor", "waveform", "simple"],
                        default="monitor", help="运行模式")
    parser.add_argument("-t", "--time", type=int, default=10,
                        help="简单测试模式运行时间(秒)")
    parser.add_argument("-d", "--debounce", type=int, default=50,
                        help="防抖时间(毫秒)")
    
    args = parser.parse_args()
    
    if args.mode == "monitor":
        visual_monitor()
    elif args.mode == "waveform":
        waveform_mode()
    elif args.mode == "simple":
        simple_test(args.time)
