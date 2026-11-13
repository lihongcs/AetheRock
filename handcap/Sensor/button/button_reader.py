#!/usr/bin/env python3
"""
Orange Pi 5 Pro GPIO Button Reader
GPIO1_A6 -> Pin 31 (排针31号)
"""

import os
import time
import threading
from typing import Callable, Optional

# GPIO1_A6 在 RK3588 上的sysfs编号
GPIO_NUM = 364
GPIO_PATH = f"/sys/class/gpio/gpio{GPIO_NUM}"

class ButtonReader:
    """按钮读取类，支持轮询和事件回调"""
    
    def __init__(self, gpio_num: int = GPIO_NUM, pull_up: bool = True):
        """
        初始化按钮读取器
        
        Args:
            gpio_num: GPIO编号 (默认38 = GPIO1_A6)
            pull_up: 按钮连接方式，True=上拉(按钮接GND), False=下拉(按钮接VCC)
        """
        self.gpio_num = gpio_num
        self.gpio_path = f"/sys/class/gpio/gpio{gpio_num}"
        self.pull_up = pull_up
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 回调函数
        self.on_press: Optional[Callable[[], None]] = None
        self.on_release: Optional[Callable[[], None]] = None
        
        self._setup_gpio()
    
    def _setup_gpio(self):
        """配置GPIO为输入模式"""
        # 导出GPIO
        if not os.path.exists(self.gpio_path):
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(self.gpio_num))
            time.sleep(0.1)
        
        # 配置为输入
        with open(f"{self.gpio_path}/direction", "w") as f:
            f.write("in")
        
        # 配置边缘检测 (both edges)
        with open(f"{self.gpio_path}/edge", "w") as f:
            f.write("both")
        
        print(f"GPIO {self.gpio_num} configured as input")
    
    def read_state(self) -> int:
        """
        读取当前按钮状态
        
        Returns:
            1: 按钮未按下 (高电平)
            0: 按钮按下 (低电平)
        """
        with open(f"{self.gpio_path}/value", "r") as f:
            return int(f.read().strip())
    
    def is_pressed(self) -> bool:
        """
        判断按钮是否被按下
        
        Returns:
            True: 按钮按下
            False: 按钮未按下
        """
        state = self.read_state()
        if self.pull_up:
            return state == 0  # 上拉模式: 按下=低电平
        else:
            return state == 1  # 下拉模式: 按下=高电平
    
    def wait_for_press(self, timeout: Optional[float] = None) -> bool:
        """
        等待按钮按下
        
        Args:
            timeout: 超时时间(秒)，None表示无限等待
            
        Returns:
            True: 按钮被按下
            False: 超时
        """
        fd = os.open(f"{self.gpio_path}/value", os.O_RDONLY)
        try:
            if timeout:
                start = time.time()
                while True:
                    # 读取当前状态
                    os.lseek(fd, 0, os.SEEK_SET)
                    state = int(os.read(fd, 1))
                    
                    if self.pull_up:
                        pressed = (state == 0)
                    else:
                        pressed = (state == 1)
                    
                    if pressed:
                        return True
                    
                    if time.time() - start >= timeout:
                        return False
                    
                    time.sleep(0.01)
            else:
                while True:
                    os.lseek(fd, 0, os.SEEK_SET)
                    state = int(os.read(fd, 1))
                    
                    if self.pull_up:
                        pressed = (state == 0)
                    else:
                        pressed = (state == 1)
                    
                    if pressed:
                        return True
                    time.sleep(0.01)
        finally:
            os.close(fd)
    
    def _monitor_loop(self, poll_interval: float = 0.01, debounce_ms: int = 50):
        """
        监控循环
        
        Args:
            poll_interval: 轮询间隔(秒)
            debounce_ms: 防抖时间(毫秒)
        """
        last_state = self.read_state()
        last_change_time = time.time()
        
        while self._running:
            current_state = self.read_state()
            current_time = time.time()
            
            # 检查是否需要防抖
            if current_state != last_state:
                if (current_time - last_change_time) * 1000 >= debounce_ms:
                    # 状态确实改变了
                    if self.pull_up:
                        pressed = (current_state == 0)
                    else:
                        pressed = (current_state == 1)
                    
                    if pressed and self.on_press:
                        self.on_press()
                    elif not pressed and self.on_release:
                        self.on_release()
                    
                    last_state = current_state
                    last_change_time = current_time
            else:
                last_change_time = current_time
            
            time.sleep(poll_interval)
    
    def start_monitoring(self, poll_interval: float = 0.01, debounce_ms: int = 50):
        """
        开始监控按钮事件
        
        Args:
            poll_interval: 轮询间隔(秒)
            debounce_ms: 防抖时间(毫秒)
        """
        if self._running:
            print("Already monitoring")
            return
        
        self._running = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(poll_interval, debounce_ms),
            daemon=True
        )
        self._monitor_thread.start()
        print("Button monitoring started")
    
    def stop_monitoring(self):
        """停止监控"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        print("Button monitoring stopped")
    
    def cleanup(self):
        """清理GPIO资源"""
        self.stop_monitoring()
        try:
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(self.gpio_num))
            print(f"GPIO {self.gpio_num} unexported")
        except Exception as e:
            print(f"Cleanup error: {e}")


def demo_polling():
    """轮询方式读取按钮（简单演示）"""
    print("=" * 50)
    print("Demo 1: Polling Mode (轮询模式)")
    print("=" * 50)
    
    button = ButtonReader(gpio_num=GPIO_NUM, pull_up=True)
    
    try:
        print("按 Ctrl+C 退出")
        print()
        
        while True:
            state = button.read_state()
            pressed = button.is_pressed()
            
            status = "按下" if pressed else "松开"
            print(f"\rGPIO值: {state} | 状态: {status}    ", end="", flush=True)
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n退出程序")
    finally:
        button.cleanup()


def demo_wait():
    """等待按下方式读取按钮"""
    print("=" * 50)
    print("Demo 2: Wait for Press Mode (等待按下)")
    print("=" * 50)
    
    button = ButtonReader(gpio_num=GPIO_NUM, pull_up=True)
    
    try:
        print("等待按钮按下...")
        
        while True:
            if button.wait_for_press(timeout=5):
                print("按钮已按下!")
                print(f"当前GPIO值: {button.read_state()}")
            else:
                print("等待超时...")
                
    except KeyboardInterrupt:
        print("\n\n退出程序")
    finally:
        button.cleanup()


def demo_callback():
    """回调方式读取按钮（事件驱动）"""
    print("=" * 50)
    print("Demo 3: Callback Mode (回调模式)")
    print("=" * 50)
    
    button = ButtonReader(gpio_num=GPIO_NUM, pull_up=True)
    
    # 设置回调函数
    def on_press():
        print("\n>>> 按钮按下! <<<")
    
    def on_release():
        print("\n<<< 按钮释放 >>>")
    
    button.on_press = on_press
    button.on_release = on_release
    
    try:
        button.start_monitoring(debounce_ms=50)
        print("监控中... 按 Ctrl+C 退出")
        
        while True:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n\n退出程序")
    finally:
        button.cleanup()


def print_usage():
    """打印使用说明"""
    print("使用方法:")
    print("  python button_reader.py polling   # 轮询模式")
    print("  python button_reader.py wait      # 等待按下模式")
    print("  python button_reader.py callback  # 回调模式(事件驱动)")
    print()
    print("默认使用轮询模式")


if __name__ == "__main__":
    import sys
    
    print("Orange Pi 4 Pro 按钮读取程序")
    print(f"GPIO: GPIO364 (sysfs #{GPIO_NUM}, 排针13号)")
    print()
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode == "polling":
            demo_polling()
        elif mode == "wait":
            demo_wait()
        elif mode == "callback":
            demo_callback()
        else:
            print(f"未知模式: {mode}")
            print_usage()
    else:
        print_usage()
