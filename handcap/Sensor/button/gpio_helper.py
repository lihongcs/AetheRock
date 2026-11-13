#!/usr/bin/env python3
"""
Orange Pi 5 Pro GPIO 模块
GPIO1_A6 -> Pin 31 (sysfs #38)

用法:
  from gpio_helper import GPIO, wait_for_press

  # 方式1: 读取当前值 (返回 0 或 1)
  value = GPIO.read()

  # 方式2: 等待按下 (阻塞直到按钮按下)
  wait_for_press()

  # 方式3: 检测状态变化 (有变化返回新值, 否则返回0)
  changed = GPIO.wait_for_change(timeout=5)
"""

import os
import time
from typing import Optional

# ============== 配置 ==============
GPIO_NUM = 38  # GPIO1_A6 对应的 sysfs 编号
GPIO_PATH = f"/sys/class/gpio/gpio{GPIO_NUM}"
# ===================================


class GPIO:
    """GPIO 操作类"""
    
    _initialized = False
    
    @classmethod
    def init(cls) -> None:
        """初始化GPIO (只需调用一次)"""
        if cls._initialized:
            return
        
        if not os.path.exists(GPIO_PATH):
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(GPIO_NUM))
            time.sleep(0.1)
        
        with open(f"{GPIO_PATH}/direction", "w") as f:
            f.write("in")
        
        cls._initialized = True
    
    @classmethod
    def read(cls) -> int:
        """
        读取GPIO当前值
        
        Returns:
            0: 低电平 (按钮按下)
            1: 高电平 (按钮松开)
        """
        cls.init()
        with open(f"{GPIO_PATH}/value", "r") as f:
            return int(f.read().strip())
    
    @classmethod
    def wait_for_change(cls, timeout: Optional[float] = None) -> Optional[int]:
        """
        等待GPIO状态变化
        
        Args:
            timeout: 超时时间(秒), None表示无限等待
            
        Returns:
            变化后的值 (0或1), 超时返回None
        """
        cls.init()
        
        fd = os.open(f"{GPIO_PATH}/value", os.O_RDONLY)
        last_value = int(os.read(fd, 1))
        
        try:
            start = time.time()
            
            while True:
                os.lseek(fd, 0, os.SEEK_SET)
                value = int(os.read(fd, 1))
                
                if value != last_value:
                    return value
                
                if timeout and (time.time() - start) >= timeout:
                    return None
                
                time.sleep(0.01)
        finally:
            os.close(fd)
    
    @classmethod
    def cleanup(cls) -> None:
        """释放GPIO资源"""
        try:
            with open("/sys/class/gpio/unexport", "w") as f:
                f.write(str(GPIO_NUM))
            cls._initialized = False
        except:
            pass


def read() -> int:
    """
    读取GPIO当前值 (便捷函数)
    
    Returns:
        0: 低电平
        1: 高电平
    """
    return GPIO.read()


def wait_for_press(timeout: Optional[float] = None) -> bool:
    """
    等待按钮按下 (阻塞函数)
    
    Args:
        timeout: 超时时间(秒)
        
    Returns:
        True: 按钮被按下
        False: 超时
    """
    result = GPIO.wait_for_change(timeout=timeout)
    return result is not None


def wait_for_change(timeout: Optional[float] = None) -> Optional[int]:
    """
    等待状态变化并返回新值
    
    Args:
        timeout: 超时时间(秒)
        
    Returns:
        变化后的值 (0或1), 超时返回None
    """
    return GPIO.wait_for_change(timeout=timeout)
