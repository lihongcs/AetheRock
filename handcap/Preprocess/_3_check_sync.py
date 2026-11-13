import argparse
import os
import socket
import struct
import time
import sys

# ================= 配置区域 =================
# 允许的最大误差（毫秒），超过此值会报警
MAX_OFFSET_MS = 50 

# NTP 协议的时间基准差 (1900年到1970年的秒数)
NTP_DELTA = 2208988800
# ===========================================

def get_ntp_offset(host, port=123):
    """
    发送 NTP 请求并计算本地与服务器的时间偏移量 (Offset)
    返回: offset_seconds (float) 或 None (失败)
    """
    # 创建 UDP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(2.0) # 2秒超时

    # 构造 NTP v3 客户端请求包 (48字节)
    # 0x1B = 00011011 (LI=0, VN=3, Mode=3 Client)
    msg = b'\x1b' + 47 * b'\0'

    try:
        # T1: 客户端发送请求的时间
        t1 = time.time()
        sock.sendto(msg, (host, port))

        # T4: 客户端收到响应的时间
        data, addr = sock.recvfrom(1024)
        t4 = time.time()

        if data:
            # 解析 NTP 数据包 (大端序 12个无符号整数)
            unpacked = struct.unpack('!12I', data)
            
            # 提取服务器接收时间 (T2)
            t2_seconds = unpacked[8] - NTP_DELTA
            t2_fraction = unpacked[9]
            t2 = t2_seconds + t2_fraction / 2**32

            # 提取服务器发送时间 (T3)
            t3_seconds = unpacked[10] - NTP_DELTA
            t3_fraction = unpacked[11]
            t3 = t3_seconds + t3_fraction / 2**32

            # 计算偏移量 Offset = ((T2 - T1) + (T3 - T4)) / 2
            # Offset > 0 表示服务器比本地快
            # Offset < 0 表示服务器比本地慢
            offset = ((t2 - t1) + (t3 - t4)) / 2
            return offset
            
    except socket.timeout:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        sock.close()

def main():
    parser = argparse.ArgumentParser(description="Check the clock offset against an NTP host.")
    parser.add_argument("host", nargs="?", default=os.environ.get("HANDCAP_NTP_HOST"))
    args = parser.parse_args()
    if not args.host:
        parser.error("supply an NTP host or set HANDCAP_NTP_HOST")

    print(f"正在连接 {args.host} 检查时间同步...")
    print(f"标准: 误差需小于 {MAX_OFFSET_MS}ms")
    print("-" * 60)
    print(f"{'时间':<10} | {'偏移量 (ms)':<15} | {'状态':<10} | {'判定'}")
    print("-" * 60)

    try:
        while True:
            offset = get_ntp_offset(args.host)
            current_time_str = time.strftime("%H:%M:%S", time.localtime())

            if isinstance(offset, (int, float)):
                offset_ms = offset * 1000
                abs_ms = abs(offset_ms)
                
                # 状态判定
                if abs_ms <= MAX_OFFSET_MS:
                    status = "✅ 正常"
                    color_code = "\033[92m" # 绿色
                else:
                    status = "❌ 异常"
                    color_code = "\033[91m" # 红色
                
                reset_code = "\033[0m"
                
                # 打印结果
                print(f"{current_time_str:<10} | {color_code}{offset_ms:10.4f} ms{reset_code}    | {status:<10} | ", end="")
                
                if abs_ms < 1.0:
                    print("(极佳)")
                elif abs_ms < 10.0:
                    print("(良好)")
                elif abs_ms > MAX_OFFSET_MS:
                    print(f"⚠️ 超过 {MAX_OFFSET_MS}ms!")
                else:
                    print("(合格)")

            elif offset == "TIMEOUT":
                print(f"{current_time_str:<10} | {'TIMEOUT':<15} | ⚠️ 超时      | 网络不通或被防火墙拦截")
            else:
                print(f"{current_time_str:<10} | {'ERROR':<15} | ⚠️ 错误      | {offset}")

            # 每秒检查一次
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n检查已停止。")

if __name__ == "__main__":
    main()