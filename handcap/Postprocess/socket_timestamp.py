import argparse
import datetime
import os
import shlex
import socket
import statistics
import time


PORT = 12345
NUM_ITERATIONS = 100


def main():
    parser = argparse.ArgumentParser(description="Measure the clock offset from a remote timestamp server.")
    parser.add_argument("host", help="IP address or hostname of the timestamp server")
    parser.add_argument("--ssh-user", help="SSH username for starting the server remotely")
    parser.add_argument("--remote-script", help="Path to time_server.py on the SSH host")
    args = parser.parse_args()
    if bool(args.ssh_user) != bool(args.remote_script):
        parser.error("--ssh-user and --remote-script must be supplied together")

    ssh_client = None
    remote_pid = None
    try:
        if args.remote_script:
            import paramiko

            ssh_client = paramiko.SSHClient()
            ssh_client.load_system_host_keys()
            ssh_client.connect(
                hostname=args.host,
                username=args.ssh_user,
                password=os.environ.get("HANDCAP_PI_PASSWORD"),
            )
            remote_path = shlex.quote(args.remote_script)
            command = f"nohup python3 {remote_path} </dev/null >/dev/null 2>&1 & echo $!"
            _, stdout, _ = ssh_client.exec_command(command)
            remote_pid = int(stdout.read().strip())
            time.sleep(2)

        # --- 第 B 部分: 运行本地客户端进行测量 (这是您之前的代码) ---
        time_differences_seconds = []
        successful_measurements = 0
        failed_attempts = 0

        print(f"\n[*] 开始进行 {NUM_ITERATIONS} 次时钟差测量...")
        for i in range(NUM_ITERATIONS):
            try:
                pc_time_before = datetime.datetime.now()
                with socket.create_connection((args.host, PORT), timeout=2.0) as sock:
                    pi_time_str = sock.recv(1024).decode("utf-8")
                    pc_time_after = datetime.datetime.now()

                pi_time = datetime.datetime.fromisoformat(pi_time_str)
                pc_time_avg = pc_time_before + (pc_time_after - pc_time_before) / 2
                time_difference = pi_time - pc_time_avg
            
                time_differences_seconds.append(time_difference.total_seconds())
                successful_measurements += 1
            except (OSError, ValueError):
                # 简化错误处理
                failed_attempts += 1
            # 打印一个简单的进度
            print(f"\r    测量进度: {i+1}/{NUM_ITERATIONS}", end="")
            time.sleep(0.05) # 稍微缩短等待时间

        print("\n\n" + "="*30)
        print("测量完成，开始统计结果...")
        print("="*30)
    
        if successful_measurements > 0:
            average_diff = statistics.mean(time_differences_seconds)
            min_diff = min(time_differences_seconds)
            max_diff = max(time_differences_seconds)
            std_dev = statistics.stdev(time_differences_seconds) if successful_measurements > 1 else 0.0

            print(f"成功测量次数: {successful_measurements}")
            print(f"失败尝试次数: {failed_attempts}")
            print("-" * 30)
            print(f"平均时钟差: {average_diff:+.9f} 秒")
            print(f"最小时钟差: {min_diff:+.9f} 秒")
            print(f"最大时钟差: {max_diff:+.9f} 秒")
            print(f"标准差 (稳定性): {std_dev:.9f} 秒")
        else:
            print("所有测量均失败！")


    finally:
        if ssh_client is not None:
            try:
                if remote_pid is not None:
                    ssh_client.exec_command(f"kill -- {remote_pid}")
            finally:
                ssh_client.close()


if __name__ == "__main__":
    main()
