import subprocess


def get_windows_ip():
    try:
        # 执行 shell 命令获取网关 IP
        cmd = "ip route | grep default | awk '{print $3}'"

        return subprocess.check_output(cmd, shell=True).decode().strip()

    except Exception:
        return "127.0.0.1"  # 降级备用


if __name__:
    print(get_windows_ip())
