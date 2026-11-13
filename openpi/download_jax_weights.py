import os
import requests

# 自动获取当前脚本所在的 openpi 目录，并将其存放于该目录下的 ckpt 文件夹
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR = os.path.join(BASE_DIR, "ckpt")

# GCS REST API
api_url = "https://storage.googleapis.com/storage/v1/b/openpi-assets/o?prefix=checkpoints/pi05_base/"

print(f"存放目录: {DOWNLOAD_DIR}")
print("正在获取原生 JAX 权重文件列表...")
try:
    res = requests.get(api_url)
    items = res.json().get('items', [])
except Exception as e:
    print(f"网络请求失败: {e}")
    items = []

if not items:
    print("获取文件列表失败，请检查您的网络代理 (HTTP_PROXY / HTTPS_PROXY)")
    exit(1)

for item in items:
    name = item['name']
    # 构造实际的下载直链
    url = f"https://storage.googleapis.com/openpi-assets/{name}"
    
    # 映射到本地目录：将 checkpoints/pi05_base/... 映射为 ckpt/pi05_base/...
    local_path = os.path.join(DOWNLOAD_DIR, name.replace("checkpoints/", ""))
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 支持断点续传/跳过已下载
    if os.path.exists(local_path) and os.path.getsize(local_path) == int(item['size']):
        print(f"已跳过 {name} (已完整下载)")
        continue
        
    print(f"正在下载 {name} ({int(item['size'])/1024/1024:.2f} MB)...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    except Exception as e:
        print(f"下载失败: {e}")
        
print("🎉 所有 JAX 原生权重下载完成！")
