import yt_dlp
import os

# ==========================
# 配置区域（你可以改）
# ==========================
SAVE_DIR = "D:\work\Intern\Dataset"   # 保存路径
URL_LIST = [
    # 视频链接
    "https://www.youtube.com/watch?v=eB-83C1JU_w",
]

# 是否从文件读取（批量）
USE_TXT = False
TXT_PATH = "urls.txt"

# ==========================
# 创建保存目录
# ==========================
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# ==========================
# yt-dlp 下载配置
# ==========================
ydl_opts = {
    # 下载最高质量视频+音频
    'format': 'bestvideo+bestaudio/best',

    # 输出路径格式
    'outtmpl': os.path.join(SAVE_DIR, '%(title)s.%(ext)s'),

    # 多线程下载（加速）
    'n_threads': 8,

    # 忽略错误（某些视频可能失败）
    'ignoreerrors': True,

    # 自动合并音视频
    'merge_output_format': 'mp4',

    # 显示进度
    'progress_hooks': [],
}

# ==========================
# 下载函数
# ==========================
def download_videos(urls):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                print(f"\n开始下载: {url}")
                ydl.download([url])
                print("下载完成")
            except Exception as e:
                print(f"下载失败: {url}")
                print(e)

# ==========================
# 主逻辑
# ==========================
if __name__ == "__main__":

    # 如果使用txt批量
    if USE_TXT:
        if not os.path.exists(TXT_PATH):
            print("找不到 urls.txt")
            exit()

        with open(TXT_PATH, "r") as f:
            urls = [line.strip() for line in f if line.strip()]
    else:
        urls = URL_LIST

    print(f"共 {len(urls)} 个视频待下载")

    download_videos(urls)

    print("\n全部任务完成！")