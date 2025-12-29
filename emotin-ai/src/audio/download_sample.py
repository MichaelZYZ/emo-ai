import requests

# RAVDESS公开样本（如：Actor_01/03-01-01-01-01-01-01.wav）
# 该文件可公开下载，示例链接如下（如失效可更换）：
url = 'https://github.com/yaoxingyu/emotion_dataset/raw/master/RAVDESS/Actor_01/03-01-01-01-01-01-01.wav'
output_path = 'sample.wav'

print('正在下载公开情绪语音样本...')
r = requests.get(url)
with open(output_path, 'wb') as f:
    f.write(r.content)
print('下载完成，已保存为 sample.wav')
