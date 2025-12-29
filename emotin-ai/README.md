# 智能多模态情绪识别系统

## 项目简介
本项目结合文本（NLP）、语音（Audio）、表情（CV）三种模态，实现对用户情绪的综合识别，输出“愤怒/开心/中性/悲伤/惊讶/恐惧”等情绪类别。

## 目录结构
- data/         # 数据集存放
- src/          # 主要源码
- notebooks/    # Jupyter实验
- main.py       # 主入口

## 快速开始
1. 安装依赖：`pip install -r requirements.txt`
2. 数据准备：下载公开情绪数据集，放入对应data子目录
3. 运行主程序：`python main.py`

## 依赖
见 requirements.txt

## 主要功能模块
- 文本情绪识别（src/text/）
- 语音情绪识别（src/audio/）
- 表情情绪识别（src/image/）
- 多模态融合（src/fusion/）
