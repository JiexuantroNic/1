# DeepSeek Chat Assistant

## 项目简介
基于 DeepSeek API 开发的智能对话助手，支持上下文记忆和个性化配置。

## 核心功能
- 🗣️ 持续对话上下文记忆
- 📂 对话历史自动本地存储
- ⚙️ 可定制的用户配置文件
- 🌊 流式响应交互体验

## 技术栈
- **后端框架**: Gradio
- **API**: DeepSeek
- **数据存储**: JSON 本地存储
- **Token 计算**: tiktoken

## 文件结构
├── data/ # 数据存储
│ ├── conversations/ # 对话历史（JSON）
│ └── training_data/ # 训练数据
├── server.py # 主程序入口
├── profile.json # 用户配置
└── requirements.txt # 依赖库

## 配置说明
`profile.json` 示例：
```json
{
  "my_profile": {
    "name": "用户",
    "age": 25,
    "profession": "职业",
    "interests": ["技术", "阅读"],
    "memory": []
  }

## 快速开始
