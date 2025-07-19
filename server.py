import json
import requests
import gradio as gr
from typing import Iterator, List, Dict, Tuple
from datetime import datetime
import os
import tiktoken
import glob
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化tokenizer
encoding = tiktoken.get_encoding("cl100k_base")

# 配置常量
MAX_TOKENS = 4000
MAX_HISTORY_ITEMS = 20
CONVERSATION_DIR = os.path.join("data", "conversations")
TRAINING_DATA_DIR = os.path.join("data", "training_data")

# 确保数据目录存在
os.makedirs(CONVERSATION_DIR, exist_ok=True)
os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

def convert_to_openai_format(chat_history: List[Tuple[str, str]]) -> List[Dict]:
    """将对话历史转换为OpenAI格式"""
    formatted = []
    for item in chat_history:
        if len(item) == 2:  # 确保是(user_msg, bot_msg)格式
            user_msg, bot_msg = item
            formatted.append({"role": "user", "content": user_msg})
            formatted.append({"role": "assistant", "content": bot_msg})
    return formatted

def convert_from_openai_format(formatted_history: List[Dict]) -> List[Tuple[str, str]]:
    """从OpenAI格式转换回元组格式"""
    chat_history = []
    for i in range(0, len(formatted_history), 2):
        if i+1 < len(formatted_history):
            user_msg = formatted_history[i]["content"]
            bot_msg = formatted_history[i+1]["content"]
            chat_history.append((user_msg, bot_msg))
    return chat_history

def load_profile():
    """加载用户配置文件"""
    try:
        with open('profile.json', 'r', encoding='utf-8') as f:
            profile = json.load(f)
            return profile['my_profile']
    except FileNotFoundError:
        print("错误: profile.json 文件未找到")
        default_profile = {
            "my_profile": {
                "name": "用户",
                "age": 20,
                "profession": "未设置",
                "interests": ["未设置"],
                "memory": []
            }
        }
        with open('profile.json', 'w', encoding='utf-8') as f:
            json.dump(default_profile, f, ensure_ascii=False, indent=2)
        return default_profile['my_profile']
    except Exception as e:
        print(f"加载配置文件错误: {str(e)}")
        return None

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def trim_conversation(conversation: List[Tuple[str, str]], max_tokens: int) -> List[Tuple[str, str]]:
    total_tokens = 0
    trimmed = []
    
    for item in reversed(conversation):
        if len(item) != 2:  # 跳过格式不正确的条目
            continue
            
        user_msg, bot_msg = item
        item_tokens = count_tokens(user_msg) + count_tokens(bot_msg)
        
        if total_tokens + item_tokens > max_tokens:
            break
            
        trimmed.insert(0, item)
        total_tokens += item_tokens
    
    return trimmed

def save_conversation(conversation: List[Tuple[str, str]], filename: str = None) -> str:
    """保存对话到文件"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CONVERSATION_DIR, f"conversation_{timestamp}.json")
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(convert_to_openai_format(conversation), f, ensure_ascii=False, indent=2)
        return filename
    except Exception as e:
        print(f"保存对话失败: {str(e)}")
        return ""

def load_conversation(filename: str) -> List[Tuple[str, str]]:
    """从文件加载对话"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            formatted_history = json.load(f)
            return convert_from_openai_format(formatted_history)
    except Exception as e:
        print(f"加载对话失败: {str(e)}")
        return []

def get_conversation_files() -> List[str]:
    """获取所有对话文件，按时间排序"""
    files = glob.glob(os.path.join(CONVERSATION_DIR, "conversation_*.json"))
    return sorted(files, key=os.path.getmtime, reverse=True)

def load_recent_conversations(max_count=30) -> List[Tuple[str, str]]:
    """加载最近的对话记录"""
    conversation_files = get_conversation_files()
    recent_conversations = []
    
    for file in conversation_files[:max_count]:
        try:
            conversation = load_conversation(file)
            recent_conversations.extend(conversation)
        except Exception as e:
            print(f"加载对话文件 {file} 失败: {str(e)}")
    
    return recent_conversations[-max_count:]

def prepare_api_messages(conversation: List[Tuple[str, str]], profile: dict, new_message: str) -> List[Dict]:
    system_prompt = {
        "role": "system",
        "content": f"""你正在与{profile['name']}对话:
        年龄: {profile['age']}
        职业: {profile['profession']}
        兴趣: {', '.join(profile['interests'])}"""
    }
    
    messages = [system_prompt]
    tokens_used = count_tokens(system_prompt["content"])
    
    for item in conversation[-MAX_HISTORY_ITEMS:]:
        if len(item) != 2:  # 跳过格式不正确的条目
            continue
            
        user_msg, bot_msg = item
        user_content = {"role": "user", "content": user_msg}
        bot_content = {"role": "assistant", "content": bot_msg}
        
        new_tokens = count_tokens(user_msg) + count_tokens(bot_msg)
        if tokens_used + new_tokens > MAX_TOKENS:
            break
            
        messages.extend([user_content, bot_content])
        tokens_used += new_tokens
    
    if count_tokens(new_message) + tokens_used < MAX_TOKENS:
        messages.append({"role": "user", "content": new_message})
    
    return messages

def call_deepseek_api_stream(prompt: str, conversation: List[Tuple[str, str]], profile: dict) -> Iterator[str]:
    api_url = "https://api.deepseek.com/v1/chat/completions"
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        yield "[错误] 未设置API密钥，请在.env文件中配置DEEPSEEK_API_KEY"
        return
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = prepare_api_messages(conversation, profile, prompt)
    
    data = {
        "model": "deepseek-chat",
        "messages": messages,
        "stream": True,
        "max_tokens": min(2000, MAX_TOKENS - count_tokens(str(messages)))
    }
    
    try:
        with requests.post(api_url, headers=headers, json=data, stream=True) as response:
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        if decoded_line.startswith("data:"):
                            json_data = decoded_line[5:].strip()
                            if json_data != "[DONE]":
                                try:
                                    chunk = json.loads(json_data)
                                    if "choices" in chunk and chunk["choices"]:
                                        content = chunk["choices"][0].get("delta", {}).get("content", "")
                                        if content:
                                            yield content
                                except json.JSONDecodeError:
                                    pass
            else:
                yield f"[API 错误] 状态码: {response.status_code}"
    except Exception as e:
        yield f"[连接错误] {str(e)}"

def respond(message: str, chat_history: List[Tuple[str, str]], profile: dict, current_file: str):
    if not message.strip():
        yield chat_history, current_file
        return
    
    try:
        # 加载最近30次对话并合并
        recent_chats = load_recent_conversations()
        combined_history = recent_chats + chat_history
        
        # 清理历史记录中的无效条目
        cleaned_history = [item for item in combined_history if len(item) == 2]
        
        trimmed_history = trim_conversation(cleaned_history, MAX_TOKENS // 2)
        
        bot_message = ""
        for chunk in call_deepseek_api_stream(message, trimmed_history, profile):
            bot_message += chunk
            temp_history = trimmed_history + [(message, bot_message)]
            yield temp_history, current_file
        
        full_conversation = trimmed_history + [(message, bot_message)]
        current_file = save_conversation(full_conversation, current_file)
        yield full_conversation, current_file
    except Exception as e:
        print(f"对话出错: {str(e)}")
        error_msg = f"发生错误: {str(e)}"
        error_history = chat_history + [(message, error_msg)]
        yield error_history, current_file

def create_interface(profile: dict):
    with gr.Blocks(title="DeepSeek 聊天助手", theme=gr.themes.Soft()) as demo:
        # 初始化时直接加载最近30次对话
        initial_history = load_recent_conversations()
        current_file = gr.State("")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 用户资料")
                gr.Markdown(f"""
                **姓名**: {profile['name']}  
                **年龄**: {profile['age']}  
                **职业**: {profile['profession']}  
                **兴趣**: {', '.join(profile['interests'])}
                """)
                
        chatbot = gr.Chatbot(
            value=initial_history,
            height=500,
            avatar_images=(
                "https://avatars.githubusercontent.com/u/14957082?s=200&v=4",
                "./icon.png"
            ),
            show_copy_button=True,
            layout="panel"
        )
        
        with gr.Row():
            msg = gr.Textbox(
                placeholder="输入你的消息...",
                show_label=False,
                container=False,
                autofocus=True,
                scale=4
            )
            submit_btn = gr.Button("发送", variant="primary", scale=1)
            clear = gr.Button("清空", scale=1)
        
        msg.submit(
            respond,
            [msg, chatbot, gr.State(profile), current_file],
            [chatbot, current_file]
        ).then(
            lambda: "", None, msg
        )
        
        submit_btn.click(
            respond,
            [msg, chatbot, gr.State(profile), current_file],
            [chatbot, current_file]
        ).then(
            lambda: "", None, msg
        )
        
        clear.click(lambda: [], None, chatbot, queue=False)
    
    return demo

if __name__ == "__main__":
    profile = load_profile()
    if profile:
        demo = create_interface(profile)
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True
        )
    else:
        print("无法加载用户配置，请检查profile.json文件")