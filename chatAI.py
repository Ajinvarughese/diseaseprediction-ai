import os
from dotenv import load_dotenv
from openai import OpenAI
from entity import *
from typing import Union, Literal
# ================== SYSTEM PROMPTS ==================

SYSTEM_PROMPT = """
You are a friendly finance assistant.

Reply like WhatsApp messages:
- short and clear
- human tone
- use a few emojis 🙂

You mainly help with finance-related questions (money, savings, loans, budgeting, investing).

If the user asks something completely unrelated to finance,
politely reply that you are here to help with financial issues only.
"""


SYSTEM_PROMPT_FOR_DESCRIPTION = """
You are a medical health summary generator.

Rules:
- Write exactly ONE paragraph.
- 2–3 complete sentences only.
- Do NOT end mid-sentence.
- Ensure the last sentence is fully completed.
- No emojis.
- No bullet points.
- No line breaks.
- No greetings.
- Use a professional, clear, and medically responsible tone.
- Do not provide a definitive diagnosis.
- Include a brief recommendation to consult a qualified healthcare professional when appropriate.
"""

SYSTEM_PROMPT_FOR_CLASSIFICATION = """
You are a medical risk classification engine.

Classify whether the patient's heart disease, diabetes, or Parkinsons Disease based on the given informations.

Rules:
- Output ONLY one word.
- Valid outputs are: SAFE, RISKY, HIGH.
- Do NOT explain.
- Do NOT include punctuation or extra text.

"""

# ================== CLIENT (cached) ==================

_client = None

def get_client():
    global _client
    if _client is None:
        load_dotenv()
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not found")
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
    return _client

# ================== HELPERS ==================

def clean_text(text) -> str:
    if not text or not isinstance(text, str):
        return ""
    return text.replace("\u0000", "").strip()


def build_messages(user_input, chat_log, asset, liability, user):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_log)

    messages.append({"role": "system", "content": f"User name: {user}"})
    messages.append({"role": "system", "content": f"Assets: {asset}"})
    messages.append({"role": "system", "content": f"Liabilities: {liability}"})
    messages.append({"role": "user", "content": user_input})

    return messages

# ================== AI CHAT ==================

# def askAI(
#     user_input: str,
#     chat_log: list = [],
#     asset: list = [],
#     liability: list = [],
#     user: str = ""
# ) -> str:
#     client = get_client()
#     messages = build_messages(user_input, chat_log, asset, liability, user)

#     completion = client.chat.completions.create(
#         model="openai/gpt-oss-20b",
#         messages=messages,
#         temperature=1,
#         max_tokens=4096,
#         stream=True
#     )

#     response_chunks = []

#     for chunk in completion:
#         if chunk.choices and chunk.choices[0].delta.content:
#             response_chunks.append(chunk.choices[0].delta.content)

#     response_text = clean_text("".join(response_chunks))

#     chat_log.append({"role": "user", "content": clean_text(user_input)})
#     chat_log.append({"role": "assistant", "content": response_text})

#     return response_text

# ================== RISK CLASSIFICATION ==================


def safe_message_content(msg) -> str:
    if not msg:
        return ""
    if isinstance(msg, str):
        return msg.strip()
    if isinstance(msg, list):
        return " ".join(
            part.get("text", "") for part in msg if isinstance(part, dict)
        ).strip()
    return ""

def aiRecommendation(
        dataset: Union[Heart, Diabetes, Parkinsons],
        risk_class: bool,
        disease: Literal["HEART", "DIABETES", "PARKINSONS"]
    ) -> str:

    client = get_client()
    condition_text = (
        f"This person have {disease}" if risk_class  else f"This person does not have {disease}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FOR_DESCRIPTION},
        {
            "role": "user",
            "content": f"""
            {dataset}

            predicted condition: {condition_text}
            Write a small content to inform the user about his health condition
            """
        }
    ]
   
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0,
        max_tokens=4096
    )

    return safe_message_content(completion.choices[0].message.content)

def classifyRiskClass(prediction: bool, desc: str, disease: Literal["HEART", "DIABETES", "PARKINSONS"]) -> RiskClass:
    client = get_client()
    condition_text = (
        f"This person have {disease}" if prediction  else f"This person does not have {disease}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_FOR_CLASSIFICATION},
        {
            "role": "user",
            "content": f"""
            Prediction: {condition_text}
            AI recommendation message: {desc}

            Now predict classification: 
            Must say SAFE or RISKY or HIGH
            """
        }
    ]
   
    completion = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=messages,
        temperature=0,
        max_tokens=500
    )

    return safe_message_content(completion.choices[0].message.content)
   