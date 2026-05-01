import os
import time
import json
import hashlib
from groq import Groq, RateLimitError
from dotenv import load_dotenv

load_dotenv()

# Load all available Groq keys from .env
# Keys should be named GROQ_API_KEY_1, GROQ_API_KEY_2, etc.
API_KEYS = []
for i in range(1, 11): # Support up to 10 keys
    key = os.getenv(f"GROQ_API_KEY_{i}")
    if key:
        API_KEYS.append(key)

# Fallback to the original single key if no numbered keys are found
if not API_KEYS and os.getenv("GROQ_API_KEY"):
    API_KEYS.append(os.getenv("GROQ_API_KEY"))

if not API_KEYS:
    raise Exception("No GROQ_API_KEY found in .env. Please add GROQ_API_KEY_1...10")

# Index to track which key to use
_key_index = 0

def get_client():
    global _key_index
    key = API_KEYS[_key_index % len(API_KEYS)]
    return Groq(api_key=key)

def rotate_key():
    global _key_index
    _key_index += 1
    print(f"🔄 Rotating to API Key #{(_key_index % len(API_KEYS)) + 1}")

# Create cache directory
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_key(system_prompt: str, user_content: str) -> str:
    combined = f"{system_prompt}\n{user_content}"
    return hashlib.md5(combined.encode()).hexdigest()

def call_llm(system_prompt, user_content, label="LLM", max_tokens=3000, debug=False, use_cache=True):
    cache_key = get_cache_key(system_prompt, user_content)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.json")

    if use_cache and os.path.exists(cache_path):
        if debug:
            print(f"[Stage {label}] Using cached response.")
        with open(cache_path, "r") as f:
            return json.load(f)["result"]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    max_retries = len(API_KEYS) * 2 # Retry across all keys twice
    for attempt in range(max_retries):
        try:
            client = get_client()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1
            )
            raw = response.choices[0].message.content.strip()
            
            if debug:
                print(f"[Stage {label}] Raw output received (len: {len(raw)})")
            
            # Save to cache
            with open(cache_path, "w") as f:
                json.dump({"result": raw, "timestamp": time.time()}, f)
                
            return raw
        except RateLimitError:
            rotate_key()
            # If we've tried all keys, wait a bit
            if (attempt + 1) % len(API_KEYS) == 0:
                wait = 30
                print(f"[{label}] All keys rate limited. Waiting {wait}s...")
                time.sleep(wait)
            continue
    
    raise Exception(f"[Stage {label}] Failed after rotating through all keys.")

def clean_json(raw: str) -> str:
    """Strips markdown code blocks and whitespace."""
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0]
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0]
    return raw.strip()
