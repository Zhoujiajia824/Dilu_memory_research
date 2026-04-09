import json
import os
from dataclasses import dataclass

import requests

# 兼容不同版本的 LangChain
try:
    from langchain_core.messages import AIMessage, AIMessageChunk
except ImportError:
    try:
        from langchain.schema import AIMessage
        # AIMessageChunk 可能不存在，创建一个简单的替代类
        class AIMessageChunk:
            def __init__(self, content: str):
                self.content = content
    except ImportError:
        # 如果 langchain 也没有安装，创建基础类
        class AIMessage:
            def __init__(self, content: str):
                self.content = content
                self.type = "ai"
        class AIMessageChunk:
            def __init__(self, content: str):
                self.content = content


DEFAULTS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "chat_model": "qwen-max",
        "embed_model": "text-embedding-v3",
    },
    "glm": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "chat_model": "glm-4-plus",
        "embed_model": "embedding-3",
    },
}

_MODEL_CONFIG = {}


class ProviderAPIError(RuntimeError):
    pass


@dataclass
class ModelConfig:
    provider: str
    api_key: str
    base_url: str
    chat_model: str
    embed_model: str
    timeout: int
    embed_batch_size: int


def _normalize_provider(provider):
    if not provider:
        return "qwen"
    return str(provider).strip().lower()


def _provider_title(provider):
    return provider.upper() if provider == "glm" else provider.capitalize()


def _build_provider_config(config, provider):
    defaults = DEFAULTS[provider]
    prefix = provider.upper()
    api_key = config.get(f"{prefix}_API_KEY")
    if not api_key:
        raise ValueError(f"Missing required config: {prefix}_API_KEY")
    chat_model = config.get(f"{prefix}_CHAT_MODEL") or defaults["chat_model"]
    embed_model = config.get(f"{prefix}_EMBED_MODEL") or defaults["embed_model"]
    base_url = (config.get(f"{prefix}_BASE_URL") or defaults["base_url"]).rstrip("/")
    timeout = int(config.get("MODEL_TIMEOUT", 60))
    embed_batch_size = int(config.get("EMBED_BATCH_SIZE", 8))
    return ModelConfig(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        chat_model=chat_model,
        embed_model=embed_model,
        timeout=timeout,
        embed_batch_size=embed_batch_size,
    )


def setup_model_env(config):
    provider = _normalize_provider(config.get("MODEL_PROVIDER"))
    if provider not in DEFAULTS:
        raise ValueError("Unknown MODEL_PROVIDER, should be qwen or glm")

    global _MODEL_CONFIG
    _MODEL_CONFIG = _build_provider_config(config, provider).__dict__.copy()
    os.environ["DILU_MODEL_PROVIDER"] = provider
    os.environ["DILU_CHAT_MODEL"] = _MODEL_CONFIG["chat_model"]
    os.environ["DILU_EMBED_MODEL"] = _MODEL_CONFIG["embed_model"]


def _require_model_config():
    if not _MODEL_CONFIG:
        raise RuntimeError("Model config is not initialized. Call setup_model_env(config) first.")
    return dict(_MODEL_CONFIG)


def _build_runtime_config(provider=None, model=None, api_key=None, base_url=None):
    cfg = _require_model_config()
    if provider is not None:
        provider = _normalize_provider(provider)
        if provider != cfg["provider"]:
            raise ValueError("Per-call provider override is not supported after setup_model_env().")
    if model is not None:
        cfg["chat_model"] = model
    if api_key is not None:
        cfg["api_key"] = api_key
    if base_url is not None:
        cfg["base_url"] = base_url.rstrip("/")
    return cfg


def _message_role(message):
    msg_type = getattr(message, "type", "")
    if msg_type == "human":
        return "user"
    if msg_type == "ai":
        return "assistant"
    if msg_type == "system":
        return "system"
    return "user"


def _serialize_messages(messages):
    return [
        {"role": _message_role(message), "content": message.content}
        for message in messages
    ]


def _extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif "text" in item:
                    parts.append(item.get("text", ""))
        return "".join(parts)
    if isinstance(content, dict):
        if "text" in content:
            return content.get("text", "")
    return "" if content is None else str(content)


def _extract_choice_text(choice):
    if "message" in choice:
        return _extract_text(choice["message"].get("content"))
    if "delta" in choice:
        return _extract_text(choice["delta"].get("content"))
    if "text" in choice:
        return _extract_text(choice.get("text"))
    return ""


def _post_json(url, headers, payload, timeout, stream=False):
    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=timeout,
            stream=stream,
        )
        response.raise_for_status()
        return response
    except requests.RequestException as exc:
        raise ProviderAPIError(f"Request failed for {url}: {exc}") from exc


class NativeChatModel:
    def __init__(
        self,
        provider=None,
        model=None,
        api_key=None,
        base_url=None,
        temperature=0.0,
        max_tokens=2000,
        request_timeout=None,
        streaming=False,
        callbacks=None,
    ):
        self.config = _build_runtime_config(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout or self.config["timeout"]
        self.streaming = streaming

    def __call__(self, messages):
        content = self._completion(messages)
        return AIMessage(content=content)

    def stream(self, messages):
        if not self.streaming:
            yield AIMessageChunk(content=self._completion(messages))
            return

        payload = {
            "model": self.config["chat_model"],
            "messages": _serialize_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{self.config['base_url']}/chat/completions"
        response = _post_json(url, headers, payload, self.request_timeout, stream=True)

        for raw_line in response.iter_lines(decode_unicode=True):
            if not raw_line:
                continue
            line = raw_line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta_text = _extract_choice_text(choices[0])
            if delta_text:
                yield AIMessageChunk(content=delta_text)

    def _completion(self, messages):
        payload = {
            "model": self.config["chat_model"],
            "messages": _serialize_messages(messages),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.config['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{self.config['base_url']}/chat/completions"
        response = _post_json(url, headers, payload, self.request_timeout)
        data = response.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderAPIError(f"Malformed chat response from {self.config['provider']}: {data}")
        return _extract_choice_text(choices[0])


class NativeEmbeddingModel:
    def __init__(
        self,
        provider=None,
        model=None,
        api_key=None,
        base_url=None,
        batch_size=None,
    ):
        cfg = _build_runtime_config(
            provider=provider,
            model=None,
            api_key=api_key,
            base_url=base_url,
        )
        self.provider = cfg["provider"]
        self.api_key = cfg["api_key"]
        self.base_url = cfg["base_url"]
        self.model = model or cfg["embed_model"]
        self.batch_size = batch_size or cfg["embed_batch_size"]

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self.batch_size]))
        return vectors

    def _embed_batch(self, texts):
        payload = {
            "model": self.model,
            "input": texts,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        url = f"{self.base_url}/embeddings"
        response = _post_json(url, headers, payload, _require_model_config()["timeout"])
        data = response.json()
        items = sorted(data.get("data") or [], key=lambda item: item.get("index", 0))
        if not items:
            raise ProviderAPIError(f"Malformed embedding response from {self.provider}: {data}")
        return [item["embedding"] for item in items]


def build_chat_model(
    provider=None,
    model=None,
    api_key=None,
    base_url=None,
    temperature=0.0,
    max_tokens=2000,
    request_timeout=60,
    streaming=False,
    callbacks=None,
):
    return NativeChatModel(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        streaming=streaming,
        callbacks=callbacks,
    )


def build_chat_llm(
    temperature=0.0,
    max_tokens=2000,
    request_timeout=60,
    streaming=False,
    callbacks=None,
):
    return build_chat_model(
        temperature=temperature,
        max_tokens=max_tokens,
        request_timeout=request_timeout,
        streaming=streaming,
        callbacks=callbacks,
    )


def build_embedding_model(
    provider=None,
    model=None,
    api_key=None,
    base_url=None,
    batch_size=None,
):
    return NativeEmbeddingModel(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        batch_size=batch_size,
    )


def get_model_label():
    cfg = _require_model_config()
    return f"{_provider_title(cfg['provider'])} ({cfg['chat_model']})"


def get_embedding_signature():
    cfg = _require_model_config()
    return {
        "provider": cfg["provider"],
        "embed_model": cfg["embed_model"],
    }
