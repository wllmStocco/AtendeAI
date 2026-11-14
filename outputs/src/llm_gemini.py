"""ChatOpenAI adaptado ao endpoint de compatibilidade OpenAI do Gemini 3 (corrige o erro 400 de thought_signature)."""
from langchain_openai import ChatOpenAI

from src.gemini_assinaturas import coletar_assinaturas, injetar_assinaturas


class ChatOpenAIGemini(ChatOpenAI):
    def _get_request_payload(self, input_, *, stop=None, **kwargs):
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        try:
            origem = self._convert_input(input_).to_messages()
            injetar_assinaturas(payload.get("messages", []), coletar_assinaturas(origem))
        except Exception:  # o adaptador nunca deve derrubar a chamada; no pior caso o erro 400 original reaparece
            pass
        return payload
