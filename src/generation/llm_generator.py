import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dotenv import load_dotenv

# Load environment variables (.env file)
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

SYSTEM_PROMPT = """You are a Senior Equity Research & IPO Analyst writing for institutional investors.

Your objective is to provide a clean, insightful, professional research answer based STRICTLY on the retrieved prospectus context provided.

CRITICAL PRESENTATION & TONE RULES (MINIMAL CLUTTER):
1. WRITE FOR HUMAN EXECUTIVES:
   - Provide direct, readable, well-structured answers.
   - DO NOT use excessive markdown hashtags (avoid ###, ####) and avoid horizontal lines (---).
   - Use clean, natural text with bold headers (e.g. **Executive Summary**, **Key Findings**, **Analyst Commentary**).
   - If presenting comparisons, use clean markdown tables or clean bullet comparisons.

2. STRICT FACTUAL GROUNDING:
   - Use ONLY facts, metrics, and statements from the provided context. Never extrapolate or introduce outside information.
   - If evidence is partial or unavailable, clearly state:
     "Insufficient evidence in the provided prospectus context to answer this question."

3. EXPLICIT INLINE CITATIONS:
   - Every factual figure, disclosure, or claim must cite its source in brackets: `[Source: <Source ID>, Page <Page Number>]`.
   - Examples: `[Source: hdbfs_rhp_p56_t2, Page 56]`, `[Source: moe_p86_t1, Page 86]`.
   - Do NOT output a raw text table of sources at the end (the user interface already presents interactive source cards).
"""

USER_PROMPT_TEMPLATE = """RESEARCH QUERY:
{query}

RETRIEVED DRHP CONTEXT:
{context}

Please provide your rigorous, cited analysis following the instructions above.
"""


class ResilientChatGroq:
    """
    Transparent wrapper around ChatGroq that automatically switches to
    GROQ_API_KEY_BACKUP if a 429 rate limit error is encountered.
    """
    def __init__(self, primary_llm: Any, backup_llm: Optional[Any] = None):
        self.primary_llm = primary_llm
        self.backup_llm = backup_llm

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self.primary_llm.invoke(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                # 1. Try backup key if available
                if self.backup_llm is not None:
                    try:
                        return self.backup_llm.invoke(*args, **kwargs)
                    except Exception:
                        pass
                # 2. Try fast secondary model (gpt-oss-20b) with separate token allocation
                try:
                    from langchain_groq import ChatGroq
                    key = os.getenv("GROQ_API_KEY")
                    fallback_llm = ChatGroq(
                        model_name="openai/gpt-oss-20b",
                        api_key=key,
                        temperature=0.0
                    )
                    return fallback_llm.invoke(*args, **kwargs)
                except Exception as fb_err:
                    print("FALLBACK_LLM ERROR:", fb_err)
            raise e

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        primary_so = self.primary_llm.with_structured_output(schema, **kwargs)
        backup_so = self.backup_llm.with_structured_output(schema, **kwargs) if self.backup_llm is not None else None
        return ResilientChatGroq(primary_so, backup_so)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.primary_llm, name)


def get_chat_llm(
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0,
    **kwargs: Any
):
    """
    Factory to instantiate chat model providers modularly.
    Supports Groq (default), OpenAI, Gemini, and Mock.
    """
    provider = provider or os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = kwargs.pop("api_key", None) or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from environment or .env file.")
        # Default to openai/gpt-oss-120b or GROQ_MODEL
        model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        max_tokens_val = kwargs.pop("max_tokens", int(os.getenv("GROQ_MAX_TOKENS", "2048")))
        primary = ChatGroq(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens_val,
            **kwargs
        )
        backup_key = os.getenv("GROQ_API_KEY_BACKUP")
        backup = None
        if backup_key and backup_key != api_key:
            backup = ChatGroq(
                model_name=model,
                api_key=backup_key,
                temperature=temperature,
                max_tokens=max_tokens_val,
                **kwargs
            )
        return ResilientChatGroq(primary, backup)


    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
            **kwargs
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        model = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=temperature,
            **kwargs
        )

    elif provider == "mock":
        from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
        from langchain_core.messages import AIMessage
        return FakeMessagesListChatModel(responses=[
            AIMessage(content="[MOCK ANSWER] Based on retrieved context [drhp_p1_c1, Page 1].\n\n### Sources Cited\n- drhp_p1_c1 (Page 1, Text)")
        ])

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def generate_answer(
    query: str,
    context: str,
    llm: Optional[Any] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    temperature: float = 0.0
) -> str:
    """
    Generate an answer strictly grounded in the retrieved context with citations.
    """
    if llm is None:
        llm = get_chat_llm(provider=provider, model_name=model_name, temperature=temperature)

    messages = [
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT_TEMPLATE.format(query=query, context=context))
    ]

    try:
        response = llm.invoke(messages)
    except Exception as e:
        if "429" in str(e) or "rate_limit" in str(e).lower():
            backup_key = os.getenv("GROQ_API_KEY_BACKUP")
            if backup_key:
                fallback_llm = get_chat_llm(api_key=backup_key)
                response = fallback_llm.invoke(messages)
            else:
                raise e
        else:
            raise e
    return response.content.strip()


if __name__ == "__main__":
    # Test instantiation and basic invocation
    print("Testing get_chat_llm with Groq...")
    test_llm = get_chat_llm()
    test_context = (
        "--- [SOURCE 1] ---\n"
        "Source ID: drhp_p101_c1\n"
        "Document: DRHP (Maharashtra Oil Extractions Limited)\n"
        "Page: 101\n"
        "Content Type: text\n"
        "Section: SECTION III – GENERAL INFORMATION\n\n"
        "Extracted Text:\n"
        "Registered and Corporate Office: E-140 MIDC, Awdhan, Dhule - 424001, Maharashtra, India.\n"
    )
    test_query = "Where is the registered office of the company?"
    answer = generate_answer(query=test_query, context=test_context, llm=test_llm)
    print("\n--- GENERATED ANSWER ---")
    print(answer)
