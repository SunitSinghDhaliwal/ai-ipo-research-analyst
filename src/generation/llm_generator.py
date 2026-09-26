import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dotenv import load_dotenv

# Load environment variables (.env file)
env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

SYSTEM_PROMPT = """You are a rigorous, highly disciplined IPO Research Analyst evaluating the Draft Red Herring Prospectus (DRHP) for Maharashtra Oil Extractions Limited.

Your objective is to provide precise, factual, and strictly grounded answers to research questions based ONLY on the retrieved DRHP context provided.

CRITICAL INSTRUCTIONS:
1. STRICT FACTUAL GROUNDING:
   - Answer the question solely and exclusively using the facts, numbers, names, and statements contained in the provided context.
   - Do NOT extrapolate, speculate, interpolate, or introduce outside knowledge not present in the context.

2. INSUFFICIENT EVIDENCE & HALLUCINATION PREVENTION:
   - If the provided context does not contain sufficient direct evidence or answers only a portion of the question, clearly state:
     "Insufficient evidence in the provided DRHP context to answer this question."
   - Specifically state what required information is absent from the retrieved excerpts.
   - Never guess or manufacture financial numbers, dates, legal case outcomes, or management details.

3. MANDATORY INLINE CITATIONS:
   - Every factual statement, metric, table reference, or claim MUST be immediately followed by an inline citation specifying the Source ID and Page Number in brackets.
   - Format: `[<Source ID>, Page <Page Number>]`
   - Examples: `[drhp_p30_c1, Page 30]`, `[moe_p444_t1, Page 444]`.

4. SOURCES SUMMARY TABLE:
   - Conclude your response with a structured markdown section titled `### Sources Cited` listing every source ID cited in your response, its page number, and its content type (Text/Table).
"""

USER_PROMPT_TEMPLATE = """RESEARCH QUERY:
{query}

RETRIEVED DRHP CONTEXT:
{context}

Please provide your rigorous, cited analysis following the instructions above.
"""


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
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing from environment or .env file.")
        # Default to openai/gpt-oss-120b or GROQ_MODEL
        model = model_name or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        max_tokens_val = kwargs.pop("max_tokens", int(os.getenv("GROQ_MAX_TOKENS", "800")))
        return ChatGroq(
            model_name=model,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens_val,
            **kwargs
        )

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
            fallback_llm = get_chat_llm(model_name="openai/gpt-oss-20b")
            response = fallback_llm.invoke(messages)
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
