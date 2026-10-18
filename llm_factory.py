"""Single place to choose the LLM and embedding model.

To switch providers (e.g. to OpenAI GPT-4o-mini), only edit get_llm():

    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
"""
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

DEFAULT_LLM_MODEL = "gemini-3.8-flash"  # change in the sidebar if Google renames/retires it
DEFAULT_GEMINI_EMBED_MODEL = "models/gemini-embedding-001"
HF_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_llm(api_key: str, model: str = DEFAULT_LLM_MODEL):
    # temperature=0 keeps scores consistent between runs
    return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0)


def get_embeddings(provider: str, api_key: str):
    """provider: 'huggingface' (local, free, no API calls) or 'gemini'."""
    if provider == "gemini":
        return GoogleGenerativeAIEmbeddings(
            model=DEFAULT_GEMINI_EMBED_MODEL, google_api_key=api_key
        )
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
