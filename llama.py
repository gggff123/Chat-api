from llama_cpp import Llama

_llm = None


def _is_streamlit():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except ImportError:
        return False


def _load_model():
    return Llama(
        model_path="smollm2-360m-instruct-q8_0.gguf",
        n_ctx=256,
        n_threads=2,
        n_batch=512,
        use_mlock=True,
        verbose=False
    )


def _get_llm():
    global _llm

    if _is_streamlit():
        import streamlit as st
        if "llm_model" not in st.session_state:
            st.session_state.llm_model = _load_model()
        return st.session_state.llm_model

    if _llm is None:
        _llm = _load_model()

    return _llm


def addprompt(
    user_message: str,
    system_message: str = """please provide small concise responses""",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 40,
    repeat_penalty: float = 1.3,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0
) -> str:

    llm = _get_llm()

    try:
        output = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repeat_penalty=repeat_penalty,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty
        )

        return output["choices"][0]["message"]["content"]

    except Exception as e:
        return f"Error generating story: {e}"
