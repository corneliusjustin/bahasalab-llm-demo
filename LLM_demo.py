import streamlit as st
import json
import requests
import re

st.title("Bahasalab LLM Demo")

def reset_state():
    for key in st.session_state.keys():
        if key != 'api_key':
            del st.session_state[key]

with st.sidebar:
    if st.button('New Chat'):
        reset_state()

def generate_mistral(prompt, temperature=0.5, max_tokens=1024, n=1, stop=['```']):
    endpoint = 'https://api.together.xyz/v1/completions'
    res = requests.post(endpoint, json={
        "model": "mistralai/Mistral-7B-Instruct-v0.2",
        "prompt": prompt,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'n': n,
        'stop': stop
    }, headers={
        "Authorization": f"Bearer {st.session_state.api_key}",
    })

    mistral_res = json.loads(res.content)

    return mistral_res

def generate_llama(prompt, temperature=0.7, max_tokens=1024, n=1, stop=['[INST]', '\n\n\n']):
    endpoint = 'https://api.together.xyz/v1/completions'
    res = requests.post(endpoint, json={
        "model": "togethercomputer/Llama-2-7B-32K-Instruct",
        "prompt": prompt,
        'temperature': temperature,
        'max_tokens': max_tokens,
        'n': n,
        'stop': stop
    }, headers={
        "Authorization": f"Bearer {st.session_state.api_key}",
    })

    llama_res = json.loads(res.content)

    return llama_res

def generate_qwen(messages, temperature=0.7, max_tokens=1024, n=1, stop=None):
    endpoint = 'https://api.together.xyz/v1/completions'
    res = requests.post(endpoint, json={
        "model": "Qwen/Qwen1.5-1.8B-Chat",
        "messages": messages,
        "temperature": temperature,
        'max_tokens': max_tokens,
        'n': n,
        'stop': stop
    }, headers={
        "Authorization": f"Bearer {st.session_state.api_key}",
    })

    qwen_res = json.loads(res.content)

    return qwen_res

with st.sidebar:
    selected_model = st.selectbox("Select Model", ['Mistral-7B-Instruct-v0.2', 'Llama-2-7B-32K-Instruct', 'Qwen1.5-1.8B-Chat'])

function_to_call = {'M': generate_mistral, 'L': generate_llama, 'Q': generate_qwen}

SYSTEM_PROMPT_MISTRAL = """Jelaskan hal-hal penting yang dibicarakan pada diskusi \
di atas dalam bentuk paragraf secara singkat, maksimal 200 kata!"""

SYSTEM_PROMPT_LLAMA = """Write a concise summary of a discussion given above inside\
triple backticks in English!"""

SYSTEM_PROMPT_QWEN = """Write a concise summary of a discussion given by user"""

system_prompt = {'M': f'{SYSTEM_PROMPT_MISTRAL}\nRANGKUMAN:', 'L': f'{SYSTEM_PROMPT_LLAMA}\nSUMMARY:'}

# Initialize chat history
if "prompts" not in st.session_state:
    st.session_state.prompts = []

# Display chat prompts from history on app rerun
for prompt in st.session_state.prompts:
    with st.chat_message(prompt['role']):
        st.markdown(prompt['content'])

if 'api_key' in st.session_state:
    with st.sidebar:
        st.success('API key provided!', icon='✅')
else:
    api_key = st.text_input('Enter together.ai API key:', type='password')
    if len(api_key) == 64:
        st.success('Proceed to entering your prompt message!', icon='👉')
        st.session_state.api_key = api_key

if 'api_key' in st.session_state:
    # Accept user input
    if USER_PROMPT := st.chat_input("What do you want to summarize?"):

        USER_PROMPT = re.sub(r"^\d+\n|\d+:\d+:\d+,\d+ --> \d+:\d+:\d+,\d+\n|\[.+?\]\n|\n\n+", '', USER_PROMPT)
        USER_PROMPT = re.sub(r'(\d+)(?=\n)', '', USER_PROMPT)
        USER_PROMPT = re.sub(r'\[[a-zA-Z\s,]+\]', '', USER_PROMPT)
        USER_PROMPT = re.sub(r'\n', '', USER_PROMPT)

        with st.chat_message("assistant"):
            with st.spinner(f'*Generating summary...*') as status:
                if selected_model[0] != 'Q':
                    PROMPT = f"```{USER_PROMPT}```\n" + system_prompt[selected_model[0]]
                    res = function_to_call[selected_model[0]](PROMPT)
                else:
                    messages = [
                        {
                            'content': USER_PROMPT,
                            'role': 'user'
                        },
                        {
                            'content': SYSTEM_PROMPT_QWEN,
                            'role': 'system'
                        },
                    ]

                    res = function_to_call[selected_model[0]](messages)

            summary = res["choices"][0]["text"]
            summary_json = str({"summary": summary})
            st.markdown(summary_json)

            st.session_state.prompts.append({'role': 'assistant', 'content': summary_json})