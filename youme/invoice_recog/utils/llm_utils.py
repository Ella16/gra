from typing import Any, Dict, List, Tuple

# run machine
def run_machine(
    client, model: str, messages=List[Dict[str, Any]], response_fmt: str = "text", retry=0
) -> Any:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": response_fmt},
            temperature=0.7,
        )
        return response.choices[0].message.content
    
    except Exception as e: # except openai.RateLimitError as e: catch가 안됨? 
        print(e)
        RETRY_LIMIT = 3
        if retry < RETRY_LIMIT:
            sleep = 30
            print(f"OPENAI RATELIMIT REACHED 200,000 TPM for gpt-4o-mini: gonna wait {sleep} secs and retry..{retry+1}/{RETRY_LIMIT}")
            import time
            
            for i in range(sleep,0,-1):
                print(f"{i} ", end="\r", flush=True)
                time.sleep(1)
            return run_machine(client, model, messages=messages, response_fmt=response_fmt, retry=retry+1)
        else:
            print(f"OPENAI RATELIMIT REACHED 200,000 TPM for gpt-4o-mini: retry limit reached {retry}/{RETRY_LIMIT}")
            raise e
    

def build_system_message(prompt: str) -> Dict[str, Any]:
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_human_message(
    prompt: List[str] | str = [], prompt_type: List[str] | str = []
) -> Dict[str, Any]:
    assert type(prompt) == type(
        prompt_type
    ), "Prompt and type should have the same type"

    if type(prompt) == str:
        prompt = [prompt]
        prompt_type = [prompt_type]

    assert len(prompt) == len(
        prompt_type
    ), "Prompt and type should have the same length"

    content = []
    for i, p in enumerate(prompt):
        if prompt_type[i] == "text":
            content.append({"type": "text", "text": p})
        elif prompt_type[i] == "image_url":
            content.append({"type": "image_url", "image_url": {"url": p}})
        else:
            raise ValueError("Invalid type")

    return [{"role": "user", "content": content}]
