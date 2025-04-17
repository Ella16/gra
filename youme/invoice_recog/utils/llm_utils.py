from typing import Any, Dict, List, Tuple


# run machine
def run_machine(
    client, model: str, messages=List[Dict[str, Any]], response_fmt: str = "text"
) -> Any:
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={"type": response_fmt},
        temperature=0.7,
    )
    return response.choices[0].message.content


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
