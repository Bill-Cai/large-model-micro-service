from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from LlmService.util import stream_out

router = APIRouter()


class Message(BaseModel):
    role: str
    content: str


@router.get("/test")
def test():
    return {"message": "success", "router": "/chat/test"}


@router.post("/chat_completion_with_template")
def chat_completion_with_template(
        messages: List[Message],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95
):
    from LlmService.service import pipe, verbose
    global pipe
    # print(pipe)
    stream_out("\n====> [INFO] input messages <====\n", verbose=verbose)
    print(messages)
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k,
                   top_p=top_p)
    stream_out("\n====> [INFO] llm outputs <====\n", verbose=verbose)
    stream_out(outputs[0]["generated_text"], verbose=verbose)
    return {"message": "success", "router": "/chat/single_query", "outputs": outputs[0]["generated_text"]}
