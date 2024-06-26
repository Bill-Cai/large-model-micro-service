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


@router.post("/tinyllama/chat_completion_with_template")
def chat_completion_with_template(
        messages: List[Message],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95
):
    """

    :param messages:
    :param max_new_tokens:
    :param do_sample:
    :param temperature:
    :param top_k:
    :param top_p:
    :return:

    Messages should be JSON type, and ordered as "<|system|><|user|><|assistant|><|user|>", like below:

        [
            {
                "role": "system",
                "content": "You are a friendly chatbot, always able to provide friendly and helpful answers."
            },
            {
                "role": "user",
                "content": "What athlete from the 1992 U.S. Olympic 'Dream Team' had a role in the movie 'Space Jam'?"
            },
            {
                "role": "assistant",
                "content": "The athlete from the 1992 U.S. Olympic 'Dream Team' who had a role in the movie 'Space Jam' is Michael Jordan. He played the role of the basketball player \"Lucas North\" in the film, which was released in 1996. The movie was based on a fictional basketball team called the \"Space Jam,\" which was created by Warner Bros. And NBC Universal for the release of the basketball movie \"Space Jam\" in 1996."
            },
            {
                "role": "user",
                "content": "Can you give me more information about that?"
            }
        ]

    If you want to set temperature=0.0, set do_sample=False as well.

    """
    from LlmService.service import pipe_list, verbose
    global pipe_list
    # print(pipe_list)
    stream_out("\n====> [INFO] call function: chat_completion_with_template() <====\n", verbose=verbose)
    stream_out("\n====> [INFO] input messages <====\n", verbose=verbose)
    stream_out(messages, verbose=verbose)
    prompt = pipe_list["TinyLlama-1.1B-Chat-v1.0"].tokenizer.apply_chat_template(messages, tokenize=False,
                                                                                 add_generation_prompt=True)
    outputs = pipe_list["TinyLlama-1.1B-Chat-v1.0"](prompt, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                                    temperature=temperature, top_k=top_k,
                                                    top_p=top_p)
    stream_out("\n====> [INFO] llm outputs <====\n", verbose=verbose)
    stream_out(outputs[0]["generated_text"], verbose=verbose)
    return {
        "message": "success",
        "router": "/chat/tinyllama/chat_completion_with_template",
        "params": {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        },
        "outputs": outputs[0]["generated_text"]
    }


@router.post("/llama2/chat_completion")
def chat_completion_llama2(
        messages: List[Message],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95
):
    from LlmService.service import pipe_list, verbose
    global pipe_list
    stream_out("\n====> [INFO] call function: chat_completion_llama2() <====\n", verbose=verbose)
    stream_out("\n====> [INFO] input messages <====\n", verbose=verbose)
    stream_out(messages, verbose=verbose)
    model = pipe_list["Llama-2-7b-chat-hf"]["model"]
    tokenizer = pipe_list["Llama-2-7b-chat-hf"]["tokenizer"]
    if model.device.type == "cuda":
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            "cuda")
    elif model.device.type == "cpu":
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    else:
        return {
            "message": "failed",
            "info": {
                "error": "[ERROR] Custom error: model.device.type"
            }
        }
    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                   temperature=temperature, top_k=top_k,
                                   top_p=top_p)
    result = tokenizer.batch_decode(generated_ids)
    stream_out("\n====> [INFO] llm outputs <====\n", verbose=verbose)
    stream_out(result, verbose=verbose)
    return {
        "message": "success",
        "router": "/chat/mistral/chat_completion",
        "params": {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        },
        "outputs": result[0]
    }


@router.post("/mistral/chat_completion")
def chat_completion_mistral(
        messages: List[Message],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95
):
    """

    :param messages:
    :param max_new_tokens:
    :param do_sample:
    :param temperature:
    :param top_k:
    :param top_p:
    :return:

    Messages should be JSON type, and ordered as "<|system|><|user|><|assistant|><|user|>", like below:

        [
            {
                "role": "user",
                "content": "What is your favourite condiment?"
            },
            {
                "role": "assistant",
                "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"
            },
            {
                "role": "user",
                "content": "Do you have mayonnaise recipes?"
            }
        ]


    """
    from LlmService.service import pipe_list, verbose
    global pipe_list
    stream_out("\n====> [INFO] call function: chat_completion_mistral() <====\n", verbose=verbose)
    stream_out("\n====> [INFO] input messages <====\n", verbose=verbose)
    stream_out(messages, verbose=verbose)
    model = pipe_list["Mistral-7B-Instruct-v0.2"]["model"]
    tokenizer = pipe_list["Mistral-7B-Instruct-v0.2"]["tokenizer"]
    if model.device.type == "cuda":
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            "cuda")
    elif model.device.type == "cpu":
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    else:
        return {
            "message": "failed",
            "info": {
                "error": "[ERROR] Custom error: model.device.type"
            }
        }
    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                   temperature=temperature, top_k=top_k,
                                   top_p=top_p)
    result = tokenizer.batch_decode(generated_ids)
    stream_out("\n====> [INFO] llm outputs <====\n", verbose=verbose)
    stream_out(result, verbose=verbose)
    return {
        "message": "success",
        "router": "/chat/mistral/chat_completion",
        "params": {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        },
        "outputs": result[0]
    }

@router.post("/llama3/chat_completion")
def chat_completion_llama3(
        messages: List[Message],
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95
):
    from LlmService.service import pipe_list, verbose
    global pipe_list
    stream_out("\n====> [INFO] call function: chat_completion_llama3() <====\n", verbose=verbose)
    stream_out("\n====> [INFO] input messages <====\n", verbose=verbose)
    stream_out(messages, verbose=verbose)
    model = pipe_list["Meta-Llama-3-8B-Instruct"]["model"]
    tokenizer = pipe_list["Meta-Llama-3-8B-Instruct"]["tokenizer"]
    if model.device.type == "cuda":
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(
            "cuda")
    elif model.device.type == "cpu":
        model_inputs = tokenizer.apply_chat_template(messages, return_tensors="pt")
    else:
        return {
            "message": "failed",
            "info": {
                "error": "[ERROR] Custom error: model.device.type"
            }
        }
    generated_ids = model.generate(model_inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                                   temperature=temperature, top_k=top_k,
                                   top_p=top_p)
    result = tokenizer.batch_decode(generated_ids)
    stream_out("\n====> [INFO] llm outputs <====\n", verbose=verbose)
    stream_out(result, verbose=verbose)
    return {
        "message": "success",
        "router": "/chat/mistral/chat_completion",
        "params": {
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p
        },
        "outputs": result[0]
    }