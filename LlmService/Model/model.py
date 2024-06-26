import torch
from transformers import (
    pipeline,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from fastapi import APIRouter
from LlmService.util import stream_out

router = APIRouter()


def load_model(
        model: str,
        model_id: str,
        task: str = None,
        device_map: str = "auto",
        torch_dtype: torch.dtype = None,
        verbose: bool = False
):
    stream_out("\n====> [INFO] in function: load_model() <====\n", verbose=verbose)
    if model_id == "TinyLlama-1.1B-Chat-v1.0":
        stream_out(f"\n====> [INFO] load model: {model_id} <====\n", verbose=verbose)
        return pipeline(
            task=task,
            model=model,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
    elif model_id == "Llama-2-7b-chat-hf":
        stream_out(f"\n====> [INFO] load model: {model_id} <====\n", verbose=verbose)
        # nf4" use a symmetric quantization scheme with 4 bits precision
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype
        )
        # load model
        _model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            use_cache=True,
            device_map=device_map
        )
        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return {
            "model": _model,
            "tokenizer": tokenizer
        }
    elif model_id == "Mistral-7B-Instruct-v0.2":
        stream_out(f"\n====> [INFO] load model: {model_id} <====\n", verbose=verbose)
        return {
            "model": AutoModelForCausalLM.from_pretrained(model, device_map="auto"),
            "tokenizer": AutoTokenizer.from_pretrained(model)
        }
    elif model_id == "Meta-Llama-3-8B-Instruct":
        stream_out(f"\n====> [INFO] load model: {model_id} <====\n", verbose=verbose)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model,
            quantization_config=bnb_config,
            use_cache=True,
            device_map=device_map
        )
        tokenizer = AutoTokenizer.from_pretrained(model)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return {
            "model": _model,
            "tokenizer": tokenizer
        }
    else:
        pass


@router.get("/test")
async def test():
    return "hello"
