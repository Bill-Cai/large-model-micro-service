import torch
from transformers import pipeline
from fastapi import APIRouter

router = APIRouter()


def load_model():
    print("\n====> [INFO] in function: load_model() <====\n")
    return pipeline(
        "text-generation",
        model="/home/qm/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )


@router.get("/test")
async def test():
    return "hello"
