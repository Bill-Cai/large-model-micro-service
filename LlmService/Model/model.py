import torch
from transformers import pipeline
from fastapi import APIRouter

router = APIRouter()


def load_model(
        task: str,
        model: str,
        torch_dtype: torch.dtype,
        device_map: str
):
    print("\n====> [INFO] in function: load_model() <====\n")
    return pipeline(
        task=task,
        model=model,
        torch_dtype=torch_dtype,
        device_map=device_map
    )


@router.get("/test")
async def test():
    return "hello"
