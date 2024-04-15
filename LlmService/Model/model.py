import torch
from transformers import pipeline
from fastapi import APIRouter
from LlmService.util import stream_out

router = APIRouter()


def load_model(
        task: str,
        model: str,
        torch_dtype: torch.dtype,
        device_map: str,
        verbose: bool = False
):
    stream_out("\n====> [INFO] in function: load_model() <====\n", verbose=verbose)
    return pipeline(
        task=task,
        model=model,
        torch_dtype=torch_dtype,
        device_map=device_map
    )


@router.get("/test")
async def test():
    return "hello"
