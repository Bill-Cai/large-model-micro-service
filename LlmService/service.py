import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from LlmService.Model.model import load_model
from LlmService.router.chat import router as chat_router
from LlmService.util import stream_out

pipe = None
verbose = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    stream_out("\n====> [INFO] lifespan start <====\n", verbose=verbose)
    global pipe
    pipe = load_model(
        task="text-generation",
        model="/home/qm/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        verbose=verbose
    )  # load large model
    stream_out("\n====> [INFO] model loaded <====\n", verbose=verbose)
    print()
    yield
    stream_out("\n====> [INFO] lifespan shutdown <====\n", verbose=verbose)


app = FastAPI(lifespan=lifespan)
# app.mount("/chat", chat_router)
app.include_router(
    chat_router,
    prefix="/chat"
)


@app.get("/test")
async def test():
    stream_out(pipe, verbose=True)
    return {"message": "success", "router": "/test"}
