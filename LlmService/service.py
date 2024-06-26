import torch
from fastapi import FastAPI
from contextlib import asynccontextmanager
from LlmService.Model.model import load_model
from LlmService.router.chat import router as chat_router
from LlmService.util import stream_out

verbose = True
pipe_list = {}
TINYLLAMA_MODEL_PATH = "/data/qm/huggingface/TinyLlama-1.1B-Chat-v1.0"
LLAMA2_MODEL_PATH = "/data/qm/huggingface/Llama-2-7b-chat-hf"
MISTRAL_MODEL_PATH = "/data/qm/huggingface/Mistral-7B-Instruct-v0.2/Mistral-7B-Instruct-v0.2"
LLAMA3_MODEL_PATH = "/data/qm/huggingface/Meta-Llama-3-8B-Instruct"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global verbose, pipe_list, TINYLLAMA_MODEL_PATH, LLAMA2_MODEL_PATH, MISTRAL_MODEL_PATH
    stream_out("\n====> [INFO] lifespan start <====\n", verbose=verbose)
    pipe_list["TinyLlama-1.1B-Chat-v1.0"] = load_model(
        model=TINYLLAMA_MODEL_PATH,
        model_id="TinyLlama-1.1B-Chat-v1.0",
        task="text-generation",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        verbose=verbose
    )  # load large model
    pipe_list["Llama-2-7b-chat-hf"] = load_model(
        model=LLAMA2_MODEL_PATH,
        model_id="Llama-2-7b-chat-hf",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        verbose=verbose
    )
    pipe_list["Mistral-7B-Instruct-v0.2"] = load_model(
        model=MISTRAL_MODEL_PATH,
        model_id="Mistral-7B-Instruct-v0.2",
        device_map="auto",
        verbose=verbose
    )
    pipe_list["Meta-Llama-3-8B-Instruct"] = load_model(
        model=LLAMA3_MODEL_PATH,
        model_id="Meta-Llama-3-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.bfloat16,
        verbose=verbose
    )
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
    stream_out(pipe_list, verbose=True)
    return {"message": "success", "router": "/test"}
