from fastapi import FastAPI
from contextlib import asynccontextmanager
from LlmService.Model.model import load_model
from LlmService.router.chat import router as chat_router

pipe = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n====> [INFO] lifespan start <====\n")
    global pipe
    pipe = load_model()  # load large model
    print("\n====> [INFO] model loaded <====\n")
    yield
    print("\n====> [INFO] lifespan shutdown <====\n")


app = FastAPI(lifespan=lifespan)
app.mount("/chat", chat_router)


@app.get("/test")
async def test():
    print(pipe)
    return {"message": "success", "router": "/test"}
