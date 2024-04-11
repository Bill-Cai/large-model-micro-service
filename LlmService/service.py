from fastapi import FastAPI
from contextlib import asynccontextmanager
from LlmService.Model.model import load_model, pipe


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n====> [INFO] lifespan start <====\n")
    # load_model()
    # print("\n====> [INFO] model loaded <====\n")
    yield
    print("\n====> [INFO] lifespan shutdown <====\n")


app = FastAPI(lifespan=lifespan)


@app.get("/test")
async def test():
    return "hello"
