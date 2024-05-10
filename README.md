# Large Model Micro Service

This is a simple microservice that uses a large model to generate answers to questions.

The microservice is built using [FastAPI](https://fastapi.tiangolo.com/zh/).

## Run

Run on Linux Server:

```bash
conda activate <your-virtual-env>
nohup python main.py > log.txt 2>&1 &
```

The Swagger docs of this service could be seen on url: http://host:33445/docs
