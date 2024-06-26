# Large Model Micro Service

This is a simple microservice that uses a large model to generate answers to questions.

The microservice is built using [FastAPI](https://fastapi.tiangolo.com/zh/).

## Quick Note

Run the service

- Python Env: `py311-llama2-qm`
- Project Path: `/tmp/pycharm_project_61`
- Log Path: `/home/qm/log-qm/large-model-micro-service.log`

```bash
# Run the service
conda activate py311-llama2-qm
nohup python /tmp/pycharm_project_61/main.py >/home/qm/log-qm/large-model-micro-service.log 2>&1 &

# Check the log
tail -f /home/qm/log-qm/large-model-micro-service.log
```

The Swagger docs of this service could be seen on url: http://host:33445/docs
