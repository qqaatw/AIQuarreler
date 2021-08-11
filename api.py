import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from torch_generate import PredefinedInference


# Logging
if not os.path.isdir('./api_log'):
    os.mkdir('./api_log')

formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')

file_handler = logging.FileHandler('./api_log/access.log', encoding='utf8')
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger = logging.getLogger('APILogger')
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.setLevel(logging.INFO)

app = FastAPI()

inference = PredefinedInference('last.ckpt')

class APISchema(BaseModel):
    InputText: str

@app.get("/", response_class=FileResponse)
def root(request: Request):
    return FileResponse('html/index.html')

@app.post("/api")
async def request_prediction(data: APISchema):
    try:
        conf_no=4
        response, conf_dict = inference.generate(data.InputText, conf_no=conf_no)
        logger.info(f"Conf: {conf_dict}, Input: {data.InputText}, Output:{response}")
        return {"status": "OK", "response": response}
    except:
        return {"status": "Error", "response": ""}

    