from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.ollama.ollama import get_categories

app = FastAPI()

class Item(BaseModel):
    content: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/get-category/")
async def get_category(item: Item):
    try:
        response = get_categories(item.content)

        return response
    except Exception:
        return {"error": "Too many requests in a given period"}