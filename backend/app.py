from fastapi import FastAPI

from backend.main import app

appp = FastAPI()

appp.include_router(router=app)
