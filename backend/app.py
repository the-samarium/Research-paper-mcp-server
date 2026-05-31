from fastapi import FastAPI

from backend.router.main import app

appp = FastAPI()

appp.include_router(router=app)
