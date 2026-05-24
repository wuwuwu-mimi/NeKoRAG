"""NeKoRAG REST API"""

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .documents import router as documents_router
from .query import router as query_router

app = FastAPI(title="NeKoRAG API", version="0.1.0")

# 开发环境允许所有来源，生产需收紧
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router, prefix="/v1")
app.include_router(query_router, prefix="/v1")


@app.get("/health")
async def health():
    return {"status": "ok"}
