from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config import Settings, get_settings
from backend.database import get_db
from backend.repositories.design_session import DesignSessionRepository


@lru_cache
def get_cached_settings() -> Settings:
    return get_settings()


SettingsDep = Annotated[Settings, Depends(get_cached_settings)]


async def get_db_session() -> AsyncSession:  # type: ignore[misc]
    async for session in get_db():
        yield session  # type: ignore[misc]


DbSessionDep = Annotated[AsyncSession, Depends(get_db_session)]


def get_design_repo(db: DbSessionDep) -> DesignSessionRepository:
    return DesignSessionRepository(db)


DesignRepoDep = Annotated[DesignSessionRepository, Depends(get_design_repo)]
