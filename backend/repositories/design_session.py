from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.models import DesignSessionORM


class DesignSessionRepository:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def create(self, session_data: dict) -> DesignSessionORM:
        design_session = DesignSessionORM(**session_data)
        self._db.add(design_session)
        await self._db.flush()
        await self._db.refresh(design_session)
        return design_session

    async def get(self, session_id: UUID) -> DesignSessionORM | None:
        stmt = select(DesignSessionORM).where(DesignSessionORM.id == session_id)
        result = await self._db.execute(stmt)
        return result.scalar_one_or_none()

    async def list_all(
        self, limit: int = 50, offset: int = 0
    ) -> tuple[list[DesignSessionORM], int]:
        count_stmt = select(DesignSessionORM)
        count_result = await self._db.execute(count_stmt)
        total = len(count_result.scalars().all())

        stmt = (
            select(DesignSessionORM)
            .order_by(DesignSessionORM.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        result = await self._db.execute(stmt)
        items = list(result.scalars().all())
        return items, total

    async def update(
        self, session_id: UUID, update_data: dict, expected_version: int
    ) -> DesignSessionORM:
        session = await self.get(session_id)
        if session is None:
            raise ValueError(f"Design session {session_id} not found")

        if session.version != expected_version:
            raise OptimisticLockError(
                f"Version conflict: expected {expected_version}, "
                f"found {session.version}"
            )

        for key, value in update_data.items():
            setattr(session, key, value)
        session.version += 1

        await self._db.flush()
        await self._db.refresh(session)
        return session

    async def delete(self, session_id: UUID) -> bool:
        session = await self.get(session_id)
        if session is None:
            return False
        await self._db.delete(session)
        await self._db.flush()
        return True


class OptimisticLockError(Exception):
    """Raised when an optimistic concurrency check fails."""
