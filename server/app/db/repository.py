"""Query layer over the Banchan menu collections.

Collection names follow Mongoose's pluralisation of the model names in
`services/menu-service/src/models`: MenuItem -> menuitems, Category ->
categories, Restaurant -> restaurants.

Everything here is read-only by design; the assistant must never mutate live
restaurant data.
"""

from __future__ import annotations

import re
from typing import Any

from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.models import Category, MenuItem, Restaurant

MENU_ITEMS = "menuitems"
CATEGORIES = "categories"
RESTAURANTS = "restaurants"

# Joins the category name onto each item so results can say "Rice Bowls"
# rather than an opaque ObjectId.
_JOIN_CATEGORY: list[dict[str, Any]] = [
    {
        "$lookup": {
            "from": CATEGORIES,
            "localField": "category",
            "foreignField": "_id",
            "as": "_category",
        }
    },
    {"$unwind": {"path": "$_category", "preserveNullAndEmptyArrays": True}},
    {"$addFields": {"category_name": "$_category.name"}},
    {"$project": {"_category": 0}},
]


class MenuRepository:
    """Read-only queries against the menu database."""

    def __init__(self, db: AsyncIOMotorDatabase) -> None:
        self._db = db

    async def _aggregate_items(
        self, stages: list[dict[str, Any]], limit: int
    ) -> list[MenuItem]:
        pipeline = [*stages, *_JOIN_CATEGORY, {"$limit": limit}]
        cursor = self._db[MENU_ITEMS].aggregate(pipeline)
        return [MenuItem.model_validate(doc) async for doc in cursor]

    async def search_by_name(self, term: str, limit: int = 20) -> list[MenuItem]:
        """Case-insensitive substring match on name and description."""
        pattern = re.escape(term.strip())
        return await self._aggregate_items(
            [
                {
                    "$match": {
                        "$or": [
                            {"name": {"$regex": pattern, "$options": "i"}},
                            {"description": {"$regex": pattern, "$options": "i"}},
                        ]
                    }
                },
                {"$sort": {"sortOrder": 1}},
            ],
            limit,
        )

    async def get_by_id(self, item_id: str) -> MenuItem | None:
        from bson import ObjectId
        from bson.errors import InvalidId

        try:
            oid = ObjectId(item_id)
        except (InvalidId, TypeError):
            return None

        items = await self._aggregate_items([{"$match": {"_id": oid}}], limit=1)
        return items[0] if items else None

    async def list_all(self, limit: int = 100) -> list[MenuItem]:
        return await self._aggregate_items([{"$sort": {"sortOrder": 1}}], limit)

    async def list_by_category(
        self, category_name: str, limit: int = 50
    ) -> list[MenuItem]:
        """Items in a category, matched on the category's name."""
        pattern = re.escape(category_name.strip())
        pipeline = [
            *_JOIN_CATEGORY,
            {"$match": {"category_name": {"$regex": pattern, "$options": "i"}}},
            {"$sort": {"sortOrder": 1}},
            {"$limit": limit},
        ]
        cursor = self._db[MENU_ITEMS].aggregate(pipeline)
        return [MenuItem.model_validate(doc) async for doc in cursor]

    async def filter_items(
        self,
        *,
        exclude_allergens: list[str] | None = None,
        vegetarian: bool = False,
        vegan: bool = False,
        gluten_free: bool = False,
        halal: bool = False,
        max_price: float | None = None,
        min_price: float | None = None,
        popular_only: bool = False,
        available_only: bool = True,
        limit: int = 50,
    ) -> list[MenuItem]:
        """Structured filter — the path all allergen questions must take.

        Allergen exclusion is an exact set operation (`$nin`), never a
        similarity match: a near-miss on an allergy question is a safety
        problem, not just a bad answer.
        """
        match: dict[str, Any] = {}

        if exclude_allergens:
            match["allergens"] = {
                "$nin": [a.strip().lower() for a in exclude_allergens]
            }
        if vegetarian:
            match["dietary.isVegetarian"] = True
        if vegan:
            match["dietary.isVegan"] = True
        if gluten_free:
            match["dietary.isGlutenFree"] = True
        if halal:
            match["dietary.isHalal"] = True
        if popular_only:
            match["isPopular"] = True
        if available_only:
            match["isAvailable"] = {"$ne": False}

        price: dict[str, float] = {}
        if max_price is not None:
            price["$lte"] = max_price
        if min_price is not None:
            price["$gte"] = min_price
        if price:
            match["price"] = price

        stages: list[dict[str, Any]] = []
        if match:
            stages.append({"$match": match})
        stages.append({"$sort": {"price": 1}})

        return await self._aggregate_items(stages, limit)

    async def list_categories(self) -> list[Category]:
        cursor = self._db[CATEGORIES].find({"isActive": {"$ne": False}}).sort(
            "sortOrder", 1
        )
        return [Category.model_validate(doc) async for doc in cursor]

    async def get_restaurant(self) -> Restaurant | None:
        doc = await self._db[RESTAURANTS].find_one({"isActive": {"$ne": False}})
        return Restaurant.model_validate(doc) if doc else None

    async def list_restaurants(self) -> list[Restaurant]:
        cursor = self._db[RESTAURANTS].find({})
        return [Restaurant.model_validate(doc) async for doc in cursor]
