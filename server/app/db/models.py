"""Pydantic mirrors of the Mongoose schemas in the Banchan menu service.

Field aliases map the camelCase stored in MongoDB onto snake_case Python
attributes, so `populate_by_name=True` is set throughout and documents are
parsed with `model_validate(doc)`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Allergen(StrEnum):
    """The 14 allergens UK food businesses must declare.

    Mirrors the enum on `MenuItem.allergens`. Kept as a closed set so the
    menu tool can reject typos instead of silently returning "no allergens".
    """

    CELERY = "celery"
    GLUTEN = "gluten"
    CRUSTACEANS = "crustaceans"
    EGGS = "eggs"
    FISH = "fish"
    LUPIN = "lupin"
    MILK = "milk"
    MOLLUSCS = "molluscs"
    MUSTARD = "mustard"
    NUTS = "nuts"
    PEANUTS = "peanuts"
    SESAME = "sesame"
    SOYBEANS = "soybeans"
    SULPHITES = "sulphites"


def _stringify_object_id(value: Any) -> Any:
    """Coerce a BSON ObjectId to str, leaving anything else alone."""
    if value is None:
        return None
    return str(value)


class Dietary(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    is_vegetarian: bool = Field(default=False, alias="isVegetarian")
    is_vegan: bool = Field(default=False, alias="isVegan")
    is_gluten_free: bool = Field(default=False, alias="isGlutenFree")
    is_halal: bool = Field(default=False, alias="isHalal")

    def labels(self) -> list[str]:
        """Human-readable dietary tags, for prompts and embedded passages."""
        mapping = {
            "vegetarian": self.is_vegetarian,
            "vegan": self.is_vegan,
            "gluten-free": self.is_gluten_free,
            "halal": self.is_halal,
        }
        return [label for label, present in mapping.items() if present]


class Category(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    name: str
    slug: str | None = None
    description: str = ""
    sort_order: int = Field(default=0, alias="sortOrder")
    is_active: bool = Field(default=True, alias="isActive")

    _normalise_id = field_validator("id", mode="before")(_stringify_object_id)


class MenuItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    name: str
    description: str = ""
    price: float
    category_id: str | None = Field(default=None, alias="category")
    # Populated by the $lookup in MenuRepository, not stored on the document.
    category_name: str | None = None
    image: str | None = None
    dietary: Dietary = Field(default_factory=Dietary)
    allergens: list[str] = Field(default_factory=list)
    calories: int | None = None
    preparation_time: int | None = Field(default=None, alias="preparationTime")
    is_available: bool = Field(default=True, alias="isAvailable")
    is_popular: bool = Field(default=False, alias="isPopular")

    _normalise_id = field_validator("id", "category_id", mode="before")(
        _stringify_object_id
    )

    def to_passage(self) -> str:
        """Render the item as prose for embedding.

        Vector search works on natural language, so the structured fields are
        flattened into a sentence-shaped passage rather than embedded as JSON.
        """
        parts = [f"{self.name} — {self.description}"]
        if self.category_name:
            parts.append(f"Category: {self.category_name}.")
        parts.append(f"Price: £{self.price:.2f}.")

        if labels := self.dietary.labels():
            parts.append(f"Suitable for: {', '.join(labels)}.")
        if self.allergens:
            parts.append(f"Contains allergens: {', '.join(sorted(self.allergens))}.")
        else:
            parts.append("No declared allergens.")
        if self.calories:
            parts.append(f"Approximately {self.calories} kcal.")
        if self.preparation_time:
            parts.append(f"Prepared in about {self.preparation_time} minutes.")
        if self.is_popular:
            parts.append("This is one of our most popular dishes.")
        if not self.is_available:
            parts.append("Currently unavailable.")

        return " ".join(parts)

    def to_summary(self) -> str:
        """Compact one-item rendering for structured tool output."""
        bits = [f"{self.name} — £{self.price:.2f}"]
        if self.category_name:
            bits.append(f"({self.category_name})")
        if not self.is_available:
            bits.append("[currently unavailable]")

        lines = [" ".join(bits)]
        if self.description:
            lines.append(f"    {self.description}")

        detail = []
        if labels := self.dietary.labels():
            detail.append(f"Dietary: {', '.join(labels)}")
        detail.append(
            f"Allergens: {', '.join(sorted(self.allergens))}"
            if self.allergens
            else "Allergens: none declared"
        )
        lines.append(f"    {' | '.join(detail)}")
        return "\n".join(lines)


class Address(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    line1: str = ""
    line2: str | None = None
    city: str = "London"
    postcode: str = ""

    def one_line(self) -> str:
        parts = [self.line1, self.line2, self.city, self.postcode]
        return ", ".join(p for p in parts if p)


class DayHours(BaseModel):
    open: str = ""
    close: str = ""


class Restaurant(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    name: str
    address: Address = Field(default_factory=Address)
    phone: str = ""
    email: str | None = None
    opening_hours: dict[str, DayHours] = Field(
        default_factory=dict, alias="openingHours"
    )
    delivery_radius: float = Field(default=5, alias="deliveryRadius")
    minimum_order: float = Field(default=10, alias="minimumOrder")
    delivery_fee: float = Field(default=2.99, alias="deliveryFee")
    is_active: bool = Field(default=True, alias="isActive")

    _normalise_id = field_validator("id", mode="before")(_stringify_object_id)

    def to_passage(self) -> str:
        """Render location and service details as prose for embedding."""
        parts = [
            f"{self.name} is located at {self.address.one_line()}.",
            f"Phone: {self.phone}.",
        ]
        if self.opening_hours:
            hours = ", ".join(
                f"{day.capitalize()} {h.open}-{h.close}"
                for day, h in self.opening_hours.items()
            )
            parts.append(f"Opening hours: {hours}.")
        parts.append(
            f"Delivery radius {self.delivery_radius} km, "
            f"minimum order £{self.minimum_order:.2f}, "
            f"delivery fee £{self.delivery_fee:.2f}."
        )
        return " ".join(parts)
