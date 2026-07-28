"""Structured menu tools backed by MongoDB.

These are deliberately several small, single-purpose tools rather than one
tool with an `operation` argument. The previous design documented JSON
operations in the system prompt that the tool could not actually parse; typed
argument schemas make the contract explicit and let the framework reject bad
calls before they reach the database.

Anything price-, stock- or allergen-related belongs here rather than in
semantic search, because these answers must be exact.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from app.db.models import Allergen, MenuItem
from app.db.mongo import get_database
from app.db.repository import MenuRepository

_ASYNC_ONLY = "Menu tools are async-only; invoke the agent with ainvoke/astream."

VALID_ALLERGENS = sorted(a.value for a in Allergen)


def _repo() -> MenuRepository:
    return MenuRepository(get_database())


def _render(items: list[MenuItem], header: str, empty: str) -> str:
    if not items:
        return empty
    lines = [f"{header} ({len(items)} found):", ""]
    lines.extend(item.to_summary() for item in items)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# search_menu
# --------------------------------------------------------------------------


class SearchMenuInput(BaseModel):
    query: str = Field(
        description=(
            "A dish name or keyword to look for, e.g. 'bibimbap', 'chicken', "
            "'tofu'. Matches against both dish names and descriptions."
        )
    )


class SearchMenuTool(BaseTool):
    name: str = "search_menu"
    description: str = (
        "Look up specific dishes by name or keyword and return their exact price, "
        "availability, dietary labels and declared allergens. "
        "Use this whenever the user names a dish or asks what something costs. "
        "This returns authoritative data from the menu database."
    )
    args_schema: type[BaseModel] = SearchMenuInput

    def _run(self, query: str) -> str:
        raise NotImplementedError(_ASYNC_ONLY)

    async def _arun(self, query: str) -> str:
        items = await _repo().search_by_name(query)
        return _render(
            items,
            header=f"Menu matches for '{query}'",
            empty=(
                f"No dish matching '{query}'. Try a broader term, or use "
                f"list_menu_categories to see what we serve."
            ),
        )


# --------------------------------------------------------------------------
# filter_menu
# --------------------------------------------------------------------------


class FilterMenuInput(BaseModel):
    exclude_allergens: list[str] | None = Field(
        default=None,
        description=(
            "Allergens the customer must avoid. Only these values are valid: "
            + ", ".join(VALID_ALLERGENS)
            + ". Dishes declaring any of them are excluded."
        ),
    )
    vegetarian: bool = Field(default=False, description="Only vegetarian dishes.")
    vegan: bool = Field(default=False, description="Only vegan dishes.")
    gluten_free: bool = Field(default=False, description="Only gluten-free dishes.")
    halal: bool = Field(default=False, description="Only halal dishes.")
    max_price: float | None = Field(default=None, description="Maximum price in GBP.")
    min_price: float | None = Field(default=None, description="Minimum price in GBP.")
    popular_only: bool = Field(default=False, description="Only popular dishes.")


class FilterMenuTool(BaseTool):
    name: str = "filter_menu"
    description: str = (
        "Filter the whole menu by allergens to avoid, dietary requirement, or price. "
        "ALWAYS use this tool for allergy and dietary questions — it filters on exact "
        "structured allergen data. Never answer an allergy question from a dish "
        "description or from search_knowledge. "
        "Also use it for price questions like 'what is under £6'."
    )
    args_schema: type[BaseModel] = FilterMenuInput

    def _run(self, **kwargs: object) -> str:
        raise NotImplementedError(_ASYNC_ONLY)

    async def _arun(
        self,
        exclude_allergens: list[str] | None = None,
        vegetarian: bool = False,
        vegan: bool = False,
        gluten_free: bool = False,
        halal: bool = False,
        max_price: float | None = None,
        min_price: float | None = None,
        popular_only: bool = False,
    ) -> str:
        if exclude_allergens:
            normalised = [a.strip().lower() for a in exclude_allergens]
            unknown = [a for a in normalised if a not in VALID_ALLERGENS]
            if unknown:
                # Refuse rather than silently ignoring: quietly dropping an
                # unrecognised allergen would return dishes containing it.
                return (
                    f"Cannot filter on unrecognised allergen(s): {', '.join(unknown)}. "
                    f"Valid allergens are: {', '.join(VALID_ALLERGENS)}. "
                    f"Ask the customer to clarify, and advise them to phone the "
                    f"restaurant if their allergy is not on the list."
                )
            exclude_allergens = normalised

        items = await _repo().filter_items(
            exclude_allergens=exclude_allergens,
            vegetarian=vegetarian,
            vegan=vegan,
            gluten_free=gluten_free,
            halal=halal,
            max_price=max_price,
            min_price=min_price,
            popular_only=popular_only,
        )

        criteria = []
        if exclude_allergens:
            criteria.append(f"without {', '.join(exclude_allergens)}")
        for flag, label in (
            (vegetarian, "vegetarian"),
            (vegan, "vegan"),
            (gluten_free, "gluten-free"),
            (halal, "halal"),
        ):
            if flag:
                criteria.append(label)
        if max_price is not None:
            criteria.append(f"under £{max_price:.2f}")
        if min_price is not None:
            criteria.append(f"over £{min_price:.2f}")
        if popular_only:
            criteria.append("popular")

        described = " and ".join(criteria) if criteria else "all dishes"
        result = _render(
            items,
            header=f"Dishes matching: {described}",
            empty=f"No dishes match: {described}.",
        )

        if exclude_allergens and items:
            result += (
                "\n\nNote: our kitchen is shared, so traces cannot be guaranteed "
                "absent. Advise customers with severe allergies to phone the "
                "restaurant."
            )
        return result


# --------------------------------------------------------------------------
# list_menu_categories
# --------------------------------------------------------------------------


class ListCategoriesInput(BaseModel):
    category: str | None = Field(
        default=None,
        description=(
            "Optional category name to list the dishes of, e.g. 'Rice Bowls'. "
            "Omit to list the category names themselves."
        ),
    )


class ListCategoriesTool(BaseTool):
    name: str = "list_menu_categories"
    description: str = (
        "List the menu categories, or list every dish inside one category. "
        "Use this to orient a customer who does not know what we serve, or when "
        "asked for 'the whole menu'."
    )
    args_schema: type[BaseModel] = ListCategoriesInput

    def _run(self, category: str | None = None) -> str:
        raise NotImplementedError(_ASYNC_ONLY)

    async def _arun(self, category: str | None = None) -> str:
        repo = _repo()

        if category:
            items = await repo.list_by_category(category)
            return _render(
                items,
                header=f"Dishes in '{category}'",
                empty=f"No category named '{category}'.",
            )

        categories = await repo.list_categories()
        if not categories:
            return "No menu categories are configured."
        listed = "\n".join(f"• {c.name}" for c in categories)
        return f"Menu categories:\n{listed}\n\nAsk about any one to see its dishes."


# --------------------------------------------------------------------------
# get_restaurant_info
# --------------------------------------------------------------------------


class RestaurantInfoInput(BaseModel):
    """No arguments — returns the single active location."""


class RestaurantInfoTool(BaseTool):
    name: str = "get_restaurant_info"
    description: str = (
        "Get the restaurant's address, phone number, opening hours, delivery radius, "
        "minimum order value and delivery fee. Use for 'where are you', 'when do you "
        "open', 'do you deliver to me' and 'what is the minimum order'."
    )
    args_schema: type[BaseModel] = RestaurantInfoInput

    def _run(self) -> str:
        raise NotImplementedError(_ASYNC_ONLY)

    async def _arun(self) -> str:
        restaurant = await _repo().get_restaurant()
        if restaurant is None:
            return "No restaurant record is configured."

        lines = [
            restaurant.name,
            f"Address: {restaurant.address.one_line()}",
            f"Phone: {restaurant.phone}",
            "",
            "Opening hours:",
        ]
        lines.extend(
            f"  {day.capitalize():<10} {hours.open} - {hours.close}"
            for day, hours in restaurant.opening_hours.items()
        )
        lines.extend(
            [
                "",
                # :g so a whole-number radius reads "5 km", not "5.0 km".
                f"Delivery radius: {restaurant.delivery_radius:g} km",
                f"Minimum order:   £{restaurant.minimum_order:.2f}",
                f"Delivery fee:    £{restaurant.delivery_fee:.2f}",
            ]
        )
        return "\n".join(lines)
