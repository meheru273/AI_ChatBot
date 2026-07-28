"""Tests for the agent's tools.

The menu tools are the safety-critical ones: an allergen filter that silently
does the wrong thing is worse than one that fails loudly.
"""

from __future__ import annotations

import pytest

from app.db.models import Dietary, MenuItem, Restaurant
from app.tools.menu import (
    VALID_ALLERGENS,
    FilterMenuTool,
    ListCategoriesTool,
    RestaurantInfoTool,
    SearchMenuTool,
)


def make_item(name: str, price: float, allergens: list[str] | None = None, **kwargs):
    return MenuItem(
        _id=f"id-{name}",
        name=name,
        description=kwargs.pop("description", f"{name} description"),
        price=price,
        category_name=kwargs.pop("category_name", "Sides"),
        allergens=allergens or [],
        dietary=kwargs.pop("dietary", Dietary()),
        **kwargs,
    )


class FakeRepo:
    """Records the arguments it was called with, so routing can be asserted."""

    def __init__(self, items=None, categories=None, restaurant=None):
        self.items = items if items is not None else []
        self.categories = categories or []
        self.restaurant = restaurant
        self.filter_kwargs = None
        self.search_term = None
        self.category_arg = None

    async def search_by_name(self, term, limit=20):
        self.search_term = term
        return self.items

    async def filter_items(self, **kwargs):
        self.filter_kwargs = kwargs
        return self.items

    async def list_by_category(self, category, limit=50):
        self.category_arg = category
        return self.items

    async def list_categories(self):
        return self.categories

    async def get_restaurant(self):
        return self.restaurant


@pytest.fixture
def patch_repo(monkeypatch):
    def _install(repo: FakeRepo) -> FakeRepo:
        monkeypatch.setattr("app.tools.menu._repo", lambda: repo)
        return repo

    return _install


# --------------------------------------------------------------------------
# search_menu
# --------------------------------------------------------------------------


async def test_search_menu_reports_price_and_allergens(patch_repo):
    patch_repo(FakeRepo(items=[make_item("Bibimbap", 9.5, ["eggs", "sesame"])]))

    result = await SearchMenuTool().arun({"query": "bibimbap"})

    assert "Bibimbap" in result
    assert "£9.50" in result
    assert "eggs" in result and "sesame" in result


async def test_search_menu_with_no_match_suggests_a_next_step(patch_repo):
    patch_repo(FakeRepo(items=[]))

    result = await SearchMenuTool().arun({"query": "sushi"})

    assert "No dish matching" in result
    assert "list_menu_categories" in result


async def test_search_menu_flags_unavailable_items(patch_repo):
    patch_repo(FakeRepo(items=[make_item("Mandu", 4.5, isAvailable=False)]))

    assert "currently unavailable" in await SearchMenuTool().arun({"query": "mandu"})


# --------------------------------------------------------------------------
# filter_menu — allergen safety
# --------------------------------------------------------------------------


async def test_filter_menu_passes_allergen_exclusions_to_the_repository(patch_repo):
    repo = patch_repo(FakeRepo(items=[make_item("Kimchi", 2.5)]))

    await FilterMenuTool().arun({"exclude_allergens": ["sesame", "gluten"]})

    assert repo.filter_kwargs["exclude_allergens"] == ["sesame", "gluten"]


async def test_filter_menu_normalises_allergen_casing(patch_repo):
    repo = patch_repo(FakeRepo(items=[]))

    await FilterMenuTool().arun({"exclude_allergens": ["  SESAME ", "Gluten"]})

    assert repo.filter_kwargs["exclude_allergens"] == ["sesame", "gluten"]


async def test_filter_menu_refuses_an_unrecognised_allergen(patch_repo):
    """Silently dropping an unknown allergen would return dishes containing it."""
    repo = patch_repo(FakeRepo(items=[make_item("Kimchi", 2.5)]))

    result = await FilterMenuTool().arun({"exclude_allergens": ["shellfish"]})

    assert "unrecognised allergen" in result
    assert "shellfish" in result
    assert repo.filter_kwargs is None, "must not query with an unknown allergen"


async def test_filter_menu_lists_the_valid_allergens_when_rejecting(patch_repo):
    patch_repo(FakeRepo(items=[]))

    result = await FilterMenuTool().arun({"exclude_allergens": ["dairy"]})

    assert "milk" in result  # the correct term for the one they meant
    assert "phone the restaurant" in result


async def test_filter_menu_warns_about_cross_contamination(patch_repo):
    patch_repo(FakeRepo(items=[make_item("Kimchi", 2.5)]))

    result = await FilterMenuTool().arun({"exclude_allergens": ["nuts"]})

    assert "traces" in result
    assert "phone" in result


async def test_cross_contamination_warning_is_omitted_when_nothing_matched(patch_repo):
    patch_repo(FakeRepo(items=[]))

    result = await FilterMenuTool().arun({"exclude_allergens": ["nuts"]})

    assert "No dishes match" in result
    assert "traces" not in result


async def test_valid_allergens_match_the_uk_regulated_fourteen():
    assert len(VALID_ALLERGENS) == 14
    for expected in ("gluten", "peanuts", "sesame", "sulphites", "molluscs"):
        assert expected in VALID_ALLERGENS


# --------------------------------------------------------------------------
# filter_menu — dietary and price
# --------------------------------------------------------------------------


async def test_filter_menu_forwards_dietary_flags(patch_repo):
    repo = patch_repo(FakeRepo(items=[]))

    await FilterMenuTool().arun({"vegan": True, "halal": False})

    assert repo.filter_kwargs["vegan"] is True
    assert repo.filter_kwargs["halal"] is False


async def test_filter_menu_forwards_a_price_ceiling(patch_repo):
    repo = patch_repo(FakeRepo(items=[make_item("Kimchi", 2.5)]))

    result = await FilterMenuTool().arun({"max_price": 6.0})

    assert repo.filter_kwargs["max_price"] == 6.0
    assert "under £6.00" in result


async def test_filter_menu_describes_combined_criteria(patch_repo):
    patch_repo(FakeRepo(items=[]))

    result = await FilterMenuTool().arun({"vegan": True, "max_price": 5.0})

    assert "vegan" in result
    assert "under £5.00" in result


# --------------------------------------------------------------------------
# list_menu_categories and get_restaurant_info
# --------------------------------------------------------------------------


async def test_list_categories_without_argument_lists_category_names(patch_repo):
    from app.db.models import Category

    patch_repo(
        FakeRepo(
            categories=[
                Category(_id="1", name="Rice Bowls"),
                Category(_id="2", name="Sides"),
            ]
        )
    )

    result = await ListCategoriesTool().arun({})

    assert "Rice Bowls" in result and "Sides" in result


async def test_list_categories_with_argument_lists_the_dishes(patch_repo):
    repo = patch_repo(FakeRepo(items=[make_item("Gimbap", 4.5)]))

    result = await ListCategoriesTool().arun({"category": "Sides"})

    assert repo.category_arg == "Sides"
    assert "Gimbap" in result


async def test_restaurant_info_reports_hours_and_delivery_terms(patch_repo):
    patch_repo(
        FakeRepo(
            restaurant=Restaurant(
                _id="r1",
                name="Banchan West Drayton",
                address={"line1": "121 Faling Lane", "postcode": "UB7 8AG"},
                phone="+44 20 7123 4567",
                openingHours={"monday": {"open": "11:00", "close": "22:00"}},
                minimumOrder=10,
                deliveryFee=2.99,
                deliveryRadius=5,
            )
        )
    )

    result = await RestaurantInfoTool().arun({})

    assert "UB7 8AG" in result
    assert "11:00 - 22:00" in result
    assert "£10.00" in result
    assert "£2.99" in result
    assert "5 km" in result


async def test_restaurant_info_handles_a_missing_record(patch_repo):
    patch_repo(FakeRepo(restaurant=None))
    assert "No restaurant record" in await RestaurantInfoTool().arun({})


# --------------------------------------------------------------------------
# knowledge tool
# --------------------------------------------------------------------------


async def test_knowledge_tool_returns_passages_with_sources(monkeypatch, vector_store):
    from app.tools.knowledge import SearchKnowledgeTool

    monkeypatch.setattr("app.tools.knowledge.get_vector_store", lambda: vector_store)

    result = await SearchKnowledgeTool().arun({"query": "what is gochujang"})

    assert "Gochujang" in result
    assert "source:" in result


async def test_knowledge_tool_degrades_when_no_index_is_built(monkeypatch):
    from app.tools.knowledge import SearchKnowledgeTool

    monkeypatch.setattr("app.tools.knowledge.get_vector_store", lambda: None)

    result = await SearchKnowledgeTool().arun({"query": "anything"})

    assert "not built" in result
    # Points the model at the tools that still work.
    assert "search_menu" in result


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_registry_exposes_every_expected_tool():
    from app.tools import get_tool_names

    names = get_tool_names()
    for expected in (
        "search_menu",
        "filter_menu",
        "list_menu_categories",
        "get_restaurant_info",
        "search_knowledge",
    ):
        assert expected in names


def test_registry_tool_names_are_unique():
    from app.tools import get_tool_names

    names = get_tool_names()
    assert len(names) == len(set(names))


def test_system_prompt_documents_only_real_tools():
    """The old prompt described a tool signature that did not exist."""
    from app.agent.prompts import get_system_prompt
    from app.tools import get_tool_names

    prompt = get_system_prompt()
    for name in get_tool_names():
        assert name in prompt, f"{name} is registered but absent from the prompt"
