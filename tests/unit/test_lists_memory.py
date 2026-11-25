"""Local lists, remembered notes, fuzzy matching and the JSON file store."""

from __future__ import annotations

import asyncio
import stat
from datetime import datetime, timezone
from pathlib import Path

import pytest

from assistant.adapters.fakes import MemoryStore
from assistant.adapters.store.json_file import JsonFileStore
from assistant.flows.lists import ListsFlow, list_name
from assistant.flows.matching import best_match, similarity
from assistant.flows.memory import MemoryFlow
from assistant.ports.flow import FlowContext

NOW = datetime(2026, 11, 25, 18, 0, tzinfo=timezone.utc)


def run(flow: ListsFlow | MemoryFlow, **arguments: object) -> str:
    return asyncio.run(flow.run(FlowContext("", arguments)))


def test_list_names_are_normalised() -> None:
    assert list_name("My Shopping List") == "shopping"
    assert list_name("todo") == "to-do" and list_name("to do") == "to-do"
    assert list_name("groceries") == "shopping" and list_name(None) == "shopping"
    assert list_name("packing") == "packing"


def test_list_add_read_remove_clear() -> None:
    store = MemoryStore()
    lists = ListsFlow(store)

    assert run(lists, action="add", list="shopping", items=["milk", "eggs"]) == "Added milk and eggs to your shopping list."
    assert run(lists, action="add", list="shopping", items=["Milk", "bread"]) == "Added bread to your shopping list. Milk is already on it."
    assert run(lists, action="read", list="shopping") == "Your shopping list has 3 items: milk, eggs, and bread."
    assert run(lists, action="remove", list="shopping", items=["the eggs", "cheese"]) == (
        "Removed eggs from your shopping list. I couldn't find cheese on it."
    )
    assert run(lists, action="clear", list="shopping") == "Okay, your shopping list is empty now."
    assert run(lists, action="read", list="shopping") == "Your shopping list is empty."
    assert store.documents["lists"] == {"lists": {}}  # empty lists are not kept


def test_lists_are_separate_and_survive_a_new_flow() -> None:
    store = MemoryStore()
    run(ListsFlow(store), action="add", list="to-do", items=["call the plumber"])
    assert run(ListsFlow(store), action="read", list="todo") == "Your to-do list has 1 item: call the plumber."
    assert run(ListsFlow(store), action="read", list="shopping") == "Your shopping list is empty."


def test_similarity_handles_word_forms_and_order() -> None:
    assert similarity("where did I park", "where I parked") >= 0.5
    assert similarity("wifi password", "the password for the wifi") >= 0.5
    assert similarity("wifi password", "locker code") < 0.5
    assert best_match("egg", ["milk", "eggs"], 0.75) == "eggs"
    assert best_match("cheese", ["milk", "eggs"], 0.75) is None


def test_memory_remember_recall_forget() -> None:
    store = MemoryStore()
    memory = MemoryFlow(store, lambda: NOW)

    assert run(memory, action="remember", key="where I parked", value="level 3") == "Okay, I'll remember where I parked: level 3."
    assert run(memory, action="recall", key="where did I park") == "Where I parked: level 3."
    assert run(memory, action="recall", key="locker code") == "I don't have anything saved about locker code."
    assert run(memory, action="remember", key="where I parked", value="level 5")  # replaces, not duplicates
    assert len(store.documents["memory"]["notes"]) == 1
    assert run(memory, action="recall", key="parking") == "Where I parked: level 5."
    assert memory.describe({"key": "where did I park"}) == "Forget where I parked?"
    assert run(memory, action="forget", key="where I parked") == "Okay, I forgot where I parked."
    assert run(memory, action="recall", key="where I parked") == "I don't have anything saved about where I parked."


def test_memory_threshold_is_configurable() -> None:
    strict = MemoryFlow(MemoryStore(), lambda: NOW, match_threshold=0.95)
    run(strict, action="remember", key="wifi password", value="sunflower22")
    assert run(strict, action="recall", key="password").startswith("I don't have anything saved")
    assert run(strict, action="recall", key="wifi password") == "Wifi password: sunflower22."


def test_json_store_round_trip_is_private(tmp_path: Path) -> None:
    store = JsonFileStore(tmp_path / "data")
    assert store.load("lists") == {}
    store.save("lists", {"lists": {"shopping": ["milk"]}})

    assert JsonFileStore(tmp_path / "data").load("lists") == {"lists": {"shopping": ["milk"]}}
    assert stat.S_IMODE((tmp_path / "data" / "lists.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((tmp_path / "data").stat().st_mode) == 0o700
    assert [p.name for p in (tmp_path / "data").iterdir()] == ["lists.json"]  # no temp files left
    with pytest.raises(ValueError):
        store.load("../escape")
