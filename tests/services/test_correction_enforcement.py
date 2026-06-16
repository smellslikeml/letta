"""Tests for TRACE-style correction enforcement on the tool-execution path.

These exercise both the standalone enforcement helpers and the wiring edit in
``ToolExecutionManager.execute_tool_async`` (the existing call site), using real
Letta ``Block``/``Memory`` objects so the integration with block_manager-backed
memory is covered end to end.
"""

from types import SimpleNamespace

import pytest

# Non-new modules: the call site we wired into, plus the real memory schema.
from letta.schemas.block import Block
from letta.schemas.memory import Memory
from letta.services.tool_executor.correction_enforcement import (
    check_tool_call,
    enforce_corrections,
    extract_correction_rules,
    parse_correction_rules,
)
from letta.services.tool_executor.tool_execution_manager import ToolExecutionManager

SNAKE_CASE_RULE = "forbid | run_code | code | [a-z][a-zA-Z0-9]*[A-Z] | use snake_case for variable names, not camelCase"


def _memory_with(*rule_lines: str) -> Memory:
    block = Block(label="corrections", value="\n".join(rule_lines))
    return Memory(blocks=[block])


def test_parse_skips_prose_and_keeps_rules():
    text = "# my preferences\nPlease be concise.\n" + SNAKE_CASE_RULE + "\nnot | enough"
    rules = parse_correction_rules(text)
    assert len(rules) == 1
    assert rules[0].mode == "forbid"
    assert rules[0].field == "code"


def test_extract_rules_from_real_memory_blocks():
    rules = extract_correction_rules(SimpleNamespace(memory=_memory_with(SNAKE_CASE_RULE)))
    assert len(rules) == 1


def test_forbid_rule_blocks_matching_call():
    rules = parse_correction_rules(SNAKE_CASE_RULE)
    report = check_tool_call("run_code", {"code": "myVar = 1"}, rules)
    assert not report.allowed
    assert "snake_case" in report.violations[0]


def test_forbid_rule_allows_compliant_call():
    rules = parse_correction_rules(SNAKE_CASE_RULE)
    report = check_tool_call("run_code", {"code": "my_var = 1"}, rules)
    assert report.allowed


def test_rule_does_not_apply_to_other_tools():
    rules = parse_correction_rules(SNAKE_CASE_RULE)
    report = check_tool_call("send_message", {"code": "myVar = 1"}, rules)
    assert report.allowed


def test_require_mode_blocks_when_pattern_absent():
    rule = "require | save_file | path | \\.py$ | only write Python files in this project"
    rules = parse_correction_rules(rule)
    assert not check_tool_call("save_file", {"path": "notes.txt"}, rules).allowed
    assert check_tool_call("save_file", {"path": "main.py"}, rules).allowed


def test_enforce_corrections_no_rules_is_noop():
    report = enforce_corrections("run_code", {"code": "myVar = 1"}, SimpleNamespace(memory=Memory(blocks=[])))
    assert report.allowed
    assert enforce_corrections("run_code", {"code": "myVar = 1"}, None).allowed


@pytest.mark.asyncio
async def test_execution_manager_blocks_violating_call():
    """The wiring edit: execute_tool_async returns an error (without running the
    executor) when a proposed call violates a stored correction."""
    mgr = ToolExecutionManager.__new__(ToolExecutionManager)
    mgr.agent_state = SimpleNamespace(memory=_memory_with(SNAKE_CASE_RULE))
    mgr.logger = SimpleNamespace(info=lambda *a, **k: None, error=lambda *a, **k: None)

    tool = SimpleNamespace(name="run_code", tool_type="custom", return_char_limit=1000)
    result = await mgr.execute_tool_async("run_code", {"code": "myVar = 1"}, tool)

    assert result.status == "error"
    assert "snake_case" in result.func_return
    assert "run_code" in result.func_return
