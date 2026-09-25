"""AttentionDQNPolicy had no unit test, and defined `_update_target` twice.

The second definition silently replaced the first. They were identical, so nothing
changed at run time, but a later edit to one copy would have been dead code. The
tripwire below scans the repository for any class that defines a method twice.
"""
from __future__ import annotations

import ast
from pathlib import Path

import torch

from models.self_model.attention_dqn_policy import AttentionDQNPolicy

ROOT = Path(__file__).resolve().parents[1]


def test_update_target_copies_the_online_weights():
    policy = AttentionDQNPolicy({"action_dim": 5, "device": "cpu",
                                 "attn_d_model": 16, "attn_d_ff": 32}, None, None)
    with torch.no_grad():
        for p in policy.q_head.parameters():
            p.add_(1.0)
    policy._update_target()
    for online, target in ((policy.sample_proj, policy.target_sample_proj),
                           (policy.patch_proj, policy.target_patch_proj),
                           (policy.q_head, policy.target_q_head)):
        for a, b in zip(online.parameters(), target.parameters()):
            assert torch.equal(a, b)


def test_no_class_defines_a_method_twice():
    duplicates = []
    for folder in ("models", "scripts", "simulations"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                seen = set()
                for item in node.body:
                    if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        continue
                    # property setters and similar reuse a name on purpose
                    if any(isinstance(d, ast.Attribute) for d in item.decorator_list):
                        continue
                    if item.name in seen:
                        duplicates.append(f"{path.relative_to(ROOT)}:{item.lineno} "
                                          f"{node.name}.{item.name}")
                    seen.add(item.name)
    assert duplicates == []
