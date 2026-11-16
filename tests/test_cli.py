"""Regression checks for documented installed entry point.

The tests do not download external datasets or invoke long research training.
"""
from __future__ import annotations
from pathlib import Path
from types import SimpleNamespace
import sys

from adaptiveclusterlab import cli


def test_reproduction_cli_forwards_to_repository_runner(monkeypatch):
    recorded = {}
    def pretend_run(command, check):
        recorded["command"] = command
        recorded["check"] = check
        return SimpleNamespace(returncode=0)
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["adaptiveclusterlab", "electricity", "--quick"])
    monkeypatch.setattr(cli.subprocess, "run", pretend_run)
    cli.main()
    assert recorded["check"] is True
    assert recorded["command"][0] == sys.executable
    assert Path(recorded["command"][1]).name == "reproduce.py"
    assert recorded["command"][2:] == ["electricity", "--quick"]


def test_reference_cli_remains_compatible(monkeypatch, tmp_path):
    outfile = tmp_path / "reference.json"
    monkeypatch.setattr(sys, "argv", ["adaptiveclusterlab", "--seed", "42",
                                        "--batches", "2", "--clusters", "3",
                                        "--output", str(outfile)])
    cli.main()
    assert outfile.is_file()

def test_packaged_controller_is_importable():
    from adaptiveclusterlab.controller import (
        ACTIONS, PolicyEnvironment, TabularQLearning, make_aba
    )
    assert len(ACTIONS) == 12
    assert len(make_aba(seed=1, batches=3)) == 3
    assert PolicyEnvironment is not None and TabularQLearning is not None
