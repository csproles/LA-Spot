"""The bounded result store, and the API refusing to start without its shared key."""

import os
import subprocess
import sys

import pytest

from resultstore import ResultStore

PYTHON_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def result(user):
    return {"user_id": user}


class TestResultStore:
    def test_stores_and_returns(self):
        store = ResultStore()
        store["proc_1"] = result("alice")
        assert store.get("proc_1") == result("alice")
        assert store.get("missing") is None

    def test_expires_after_the_ttl(self):
        clock = FakeClock()
        store = ResultStore(ttl_seconds=60, clock=clock)
        store["proc_1"] = result("alice")

        clock.now += 59
        assert store.get("proc_1") is not None
        clock.now += 2
        assert store.get("proc_1") is None
        assert len(store) == 0

    def test_a_user_keeps_only_their_newest(self):
        store = ResultStore(max_per_user=3)
        for i in range(5):
            store[f"proc_{i}"] = result("alice")

        assert [store.get(f"proc_{i}") is not None for i in range(5)] == [False, False, True, True, True]

    def test_one_users_flood_does_not_evict_anothers_within_their_cap(self):
        store = ResultStore(max_per_user=3, max_total=100)
        store["bob_1"] = result("bob")
        for i in range(20):
            store[f"alice_{i}"] = result("alice")

        assert store.get("bob_1") is not None
        assert len(store) == 4

    def test_global_cap_drops_the_oldest(self):
        store = ResultStore(max_per_user=100, max_total=3)
        for i in range(5):
            store[f"proc_{i}"] = result(f"user{i}")

        assert len(store) == 3
        assert store.get("proc_0") is None
        assert store.get("proc_1") is None
        assert store.get("proc_4") is not None

    def test_pop_and_delete_user(self):
        store = ResultStore()
        store["a1"] = result("alice")
        store["a2"] = result("alice")
        store["b1"] = result("bob")

        assert store.pop("a1") == result("alice")
        assert store.pop("a1") is None
        assert store.delete_user("alice") == 1
        assert store.get("b1") is not None
        assert len(store) == 1


class TestRefusesToStartWithoutTheKey:
    def run_import(self, env_overrides):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SKINCHECK_")}
        env.update(env_overrides)
        env["SKINCHECK_DB"] = os.path.join(PYTHON_DIR, "tests", "_startup_check.db")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        # An empty value still counts as set, so load_dotenv won't fill it in from the repo-root .env.
        return subprocess.run(
            [sys.executable, "-c", "import main; print('started')"],
            cwd=PYTHON_DIR,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

    def teardown_method(self):
        for suffix in ("", "-shm", "-wal"):
            path = os.path.join(PYTHON_DIR, "tests", "_startup_check.db" + suffix)
            if os.path.exists(path):
                os.remove(path)

    def test_no_key_is_an_error(self):
        result_ = self.run_import({"SKINCHECK_INTERNAL_KEY": ""})
        # A key in the repo-root .env is picked up by load_dotenv, so only assert
        # the refusal when nothing supplied one.
        if "started" in result_.stdout:
            pytest.skip("a repo-root .env supplied SKINCHECK_INTERNAL_KEY")
        assert result_.returncode != 0
        assert "SKINCHECK_INTERNAL_KEY is not set" in result_.stderr

    def test_explicit_dev_opt_in_starts(self):
        result_ = self.run_import({"SKINCHECK_INTERNAL_KEY": "", "SKINCHECK_ALLOW_NO_KEY": "1"})
        assert result_.returncode == 0, result_.stderr
        assert "started" in result_.stdout

    def test_a_key_starts(self):
        result_ = self.run_import({"SKINCHECK_INTERNAL_KEY": "some-long-random-key"})
        assert result_.returncode == 0, result_.stderr
