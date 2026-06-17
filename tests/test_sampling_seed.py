import unittest
import unittest.mock

from openrouter_client import complete_text
from sampling_seed import VALIDATOR_TOP_P, deterministic_sampling_seed, judge_seed_material, solver_seed_material
from tau.io.openrouter import LLMRequest


class SamplingSeedTest(unittest.TestCase):
    def test_deterministic_material_is_stable(self):
        a = deterministic_sampling_seed(configured=None, material="judge:task:model:abc")
        b = deterministic_sampling_seed(configured=None, material="judge:task:model:abc")
        c = deterministic_sampling_seed(configured=None, material="judge:task:model:xyz")
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_configured_seed_overrides_material(self):
        self.assertEqual(
            deterministic_sampling_seed(configured=42, material="anything"),
            42,
        )

    def test_judge_seed_material_changes_with_patch(self):
        base = judge_seed_material(
            task_name="task-a",
            model="google/gemini-3.1-flash-lite",
            king_patch="king",
            challenger_patch="challenger",
        )
        changed = judge_seed_material(
            task_name="task-a",
            model="google/gemini-3.1-flash-lite",
            king_patch="king",
            challenger_patch="challenger-v2",
        )
        self.assertNotEqual(base, changed)

    def test_solver_seed_material_includes_agent_hash(self):
        a = solver_seed_material(task_name="t", solution_name="s", agent_hash="aaa")
        b = solver_seed_material(task_name="t", solution_name="s", agent_hash="bbb")
        self.assertNotEqual(a, b)

    def test_validator_top_p_default(self):
        self.assertEqual(VALIDATOR_TOP_P, 0.01)

    def test_llm_request_cache_key_includes_seed(self):
        a = LLMRequest(prompt="q", model="m", seed=1)
        b = LLMRequest(prompt="q", model="m", seed=2)
        self.assertNotEqual(a.cache_key(), b.cache_key())


class OpenRouterSeedPayloadTest(unittest.TestCase):
    def test_complete_text_passes_seed(self):
        captured: dict[str, object] = {}

        class FakeClient:
            def complete_text(self, request, *, timeout):
                captured["seed"] = request.seed
                captured["timeout"] = timeout
                return '{"winner":"tie","candidate_a_score":50,"candidate_b_score":50,"rationale":"ok"}'

        with unittest.mock.patch("openrouter_client._build_client", return_value=FakeClient()):
            complete_text(
                prompt="hello",
                model="google/gemini-3.1-flash-lite",
                timeout=10,
                openrouter_api_key="key",
                temperature=0,
                top_p=VALIDATOR_TOP_P,
                seed=12345,
            )
        self.assertEqual(captured["seed"], 12345)


if __name__ == "__main__":
    unittest.main()
