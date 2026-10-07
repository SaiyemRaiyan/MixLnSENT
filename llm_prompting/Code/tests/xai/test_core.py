import itertools
import json
import tempfile
import unittest
from pathlib import Path

from src.xai.attribution import fit_kernel_attribution
from src.xai.config import FULL_MODELS
from src.xai.gates import _validate_rationale_prompt_freeze
from src.xai.oracle import _batches, run_variants
from src.xai.rationale import (
    match_rationale_words, parse_rationale_batch, validate_rationales,
)
from src.xai.report import load_complete_run
from src.xai.stats import holm_adjust, mcnemar_exact
from src.xai.tokenize import delete_tokens, keep_tokens, mask_tokens, tokenize
from src.xai.validity import compare_occlusion_to_lr
from src.xai.variants import (
    exact_shapley, occlusion_variants, sample_coalitions, variant_id,
)


class XaiCoreTests(unittest.TestCase):
    def test_tokenizer_preserves_whitespace_token_punctuation_and_spans(self):
        text = "bhalo,   kharap!"
        tokens = tokenize(text)
        self.assertEqual([token.text for token in tokens], ["bhalo,", "kharap!"])
        self.assertEqual(text[tokens[0].start:tokens[0].end], "bhalo,")
        self.assertEqual(delete_tokens(text, [0]), "kharap!")
        self.assertEqual(mask_tokens(text, [1]), "bhalo, [MASK]")
        self.assertEqual(keep_tokens(text, [0]), "bhalo,")

    def test_single_token_occlusion_is_skipped_and_bad_index_is_rejected(self):
        self.assertEqual(
            occlusion_variants({"index": 4, "sentence": "bhalo", "gold": 0}), []
        )
        with self.assertRaises(IndexError):
            delete_tokens("bhalo", [1])

    def test_exact_shapley_recovers_additive_games_through_eight_features(self):
        for n in range(2, 9):
            weights = [float(index + 1) for index in range(n)]
            values = {
                coalition: sum(weights[index] for index in coalition)
                for size in range(n + 1)
                for coalition in itertools.combinations(range(n), size)
            }
            actual = exact_shapley(values, n)
            for observed, expected in zip(actual, weights):
                self.assertAlmostEqual(observed, expected, delta=1e-12)

    def test_sampled_coalitions_are_seeded_nonempty_nonfull_and_complement_paired(self):
        first = sample_coalitions(12, 32, seed=17)
        self.assertEqual(first, sample_coalitions(12, 32, seed=17))
        self.assertEqual(len(first), 32)
        masks = set(first)
        self.assertNotIn((), masks)
        self.assertNotIn(tuple(range(12)), masks)
        for mask in masks:
            complement = tuple(index for index in range(12) if index not in mask)
            self.assertIn(complement, masks)

    def test_weighted_kernel_fit_recovers_additive_interior_game(self):
        n = 3
        values = {
            coalition: float(0 in coalition)
            for size in range(1, n)
            for coalition in itertools.combinations(range(n), size)
        }
        scores = fit_kernel_attribution(values, n, alpha=0)
        for observed, expected in zip(scores, [1.0, 0.0, 0.0]):
            self.assertAlmostEqual(observed, expected, delta=1e-10)

    def test_batch_packing_never_repeats_a_source_sentence(self):
        variants = [
            {"src_index": 1, "variant_id": "1a"},
            {"src_index": 1, "variant_id": "1b"},
            {"src_index": 2, "variant_id": "2a"},
            {"src_index": 3, "variant_id": "3a"},
        ]
        batches = list(_batches(variants, batch_size=3, seed=3))
        self.assertEqual(
            sorted(item["variant_id"] for batch in batches for item in batch),
            ["1a", "1b", "2a", "3a"],
        )
        self.assertTrue(all(
            len({item["src_index"] for item in batch}) == len(batch)
            for batch in batches
        ))

    def test_oracle_resume_and_metadata(self):
        variants = [
            {
                "variant_id": f"{index}:test:00000000",
                "src_index": index,
                "kind": "test",
                "mask": [],
                "rep": 0,
                "text": f"sentence {index}",
            }
            for index in range(3)
        ]
        calls = []

        def sender(prompt):
            calls.append(prompt)
            count = int(prompt.split("There are exactly ", 1)[1].split(" texts", 1)[0])
            return "\n".join(f"{index}: Positive" for index in range(1, count + 1))

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.jsonl"
            first = run_variants(
                variants, "mock", "local", path, scope="dev",
                sender=sender, batch_size=2, seed=5, log=lambda _: None,
            )
            self.assertEqual(len(first), 3)
            self.assertEqual(len(calls), 2)
            records = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertTrue(all(record["prediction"] == 0 for record in records))
            self.assertTrue(all(
                record["prompt"] == "sentiment_zero_shot_minimal_v1"
                for record in records
            ))
            manifest = json.loads(
                path.with_suffix(".jsonl.manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["complete"])

            again = run_variants(
                variants, "mock", "local", path, scope="dev",
                sender=sender, batch_size=2, seed=5, log=lambda _: None,
            )
            self.assertEqual(len(again), 3)
            self.assertEqual(len(calls), 2)

    def test_rationale_parse_and_repeated_token_alignment(self):
        parsed, missing = parse_rationale_batch("1: bhalo | bhalo\n2: NONE", 2)
        self.assertEqual(parsed, ["bhalo | bhalo", "NONE"])
        self.assertEqual(missing, [])
        match = match_rationale_words("bhalo, bhalo!", "bhalo | absent")
        self.assertEqual(match["matched"][0]["token_index"], 0)
        self.assertEqual(match["unmatched"], ["absent"])

    def test_rationale_validation_reports_parse_and_verbatim_rates(self):
        rows = [
            {"index": 1, "sentence": "bhalo movie"},
            {"index": 2, "sentence": "bad acting"},
        ]
        outputs = [
            {"src_index": 1, "rationale": "bhalo | movie"},
            {"src_index": 2, "rationale": "bad | missing"},
        ]
        result = validate_rationales(rows, outputs)
        self.assertEqual(result["parse_rate"], 1.0)
        self.assertEqual(result["verbatim_match_rate"], 0.75)
        self.assertEqual(result["matched_words"], 3)

    def test_rationale_prompt_freeze_requires_thresholds_for_every_full_model(self):
        passing = {
            f"{provider}/{model}": {
                "parse_rate": 0.95,
                "verbatim_match_rate": 0.90,
            }
            for provider, model in FULL_MODELS
        }
        _validate_rationale_prompt_freeze(passing)
        passing.pop(next(iter(passing)))
        with self.assertRaisesRegex(ValueError, "missing"):
            _validate_rationale_prompt_freeze(passing)
        passing = {
            f"{provider}/{model}": {
                "parse_rate": 0.95,
                "verbatim_match_rate": float("nan"),
            }
            for provider, model in FULL_MODELS
        }
        with self.assertRaisesRegex(ValueError, "verbatim_match_rate"):
            _validate_rationale_prompt_freeze(passing)

    def test_partial_run_is_rejected_by_report_loader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "partial.jsonl"
            path.write_text('{"variant_id":"x"}\n', encoding="utf-8")
            manifest_path = path.with_suffix(".jsonl.manifest.json")
            manifest_path.write_text(
                json.dumps({"complete": False}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "partial run"):
                load_complete_run(path)

    def test_lr_gate_excludes_unidentifiable_constant_hard_rankings(self):
        class MockLR:
            classes_ = [0, 1]

            def predict(self, texts):
                predictions = []
                for text in texts:
                    if {"steady", "alpha", "beta", "gamma"} & set(text.split()):
                        predictions.append(0)
                    else:
                        predictions.append(0 if "cue" in text.split() else 1)
                return predictions

            def predict_proba(self, texts):
                probabilities = []
                for text in texts:
                    if {"steady", "alpha", "beta", "gamma"} & set(text.split()):
                        probabilities.append([0.8, 0.2])
                    else:
                        probabilities.append(
                            [0.8, 0.2] if "cue" in text.split()
                            else [0.3, 0.7]
                        )
                return probabilities

        result = compare_occlusion_to_lr(
            MockLR(),
            [
                {"index": 1, "sentence": "cue word three four"},
                {"index": 2, "sentence": "steady alpha beta gamma"},
            ],
        )
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["identifiable_n"], 1)
        self.assertEqual(result["unidentifiable_n"], 1)
        self.assertEqual(result["median_spearman"], 1.0)
        self.assertEqual(result["top1_agreement"], 1.0)
        self.assertIsNone(result["per_sentence"][1]["spearman"])

    def test_paired_tests_and_holm_adjustment(self):
        result = mcnemar_exact([True, False, True], [False, True, True])
        self.assertEqual(result["first_only"], 1)
        self.assertEqual(result["second_only"], 1)
        for observed, expected in zip(
            holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]
        ):
            self.assertAlmostEqual(observed, expected)

    def test_variant_ids_change_for_replicates(self):
        self.assertNotEqual(
            variant_id(1, "baseline", [], 1),
            variant_id(1, "baseline", [], 2),
        )


if __name__ == "__main__":
    unittest.main()
