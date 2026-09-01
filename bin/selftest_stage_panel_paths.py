#!/usr/bin/env python3
"""Focused, offline checks for symlink-safe token-panel staging."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

import stage_panel_paths as S
from fidelity import panel as panel_contract


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.panel = root / "fetched-panel"
        self.source = self.panel / "calibration" / "panel-v1"
        self.source.mkdir(parents=True)
        self.anchor = root / "destination-anchor"
        self.anchor.mkdir()
        self.destination = self.anchor / "artifacts" / "dataset" / "calibration" / "panel-v1"
        self.contents = {
            "panel.json": b'{"schema":"fixture-panel"}\n',
            "arrays/window-0001.tokens.npy": b"fixture-token-array\x00\x01",
        }
        for relative, raw in self.contents.items():
            target = self.source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        self.receipt = self._base_receipt()
        self.receipt_path = self.panel / "token-panel-receipt.json"
        self.seal_and_write()

    def _base_receipt(self):
        rows = []
        for relative, raw in sorted(self.contents.items()):
            rows.append({
                "path": str(S.LEGACY_PREFIX) + "/" + relative,
                "bytes": len(raw),
                "sha256": sha(raw),
            })
        return {
            "schema": panel_contract.ARTIFACT_RECEIPT_SCHEMA,
            "token_panel_artifact_sha256": sha(self.contents["panel.json"]),
            "artifacts": rows,
            "corpus_receipt_sha256": "c" * 64,
            "domains": list(S.PINNED_DOMAINS),
            "final_windows": 1,
            "final_prediction_positions": 3,
            "roles": list(S.PINNED_ROLES),
            "tokenizer_receipt_sha256": "a" * 64,
        }

    def seal_and_write(self) -> str:
        body = dict(self.receipt)
        body.pop("receipt_sha256", None)
        claimed = sha(panel_contract._canonical(body, newline=True))
        self.receipt = dict(body)
        self.receipt["receipt_sha256"] = claimed
        self.receipt_path.write_bytes(panel_contract._canonical(self.receipt, newline=True))
        return claimed

    def stage(self, *, check_only=False, expected_receipt_sha256=None):
        claimed = expected_receipt_sha256 or self.receipt["receipt_sha256"]
        return S.stage_panel(
            self.panel, self.receipt_path, check_only=check_only,
            destination_prefix=self.destination,
            destination_anchor=self.anchor,
            expected_receipt_sha256=claimed,
            expected_panel_sha256=self.receipt["token_panel_artifact_sha256"],
            expected_artifact_count=len(self.receipt["artifacts"]),
            expected_final_windows=1,
            expected_final_prediction_positions=3,
        )


class StagePanelPathsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="stage-panel-selftest-")
        self.fixture = Fixture(Path(self.temporary.name))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def assert_refused_without_destination(self, error=S.StageError) -> None:
        with self.assertRaises(error):
            self.fixture.stage()
        self.assertFalse(self.fixture.destination.exists())

    def reseal(self) -> str:
        return self.fixture.seal_and_write()

    def test_valid_copy_uses_exact_bytes_and_public_mode(self) -> None:
        summary = self.fixture.stage()
        self.assertEqual(summary["staged"], len(self.fixture.contents))
        self.assertEqual(summary["already_present"], 0)
        for relative, raw in self.fixture.contents.items():
            target = self.fixture.destination / relative
            self.assertEqual(target.read_bytes(), raw)
            self.assertTrue(stat.S_ISREG(target.lstat().st_mode))
            self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o644)

    def test_valid_preexisting_regular_files_are_accepted(self) -> None:
        for relative, raw in self.fixture.contents.items():
            target = self.fixture.destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        summary = self.fixture.stage()
        self.assertEqual(summary["already_present"], len(self.fixture.contents))
        self.assertEqual(summary["staged"], 0)

    def test_check_only_reports_without_any_mutation(self) -> None:
        summary = self.fixture.stage(check_only=True)
        self.assertEqual(summary["would_stage"], len(self.fixture.contents))
        self.assertEqual(summary["staged"], 0)
        self.assertFalse(self.fixture.destination.exists())

    def test_bad_legacy_seal_refuses_before_mutation(self) -> None:
        claimed = self.fixture.receipt["receipt_sha256"]
        bad = claimed[:-1] + ("0" if claimed[-1] != "0" else "1")
        self.fixture.receipt["receipt_sha256"] = bad
        self.fixture.receipt_path.write_bytes(
            panel_contract._canonical(self.fixture.receipt, newline=True))
        with self.assertRaises(S.ReceiptError):
            self.fixture.stage(expected_receipt_sha256=bad)
        self.assertFalse(self.fixture.destination.exists())

    def test_wrong_schema_and_extra_top_level_field_refuse(self) -> None:
        for mutation in (
                lambda doc: doc.__setitem__("schema", "wrong.schema"),
                lambda doc: doc.__setitem__("unexpected", True)):
            with self.subTest(mutation=mutation):
                fixture = Fixture(Path(self.temporary.name) / ("case-%d" % id(mutation)))
                mutation(fixture.receipt)
                fixture.seal_and_write()
                with self.assertRaises(S.ReceiptError):
                    fixture.stage()
                self.assertFalse(fixture.destination.exists())

    def test_row_shape_and_exact_numeric_type_refuse(self) -> None:
        for mutation in (
                lambda row: row.__setitem__("unexpected", 1),
                lambda row: row.__setitem__("bytes", True),
                lambda row: row.__setitem__("sha256", row["sha256"].upper())):
            with self.subTest(mutation=mutation):
                fixture = Fixture(Path(self.temporary.name) / ("row-%d" % id(mutation)))
                mutation(fixture.receipt["artifacts"][0])
                fixture.seal_and_write()
                with self.assertRaises(S.ReceiptError):
                    fixture.stage()
                self.assertFalse(fixture.destination.exists())

    def test_noncanonical_relative_traversal_and_wrong_prefix_refuse(self) -> None:
        paths = (
            "calibration/panel-v1/panel.json",
            str(S.LEGACY_PREFIX) + "/arrays/../panel.json",
            "/workspace/artifacts/dataset/calibration/other/panel.json",
            str(S.LEGACY_PREFIX) + "//panel.json",
        )
        for index, path in enumerate(paths):
            with self.subTest(path=path):
                fixture = Fixture(Path(self.temporary.name) / ("path-%d" % index))
                fixture.receipt["artifacts"][0]["path"] = path
                fixture.seal_and_write()
                with self.assertRaises(S.ReceiptError):
                    fixture.stage()
                self.assertFalse(fixture.destination.exists())

    def test_duplicate_destination_and_digest_ambiguity_refuse(self) -> None:
        fixture = self.fixture
        fixture.receipt["artifacts"].append(dict(fixture.receipt["artifacts"][0]))
        fixture.seal_and_write()
        with self.assertRaises(S.ReceiptError):
            fixture.stage()
        self.assertFalse(fixture.destination.exists())

        other = Fixture(Path(self.temporary.name) / "duplicate-digest")
        other.receipt["artifacts"][1]["sha256"] = other.receipt["artifacts"][0]["sha256"]
        other.seal_and_write()
        with self.assertRaises(S.ReceiptError):
            other.stage()
        self.assertFalse(other.destination.exists())

    def test_duplicate_json_key_and_nonfinite_json_refuse(self) -> None:
        raw = self.fixture.receipt_path.read_text(encoding="utf-8")
        duplicated = raw.replace('"schema":', '"schema":"duplicate","schema":', 1)
        self.fixture.receipt_path.write_text(duplicated, encoding="utf-8")
        self.assert_refused_without_destination(S.ReceiptError)

        fixture = Fixture(Path(self.temporary.name) / "nonfinite")
        raw = fixture.receipt_path.read_text(encoding="utf-8")
        fixture.receipt_path.write_text(raw[:-2] + ',"x":NaN}\n', encoding="utf-8")
        with self.assertRaises(S.ReceiptError):
            fixture.stage()
        self.assertFalse(fixture.destination.exists())

    def test_source_leaf_and_ancestor_symlinks_refuse(self) -> None:
        leaf = self.fixture.source / "arrays" / "window-0001.tokens.npy"
        external = self.fixture.root / "external-array"
        external.write_bytes(self.fixture.contents["arrays/window-0001.tokens.npy"])
        leaf.unlink()
        leaf.symlink_to(external)
        self.assert_refused_without_destination()

        fixture = Fixture(Path(self.temporary.name) / "ancestor")
        real_arrays = fixture.root / "real-arrays"
        (fixture.source / "arrays").rename(real_arrays)
        (fixture.source / "arrays").symlink_to(real_arrays, target_is_directory=True)
        with self.assertRaises(S.StageError):
            fixture.stage()
        self.assertFalse(fixture.destination.exists())

    def test_destination_ancestor_symlink_refuses(self) -> None:
        self.fixture.destination.mkdir(parents=True)
        outside = self.fixture.root / "outside-destination"
        outside.mkdir()
        (self.fixture.destination / "arrays").symlink_to(
            outside, target_is_directory=True)
        with self.assertRaises(S.StageError):
            self.fixture.stage()
        self.assertEqual(list(outside.iterdir()), [])

    def test_destination_mismatch_refuses_without_overwrite(self) -> None:
        target = self.fixture.destination / "panel.json"
        target.parent.mkdir(parents=True)
        hostile = b"do-not-overwrite"
        target.write_bytes(hostile)
        with self.assertRaises(S.StageError):
            self.fixture.stage()
        self.assertEqual(target.read_bytes(), hostile)
        self.assertFalse((self.fixture.destination / "arrays").exists())

    def test_capture_receipt_never_causes_teacher_symlink_attempt(self) -> None:
        teacher_target = self.fixture.root / "legacy-teacher" / "logits"
        capture = {
            "logit_files": [{"path": str(teacher_target / "window.safetensors"),
                              "bytes": 1, "sha256": "b" * 64}]
        }
        (self.fixture.panel / "capture-receipt.json").write_text(
            json.dumps(capture), encoding="utf-8")
        (self.fixture.panel / "logits").mkdir()
        (self.fixture.panel / "logits" / "window.safetensors").write_bytes(b"x")
        self.fixture.stage()
        self.assertFalse(teacher_target.exists())
        self.assertFalse(teacher_target.parent.exists())


if __name__ == "__main__":
    unittest.main()
