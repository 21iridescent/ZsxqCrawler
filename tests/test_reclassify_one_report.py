"""Helper that reclassifies one report and can force MinerU OCR."""

from __future__ import annotations

import importlib.util
import json
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from deepseek_classifier import ClassifyResult


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "reclassify_one_report.py"
    spec = importlib.util.spec_from_file_location("reclassify_one_report", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReclassifyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _load_script()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_report_id_uses_existing_text_without_ocr(self) -> None:
        report_dir = self.root / "reports" / "42"
        report_dir.mkdir(parents=True)
        (report_dir / "note.pdf").write_bytes(b"%PDF-1.4")
        (report_dir / "note.md").write_text("已有 OCR 正文。", encoding="utf-8")
        located = self.mod.locate_report(
            report_id="42",
            pdf=None,
            text=None,
            title="",
            reports_dir=self.root / "reports",
        )
        self.assertEqual(located.title, "note")
        with mock.patch.object(self.mod, "run_mineru") as mineru:
            path, text = self.mod.prepare_text(located, force_ocr=False, mineru_bin="mineru")
        mineru.assert_not_called()
        self.assertEqual(text, "已有 OCR 正文。")
        self.assertEqual(path.name, "note.md")

    def test_force_ocr_runs_mineru_and_classifies_its_markdown(self) -> None:
        pdf = self.root / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        bindir = self.root / "bin"
        bindir.mkdir()
        mineru = bindir / "mineru"
        mineru.write_text(
            "#!/usr/bin/env python3\n"
            "import sys\n"
            "from pathlib import Path\n"
            "args = sys.argv[1:]\n"
            "out = Path(args[args.index('-o') + 1])\n"
            "dest = out / 'auto'\n"
            "dest.mkdir(parents=True, exist_ok=True)\n"
            "(dest / 'report.md').write_text('# OCR\\n\\nMinerU 正文。\\n', encoding='utf-8')\n",
            encoding="utf-8",
        )
        mineru.chmod(mineru.stat().st_mode | stat.S_IEXEC)

        result = ClassifyResult(
            provider="openrouter",
            model="minimax/minimax-m3.1-flash",
            content="{}",
            data={"report_type": "其他", "summary": "MinerU 正文。"},
        )
        output = self.root / "out.json"
        seen: dict[str, str] = {}

        def fake_classify(text, *, title):
            seen["text"] = text
            seen["title"] = title
            return result

        with mock.patch.object(self.mod, "classify_report", side_effect=fake_classify):
            code = self.mod.main(
                [
                    "--pdf",
                    str(pdf),
                    "--force-ocr",
                    "--mineru-bin",
                    str(mineru),
                    "--output",
                    str(output),
                    "--title",
                    "示例研报",
                ]
            )
        self.assertEqual(code, 0)
        self.assertIn("MinerU 正文。", seen["text"])
        self.assertEqual(seen["title"], "示例研报")
        payload = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(payload["provider"], "openrouter")
        self.assertEqual(payload["model"], "minimax/minimax-m3.1-flash")
        self.assertEqual(payload["classification"]["report_type"], "其他")
        self.assertTrue(Path(payload["source"]).is_file())

    def test_missing_mineru_binary_is_a_clean_error(self) -> None:
        pdf = self.root / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4")
        code = self.mod.main(
            [
                "--pdf",
                str(pdf),
                "--force-ocr",
                "--mineru-bin",
                str(self.root / "missing-mineru"),
                "--output",
                str(self.root / "out.json"),
            ]
        )
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "out.json").exists())

    def test_force_ocr_without_pdf_fails(self) -> None:
        text = self.root / "only.md"
        text.write_text("只有文本。", encoding="utf-8")
        code = self.mod.main(["--text", str(text), "--force-ocr", "--output", str(self.root / "x.json")])
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "x.json").exists())

    def test_rejects_path_traversal_report_id(self) -> None:
        with self.assertRaises(self.mod.ReportInputError):
            self.mod.locate_report(
                report_id="../secrets",
                pdf=None,
                text=None,
                title="",
                reports_dir=self.root,
            )


if __name__ == "__main__":
    unittest.main()
