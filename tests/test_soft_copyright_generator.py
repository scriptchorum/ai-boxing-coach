"""Tests for the CPCC software copyright material generator."""

from __future__ import annotations

import pathlib
import unittest

from tools.generate_soft_copyright_materials import (
    DEFAULT_SHORT_NAME,
    DEFAULT_SOFTWARE_NAME,
    DEFAULT_VERSION,
    LINES_PER_PAGE,
    TOTAL_CODE_PAGES,
    build_application_guide_markdown,
    build_source_code_html,
    build_user_manual_html,
    collect_ordered_source_files,
    sanitize_and_wrap_source_lines,
)


class TestSoftCopyrightGenerator(unittest.TestCase):
    def test_source_code_and_manual_generation_invariants(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        software_title = f"{DEFAULT_SOFTWARE_NAME} {DEFAULT_VERSION}"

        files = collect_ordered_source_files(repo_root)
        self.assertGreaterEqual(len(files), 15)
        self.assertEqual(files[0].name, "main.py")

        physical_lines, raw_count = sanitize_and_wrap_source_lines(
            files, repo_root, software_title
        )
        self.assertGreaterEqual(raw_count, 5000)
        self.assertGreaterEqual(len(physical_lines), TOTAL_CODE_PAGES * LINES_PER_PAGE)

        code_html, code_txt = build_source_code_html(physical_lines, software_title)
        self.assertEqual(code_html.count('<section class="page">'), TOTAL_CODE_PAGES)
        self.assertIn(f"第 1 页 共 {TOTAL_CODE_PAGES} 页", code_txt)
        self.assertIn(f"第 {TOTAL_CODE_PAGES} 页 共 {TOTAL_CODE_PAGES} 页", code_txt)

        manual_html = build_user_manual_html(
            software_title=software_title,
            software_name=DEFAULT_SOFTWARE_NAME,
            short_name=DEFAULT_SHORT_NAME,
            version=DEFAULT_VERSION,
            total_code_lines=raw_count,
        )
        self.assertEqual(manual_html.count('<section class="manual-page">'), 15)
        self.assertIn("data:image/png;base64,", manual_html)

        guide_md = build_application_guide_markdown(
            software_name=DEFAULT_SOFTWARE_NAME,
            short_name=DEFAULT_SHORT_NAME,
            version=DEFAULT_VERSION,
            raw_code_lines=raw_count,
        )
        self.assertIn(DEFAULT_SOFTWARE_NAME, guide_md)
        self.assertIn(str(raw_count), guide_md)


if __name__ == "__main__":
    unittest.main()
