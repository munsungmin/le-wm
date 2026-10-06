from datetime import datetime, timezone
import unittest

from scripts.utils.records import REQUIRED_RESULT_FIELDS, make_run_id, validate_result_row


class RecordTests(unittest.TestCase):
    def test_run_id_contract(self):
        now = datetime(2026, 10, 6, 1, 2, 3, tzinfo=timezone.utc)
        self.assertEqual(
            make_run_id("pusht", "lewm_ckpt", 42, now),
            "pusht_lewm_ckpt_42_20261006T010203Z",
        )

    def test_result_contract_requires_all_eight_names(self):
        row = {name: None for name in REQUIRED_RESULT_FIELDS}
        validate_result_row(row)
        row.pop("horizon")
        with self.assertRaisesRegex(ValueError, "horizon"):
            validate_result_row(row)
