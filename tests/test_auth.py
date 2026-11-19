import os
import sqlite3
import tempfile
import unittest
from contextlib import closing

from src.auth import initialize_auth_db, register_user, verify_user


class AuthenticationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.database_path = os.path.join(self.temp_dir.name, "accounts.sqlite3")
        initialize_auth_db(self.database_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_registration_stores_a_hash_and_verifies_credentials(self):
        self.assertTrue(register_user(self.database_path, "sample_user", "safe-pass1"))
        self.assertTrue(verify_user(self.database_path, "sample_user", "safe-pass1"))
        self.assertFalse(verify_user(self.database_path, "sample_user", "wrong-pass"))

        with closing(sqlite3.connect(self.database_path)) as connection:
            stored_hash = connection.execute(
                "SELECT password_hash FROM users WHERE username = ?", ("sample_user",)
            ).fetchone()[0]
        self.assertNotEqual(stored_hash, b"safe-pass-123")

    def test_duplicate_usernames_are_case_insensitive(self):
        self.assertTrue(register_user(self.database_path, "Sample_User", "safe-pass1"))
        self.assertFalse(register_user(self.database_path, "sample_user", "safe-pass2"))
        self.assertTrue(verify_user(self.database_path, "SAMPLE_USER", "safe-pass1"))

    def test_registration_rejects_invalid_username_and_short_password(self):
        with self.assertRaises(ValueError):
            register_user(self.database_path, "ab", "safe-pass-123")
        with self.assertRaises(ValueError):
            register_user(self.database_path, "valid_user", "short")
        with self.assertRaises(ValueError):
            register_user(self.database_path, "valid_user", "longer-than-10")

    def test_registration_accepts_password_at_10_character_limit(self):
        self.assertTrue(register_user(self.database_path, "limit_user", "1234567890"))
        self.assertTrue(verify_user(self.database_path, "limit_user", "1234567890"))

    def test_registration_accepts_email_as_account_name(self):
        email = "sample.user@example.com"
        self.assertTrue(register_user(self.database_path, email, "safe-pass1"))
        self.assertTrue(verify_user(self.database_path, email, "safe-pass1"))
        self.assertFalse(verify_user(self.database_path, "sample_user", "safe-pass1"))

    def test_registration_rejects_malformed_email(self):
        with self.assertRaises(ValueError):
            register_user(self.database_path, "not-an-email@", "safe-pass1")

    def test_unknown_username_does_not_authenticate(self):
        self.assertFalse(verify_user(self.database_path, "unknown_user", "safe-pass-123"))


if __name__ == "__main__":
    unittest.main()
