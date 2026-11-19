"""Local account storage and password verification for the Streamlit prototype."""
import hashlib
import hmac
import os
import re
import secrets
import sqlite3
from contextlib import closing


_PASSWORD_ITERATIONS = 600_000
_MIN_PASSWORD_LENGTH = 8
_MAX_PASSWORD_LENGTH = 10
_SALT_BYTES = 16
_USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_EMAIL_PATTERN = re.compile(
    r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+"
    r"(?:\.[A-Za-z0-9!#$%&'*+/=?^_`{|}~-]+)*"
    r"@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
    r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+$"
)
_DUMMY_SALT = b"tenglish-insights-login-check"


def initialize_auth_db(database_path):
    """Create the local user table when it does not exist."""
    directory = os.path.dirname(os.path.abspath(database_path))
    os.makedirs(directory, exist_ok=True)
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY COLLATE NOCASE,
                    salt BLOB NOT NULL,
                    password_hash BLOB NOT NULL
                )
                """
            )


def _derive_password_hash(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, _PASSWORD_ITERATIONS
    )


def register_user(database_path, username, password):
    """Create an account, returning False if its username or email already exists."""
    username = username.strip()
    is_username = _USERNAME_PATTERN.fullmatch(username) is not None
    is_email = (
        len(username) <= 254 and _EMAIL_PATTERN.fullmatch(username) is not None
    )
    if not (is_username or is_email):
        raise ValueError(
            "Enter a username (3–32 letters, numbers, ., _ or -) "
            "or a valid email address."
        )
    if not _MIN_PASSWORD_LENGTH <= len(password) <= _MAX_PASSWORD_LENGTH:
        raise ValueError(
            "Password must contain between 8 and 10 characters."
        )

    salt = secrets.token_bytes(_SALT_BYTES)
    password_hash = _derive_password_hash(password, salt)
    try:
        with closing(sqlite3.connect(database_path)) as connection:
            with connection:
                connection.execute(
                    "INSERT INTO users (username, salt, password_hash) VALUES (?, ?, ?)",
                    (username, salt, password_hash),
                )
    except sqlite3.IntegrityError:
        return False
    return True


def verify_user(database_path, username, password):
    """Verify credentials without revealing whether a username exists."""
    username = username.strip()
    with closing(sqlite3.connect(database_path)) as connection:
        row = connection.execute(
            "SELECT salt, password_hash FROM users WHERE username = ?", (username,)
        ).fetchone()

    if row is None:
        candidate = _derive_password_hash(password, _DUMMY_SALT)
        hmac.compare_digest(candidate, bytes(len(candidate)))
        return False

    salt, expected_hash = row
    candidate = _derive_password_hash(password, salt)
    return hmac.compare_digest(candidate, expected_hash)
