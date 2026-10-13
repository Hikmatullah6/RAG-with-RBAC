"""Role-Based Access Control: which departments each role can read.
Run:  python rbac.py   (prints the access table)
"""
import hmac
import os


# Role -> departments that role may access.
# Department names must match the folder names in resources/data (lowercase).
ROLE_ACCESS = {
    "engineering": ["engineering", "general"],
    "finance": ["finance", "general"],
    "hr": ["hr", "general"],
    "marketing": ["marketing", "general"],
    "c_level": ["engineering", "finance", "general", "hr", "marketing"],
    "employee": ["general"],
}


# DEMO ONLY: one shared password for all demo accounts.
# Locally it defaults to "demo123". When deployed, set DEMO_PASSWORD as a secret,
# so the public URL isn't protected by a password that's written in the code.
DEMO_PASSWORD = os.getenv("DEMO_PASSWORD", "demo123")

USERS = {
    "tony":    {"password": DEMO_PASSWORD, "role": "engineering"},
    "sam":     {"password": DEMO_PASSWORD, "role": "finance"},
    "natasha": {"password": DEMO_PASSWORD, "role": "hr"},
    "bruce":   {"password": DEMO_PASSWORD, "role": "marketing"},
    "nick":    {"password": DEMO_PASSWORD, "role": "c_level"},
    "peter":   {"password": DEMO_PASSWORD, "role": "employee"},
}


def authenticate(username: str, password: str):
    """Return {"username", "role"} if the login is correct, otherwise None."""
    username = username.strip().lower()
    user = USERS.get(username)
    # compare_digest takes the same time whether the password is close or not (no timing hints)
    if user and hmac.compare_digest(user["password"], password):
        return {"username": username, "role": user["role"]}
    return None


def find_department(role: str) -> list[str]:
    """Return the departments a role can access. Unknown role -> [] (deny by default)."""
    role = role.strip().lower()
    return list(ROLE_ACCESS.get(role, []))  # list(...) returns a copy, so callers can't change the table


if __name__ == "__main__":
    for role in ROLE_ACCESS:
        print(f"{role:12} -> {find_department(role)}")
    print(f"{'Finance ':12} -> {find_department('Finance ')}   (case/space handled)")
    print(f"{'admin':12} -> {find_department('admin')}   (unknown role denied)")