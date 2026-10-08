"""Role-Based Access Control: which departments each role can read.
Run:  python rbac.py   (prints the access table)
"""

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


def find_department(role: str) -> list[str]:
    """Return the departments a role can access. Unknown role -> [] (deny by default)."""
    role = role.strip().lower()
    return list(ROLE_ACCESS.get(role, []))  # list(...) returns a copy, so callers can't change the table


if __name__ == "__main__":
    for role in ROLE_ACCESS:
        print(f"{role:12} -> {find_department(role)}")
    print(f"{'Finance ':12} -> {find_department('Finance ')}   (case/space handled)")
    print(f"{'admin':12} -> {find_department('admin')}   (unknown role denied)")