import sys
from retrieve import retrieve, get_vectorstore

departments = {
            "engineering" : ["engineering", "general"],
            "finance" : ["finance", "general"],
            "hr" : ["hr", "general"],
            "marketing" : ["marketing", "general"],
            "c_level" : ["engineering", "finance", "general", "hr", "marketing"],
            "employee" : ["general"]
        }

def find_department(role):
    if role.lower() in departments:
        return departments[role]
    else:
        return []

if __name__ == "__main__":
    role = input("What is your role: ")
    departments = find_department(role)
    if len(departments) == 0:
        print("Role was not found. ")
        sys.exit(0)
    else:
        print("Here are the departments you can access:", departments)
    while True:
        question = input("\nQuestion: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if not question:
            continue 
        print(retrieve(get_vectorstore(), question, 5, departments))

