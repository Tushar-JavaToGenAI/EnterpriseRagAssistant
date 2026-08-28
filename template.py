import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "enterprise_rag"

list_of_files = [
    ".env",
    "requirements.txt",
    "README.md",
    "data/leave_and_benefits.md",
    "data/it_security_policy.txt",
    "components/__init__.py",
    "components/loaders.py",
    "components/chunkers.py",
    "components/ingestion.py",
    "components/retriever.py",
    "components/generator.py",
    "app.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")