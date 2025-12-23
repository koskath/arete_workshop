# ARETE Workshop guide

## macOS Setup Guide

### Requirements

* macOS
* Python **3.9, 3.10, or 3.11**
* pip (included with Python)

Check your Python version:

```bash
python3 --version
```

---

## Setup Instructions

### 0. Navigate to the arete_workshop directory
```bash
cd arete_workshop
```

### 1. Choose Your Version

This project offers two options:

- **Mistral Version** (`mistral_version`): Uses Mistral AI API for embeddings and chat. Requires an API key.
- **Open Source Version** (`open_source_version`): Uses open-source models that run locally. Recommended for machines that cannot run large models.

**Note:** If your machine cannot run large models, use the open source version.

### 2. Create a .env File

Create a `.env` file in the project root directory:

```bash
touch .env
```

Edit the `.env` file using nano:

```bash
nano .env
```

For the **Mistral version**, add your Mistral API key:
```
MISTRAL_API_KEY=your_mistral_api_key_here
```

For the **open source version**, the `.env` file can be empty (or you may add HuggingFace token if needed).

After editing, save and exit nano by pressing `Ctrl+X`, then `Y` to confirm, and `Enter` to save.

### 3. Create a Virtual Environment

From the project root:

```bash
python3 -m venv venv
```

### 4. Activate the Virtual Environment

```bash
source venv/bin/activate
```


## Install Dependencies

Install required packages (from the project root):

```bash
cd ..
pip3 install -r requirements.txt
```

### 5. Navigate to Your Chosen Version

Navigate to either `mistral_version` or `open_source_version` depending on your needs:

```bash
# For Mistral version:
cd mistral_version

# OR for open source version:
cd open_source_version
```

---

## Create the Vector Store

From your chosen version directory:

```bash
cd vectorstore_creation
python3 create_vectorstore.py
```

---

## Run the Application

```bash
cd ../ta_inference
python3 main.py
```

---

## Notes

* Make sure the virtual environment is activated before running commands.
* If dependencies fail, re-run `pip3 install -r requirements.txt`.
