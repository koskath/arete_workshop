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

### 1. Create a Virtual Environment

From the project root:

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

```bash
source venv/bin/activate
```

---

## Install Dependencies

Install required packages:

```bash
pip3 install -r requirements.txt
```

---

## Create the Vector Store

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
