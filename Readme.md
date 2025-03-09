# AI Research QA Chatbot
----------------------------------------------------------------------------------------------------
## Important
For the Intellihack hackathon, 

```bash
git clone <repository_url>
cd <repository_directory>
```
And go through with the Readme  in that Submission directory.


Below show the all chatbot build up from scratch.

## Setup Instructions

### 1. **Clone the repository (if you haven’t already)**

If you haven't cloned the repository yet, you can do so by running:

```bash
git clone <repository_url>
cd <repository_directory>
```

### 2. **Create a Virtual Environment**

In your project directory, create a new virtual environment:

```bash
python -m venv myenv
```

### 3. **Activate the Virtual Environment**

- **For Linux/macOS:**

```bash
source myenv/bin/activate
```

- **For Windows:**

```bash
.\myenv\Scripts\activate
```

Once activated, you should see `(myenv)` at the beginning of your terminal prompt.

### 4. **Install Required Dependencies**

Ensure that you have a `requirements.txt` file in your project directory. If it’s missing, you can create it manually by listing all the dependencies for the project.

Once you have `requirements.txt` ready, install the required dependencies by running:

```bash
pip install -r requirements.txt
```

### 5. **Upgrade Pip (Optional but recommended)**

You can also upgrade pip to the latest version:

```bash
pip install --upgrade pip
```
1. You can run this to collect all the given documents and wikipedia documents together.
```bash
python Scripts/dataset-collecting.py
```

2. Now it is time for the preprocessing
```bash
python Scripts/dataset-preprocessing.py
```
3. For the furhter development. we need to synthetic data
```bash
python Scripts/synthetic-data-generator.py
```
4. After this we can fine tune the qwen 2.5 3b model, using peft
```bash
python Scripts/peft-qwen2.5.py
```
For us , this took more than 9 hours, it will depend on the infrastructure you are using.

5. In the final-architecture.py, you will find all the implemented architecture. You can select one of it and comment others and play around with that.

6. As mentioned in the report, we evaluated the model, with different metrics, you can all of those 
