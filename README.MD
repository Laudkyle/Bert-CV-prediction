# BERT-Based Job Classification Web App

This is a Streamlit-based web application for job classification using a pre-trained BERT model. The app allows users to upload job-related data, predict job categories, and view evaluation metrics like accuracy, precision, recall, AUROC, and confusion matrices.

## Requirements

This web app requires the following dependencies:

### 1. **Streamlit and Core Libraries**
- **Streamlit**: Framework for creating interactive web applications.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.

### 2. **Model and NLP Libraries**
- **Transformers**: For using pre-trained BERT models.
- **Torch**: PyTorch for model inference.
- **Scikit-learn**: For calculating evaluation metrics like accuracy, precision, AUROC, and confusion matrices.

### 3. **Data Visualization**
- **Matplotlib**: For plotting graphs like ROC curves.
- **Seaborn**: For creating visualizations like confusion matrices.

### 4. **Deployment and Compatibility**
- **Watchdog**: For better reloading during development.
- **PyYAML**: For handling configuration files (if applicable).

## Installation Instructions

### 1. Create a Virtual Environment (Optional but recommended)
It is highly recommended to use a virtual environment to avoid conflicts with other Python projects.

```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
pip install streamlit pandas numpy transformers torch scikit-learn matplotlib seaborn watchdog pyyaml
```

Make sure to run this line to make all neccesary installs before running the application

```bash
pip install streamlit pandas numpy transformers torch scikit-learn matplotlib seaborn watchdog pyyaml
```


Use this command to start the app
```bash
streamlit run server.py
```
