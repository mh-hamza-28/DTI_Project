# Drug-Target Interaction Prediction

## Explanation of Problem
Drug-Target Interaction (DTI) prediction is a crucial step in drug discovery and development. Knowing whether a chemical compound (drug) binds to a target protein can accelerate the discovery of novel therapeutics and repurpose existing drugs for new diseases. Wet-lab experiments are costly and time-consuming, making computational prediction models highly valuable.

## Methodology
This project uses Machine Learning algorithms to predict DTIs. 
1. **Data Preprocessing**: We extract structural features of the drug compound from its SMILES representation using RDKit (Molecular Weight, Num Hydrogen Donors, Num Hydrogen Acceptors).
2. **Protein Features**: We extract compositional features from the protein amino acid sequence (counts of Alanine, Leucine, Glycine, and the total sequence length).
3. **Modeling**: The concatenated feature vectors are passed into Logistic Regression and Random Forest classifiers. We evaluate both models on an 80/20 train/test split.
4. **Web Application**: The project includes a complete Full-Stack web application powered by Flask (backend API) and vanilla HTML/CSS/JS (frontend) to easily interact with the trained models.

## How to Run the Project

### 1. Install Dependencies
Ensure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Run the Command-Line Pipeline
To train the models and see the evaluation metrics directly in the terminal:
```bash
python src/main.py
```

### 3. Run the Full-Stack Web Application
To launch the interactive DTI prediction website:
```bash
python src/app.py
```
Then, open your browser and navigate to `http://localhost:5000`.

## Sample Output
```
--- Logistic Regression Results ---
Accuracy: 1.0000
ROC-AUC score: 1.0000

--- Random Forest Results ---
Accuracy: 1.0000
ROC-AUC score: 1.0000
```

## Future Improvements
- **Deep Learning**: Implementing deep sequence models (e.g., LSTMs or Transformers) for protein sequences.
- **Graph Neural Networks (GNNs)**: Representing drugs as molecular graphs to capture structural details better than simple descriptors.
- **Larger Datasets**: Scaling up to real-world datasets like BindingDB or Davis for robust evaluation.
