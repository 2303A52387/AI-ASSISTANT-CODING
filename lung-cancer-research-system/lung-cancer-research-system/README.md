# 🫁 LungAI Research System

> **Early Detection of Lung Cancer Using Predictive Modeling Incorporating CTGAN Features and Tree-Based Learning**
>
> Implementation of the methodology from:  
> **IEEE Access, Vol. 13, 2025** · DOI: [10.1109/ACCESS.2025.3543215](https://doi.org/10.1109/ACCESS.2025.3543215)  
> Author: Abdulrahman Alzahrani, University of Hafr Al Batin, Saudi Arabia

---

## 📊 Key Results

| Method     | Best Classifier | Accuracy | Precision | Recall | F1-Score |
|------------|----------------|----------|-----------|--------|----------|
| **CTGAN**  | **Random Forest** | **98.93%** | **99%** | **99%** | **99%** |
| SMOTE-ENN  | XGBoost / SGDC  | 98.38%   | 98%       | 98%    | 98%      |
| Borderline-SMOTE | XGBoost  | 96.29%   | 97%       | 96%    | 96%      |
| SMOTE      | Multiple        | 95.37%   | 95%       | 95%    | 95%      |
| Original   | Extra Trees     | 79.62%   | 80%       | 80%    | 80%      |

---

## 🏗️ Architecture

```
lung-cancer-research-system/
├── backend/                   # Node.js + Express
│   ├── server.js              # Main application
│   ├── config/db.js           # MongoDB connection
│   ├── models/
│   │   ├── User.js
│   │   ├── Dataset.js
│   │   └── Prediction.js
│   ├── routes/
│   │   ├── authRoutes.js      # Login / Register / Logout
│   │   └── dashboardRoutes.js # All dashboard pages
│   ├── middleware/
│   │   └── authMiddleware.js  # Session auth guard
│   └── uploads/               # CSV dataset uploads
│
├── frontend/
│   ├── views/                 # EJS templates
│   │   ├── login.ejs
│   │   ├── register.ejs
│   │   ├── dashboard.ejs      # Overview
│   │   ├── datasetUpload.ejs
│   │   ├── preprocess.ejs
│   │   ├── ctgan.ejs
│   │   ├── training.ejs
│   │   ├── evaluation.ejs
│   │   ├── prediction.ejs
│   │   ├── visualization.ejs
│   │   ├── history.ejs
│   │   └── partials/
│   │       ├── header.ejs
│   │       └── sidebar.ejs
│   └── public/
│       ├── css/style.css      # Full custom design system
│       └── js/dashboard.js
│
└── ml-service/                # Python Flask ML API
    ├── app.py                 # Flask endpoints
    ├── train_model.py         # 10 ML classifiers
    ├── ctgan_generator.py     # CTGAN synthesis
    └── requirements.txt
```

---

## 🚀 Quick Start

### Prerequisites
- Node.js ≥ 18
- Python ≥ 3.9
- MongoDB (local or Atlas)

---

### 1. Clone & Install — Backend

```bash
cd backend
npm install
```

### 2. Install — Python ML Service

```bash
cd ml-service
pip install -r requirements.txt
```

> ⚠️ If CTGAN installation fails, the system uses a Gaussian augmentation fallback automatically. Core functionality is unaffected.

### 3. Start MongoDB

```bash
# Local MongoDB
mongod --dbpath /data/db

# Or use MongoDB Atlas — update MONGO_URI in step 4
```

### 4. Environment (optional)

Create `backend/.env`:
```
PORT=3000
MONGO_URI=mongodb://localhost:27017/lungcancer_research
ML_SERVICE_URL=http://localhost:5000
```

### 5. Start Python ML Service

```bash
cd ml-service
python app.py
# Running at http://localhost:5000
```

### 6. Start Node.js Backend

```bash
cd backend
node server.js
# Running at http://localhost:3000
```

### 7. Open in Browser

```
http://localhost:3000
```

---

## 🔑 Demo Login

| Field    | Value                  |
|----------|------------------------|
| Email    | demo@research.edu      |
| Password | demo123                |

> The system works **without MongoDB** — demo login is always available and all ML results are simulated if the Python service is offline.

---

## 📋 Dashboard Sections

| Section              | Description |
|----------------------|-------------|
| **Overview**         | KPI cards, research pipeline, model performance table |
| **Dataset Upload**   | Upload lung cancer CSV, view statistics & class distribution |
| **Preprocessing**    | Feature descriptions, encoding info, imbalance analysis |
| **CTGAN Generation** | Configure & run CTGAN, real vs synthetic comparison |
| **Model Training**   | Train 10 classifiers with any balancing method |
| **Evaluation**       | Confusion matrix, ROC curve, cross-validation, feature importance |
| **Prediction**       | Patient feature input → cancer detection result |
| **Visualization**    | Full chart suite: accuracy comparison, ROC, feature importance, SOTA |
| **History**          | Previous prediction records |

---

## 🧠 ML Pipeline

```
Dataset (309 instances, 16 features)
    ↓
Preprocessing (encode, normalize, 80/20 split)
    ↓
CTGAN Training (epochs=300, batch=500, lr=0.0002)
    ↓
Synthetic Generation (270 Normal class samples)
    ↓
Augmented Dataset (540 balanced instances)
    ↓
Random Forest (n_estimators=200, max_depth=20)
    ↓
5-Fold Cross-Validation → 98.93% Accuracy
```

---

## 📦 Tech Stack

| Layer      | Technology |
|------------|------------|
| Frontend   | EJS, Bootstrap 5, Chart.js |
| Backend    | Node.js, Express.js |
| Database   | MongoDB + Mongoose |
| ML Service | Python, Flask, scikit-learn |
| Data Aug   | CTGAN, SMOTE, Borderline-SMOTE, SMOTE-ENN |
| Classifiers| Random Forest, XGBoost, Extra Trees, SVM, KNN, LR, DT, GBC, NB, SGDC |

---

## 📄 Paper Citation

```bibtex
@article{alzahrani2025lung,
  title={Early Detection of Lung Cancer Using Predictive Modeling 
         Incorporating CTGAN Features and Tree-Based Learning},
  author={Alzahrani, Abdulrahman},
  journal={IEEE Access},
  volume={13},
  pages={34321--34333},
  year={2025},
  publisher={IEEE},
  doi={10.1109/ACCESS.2025.3543215}
}
```

---

## 🗃️ Dataset

- **Source:** [Kaggle Lung Cancer Dataset](https://huggingface.co/datasets/nateraw/lung-cancer)
- **Instances:** 309
- **Attributes:** 16 (15 predictive + 1 class)
- **Class:** 270 Cancer / 39 Normal (87.4% / 12.6%)

---

## 📃 License

MIT License — Free for academic and research use.

---

*Built for university research · IEEE Access 2025 Implementation*
