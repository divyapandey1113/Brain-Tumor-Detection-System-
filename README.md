

## 📁 Project Structure
.
├── data/
│   ├── training/
│   │   ├── glioma/
│   │   ├── meningioma/
│   │   ├── notumor/
│   │   └── pituitary/
│   └── testing/
│       ├── glioma/
│       ├── meningioma/
│       ├── notumor/
│       └── pituitary/
├── notebooks/
│   └── BrainTumourDetection.ipynb
├── src/
│   ├── data_loader.py
│   ├── models.py
│   └── utils.py
├── requirements.txt
└── README.md
```

---

## 🛠️ Requirements

- Python ≥ 3.8  
- TensorFlow ≥ 2.x  
- scikit‑learn  
- OpenCV  
- Matplotlib, Seaborn  
- (Optional) Jupyter Notebook  

Install via:

```bash
pip install -r requirements.txt
```

---

## ⚙️ Data Preparation

1. Download the MRI dataset and organize into `data/training/<class>` and `data/testing/<class>`.  
2. Each image is resized to **128×128** (or **299×299** for Xception).  
3. Labels are one‑hot encoded for four classes:  
   - glioma  
   - meningioma  
   - notumor  
   - pituitary  

---

## 🧠 Model Architectures & Performance

| Model                               | Train epochs    | Train Acc.  | Train Loss | Val. Acc. | Val. Loss | Notes                                                 |
|-------------------------------------|-----------------|-------------|------------|-----------|-----------|-------------------------------------------------------|
| **1. Custom CNN**                   | 150             | ~70.2 %     | 0.7116     | 69.5 %    | 0.7504    | 2 Conv→Pool→BN layers, Flatten + Dense                |
| **2. Enhanced CNN**                 | 50 + callbacks  | –           | –          | –         | –         | L₂ regularization, Dropout; test accuracy ~65 %¹      |
| **3. Xception (frozen)**            | 30              | ~84.3 %     | ~0.52      | 74.7 %    | 0.6885    | ImageNet‑pretrained base; head = Pool→Dense→Softmax   |
| **4. ResNet50 (frozen)**            | 30 + callbacks  | ~88.0 %     | ~0.31      | 87.0 %    | 0.30²     | ImageNet‑pretrained base with Dense head             |
| **5. ResNet50/Xception (fine‑tuned)** | 20            | 99.82%      | 0.0058     | 97.19%    | 96.07     | Fine‑tuned last layers; Precision/Recall included ⁽³⁾ |

¹ Classification report weighted‑avg F1 ≈ 0.63  
² Weighted‑avg F1 ≈ 0.87  
³  
```
accuracy:     0.9982  
loss:         0.0058  
precision:    0.9982  
recall:       0.9982  

val_accuracy: 0.9719  
val_loss:     0.0967  
val_precision: 0.9736  
val_recall:    0.9719  
```

---

## 🚀 How to Run

1. **Data Loading & Preprocessing**  
   ```python
   from src.data_loader import load_dataset
   train_ds, val_ds = load_dataset("data/", img_size=(128,128))
   ```
2. **Train a Model**  
   ```bash
   python src/train.py --model simple_cnn --epochs 150
   ```
3. **Evaluate & Visualize**  
   ```bash
   python src/evaluate.py --model resnet_finetune
   ```
4. **Jupyter Notebook**  
   Open `notebooks/BrainTumourDetection.ipynb` for step‑by‑step code, plots, & confusion matrices.

---

## 🔍 Results & Discussion

- **Custom CNN** gave a solid baseline (~70 % accuracy) but struggled on some classes.  
- **Regularization + Dropout** helped reduce overfitting but did not boost overall accuracy significantly on this dataset.  
- **Pre‑trained Xception** accelerated convergence and improved validation accuracy (~75 %).  
- **Pre‑trained ResNet50** further improved accuracy (~87 %) thanks to its deeper architecture.  
- **Fine‑tuning** the final model pushed training accuracy to ~99.8 % with a strong validation score (~97.2 %).

---

**Feel free to open issues or pull requests if you run into any problems or have suggestions!**
