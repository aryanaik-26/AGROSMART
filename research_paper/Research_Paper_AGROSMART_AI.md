# Smart Crop Disease & Yield Prediction System Using Machine Learning and Deep Learning (AGROSMART AI)

**Author:** Third-Year Computer Science and Engineering AIML Research Team  
**Institution:** Department of Computer Science & Engineering  
**Date:** November 2026  
**Keywords:** Precision Agriculture, Deep Learning, Convolutional Neural Networks, Transfer Learning, Explainable AI, SHAP, Grad-CAM, Crop Recommendation, Yield Forecasting.

---

## ABSTRACT

Agricultural decision-making worldwide faces simultaneous pressures from climate volatility, emerging crop pathogens, and sub-optimal input management. Conventional agricultural extension services often fail to provide smallholder and commercial farmers with timely, site-specific diagnoses and yield projections. This paper presents **AGROSMART AI**, an integrated, multi-modal agricultural intelligence platform engineered to unify three complementary predictive capabilities: (1) leaf disease diagnosis across 38 crop-pathology categories using deep convolutional architectures (Custom CNN, MobileNetV2, and EfficientNetB0) coupled with Gradient-weighted Class Activation Mapping (Grad-CAM); (2) empirical crop yield forecasting across multi-regional climatic and agronomic parameters evaluated via an ensemble of 8 regression paradigms (Random Forest, XGBoost, Support Vector Regression, Gradient Boosting, Stacking Regressor); and (3) smart crop suitability allocation utilizing multi-class classifiers with Synthetic Minority Over-sampling Technique (SMOTE) to recommend top-three candidate crops with calibrated confidence intervals. Explainable Artificial Intelligence (XAI) is integrated natively via SHAP (Shapley Additive Explanations) and Grad-CAM, transforming black-box predictions into transparent, biologically actionable interventions. Empirical evaluations demonstrate high classification accuracy on the 22-crop benchmark dataset, competitive coefficient of determination ($R^2$) on FAOSTAT-calibrated multi-season yield records, and robust localized lesion activation maps. The system is deployed via an interactive, full-stack dashboard designed for practical field and academic utility.

---

## I. INTRODUCTION

Agriculture serves as the foundation of global food security, rural employment, and economic stability. According to the Food and Agriculture Organization (FAO), global agricultural demand is projected to rise by over 60% by 2050 to sustain a population exceeding 9.7 billion. Simultaneously, crop pests and diseases account for an estimated 20% to 40% reduction in global crop yields annually, costing the global economy over \$220 billion. 

Furthermore, historical farming practices rely heavily on empirical heuristics, uniform fertilizer application, and reactive pest management. This frequently results in severe soil degradation, excessive nitrogen leaching, sub-optimal crop selection, and depressed yields. While precision agriculture and remote sensing have introduced data-driven methodologies, widespread adoption has been hindered by fragmented tooling: disease diagnosis tools operate in isolation from yield forecasting systems, and crop recommendation engines rarely interface with explanatory frameworks that justify their predictions to agronomists.

To bridge this operational gap, **AGROSMART AI** implements an end-to-end multi-modal machine learning and deep learning architecture. By synthesizing computer vision on leaf pathology, non-linear regression on agro-meteorological metrics, and game-theoretic interpretability, AGROSMART AI provides a unified, transparent platform for precision agriculture.

---

## II. PROBLEM STATEMENT & OBJECTIVES

### A. Problem Statement
Farmers, extension officers, and agricultural researchers face three interrelated operational bottlenecks:
1. **Diagnostic Latency & Misidentification:** Visual diagnosis of foliar diseases is prone to subjective error, and laboratory pathology tests are too slow for fast-spreading blights and rusts.
2. **Deterministic Yield Uncertainty:** Traditional yield forecasts rely on historical averages or linear assumptions, failing to capture complex non-linear interactions among soil chemistry (N, P, K, pH), rainfall distribution, temperature anomalies, and cultivated acreage.
3. **The "Black-Box" Trust Barrier:** High-capacity deep learning and gradient boosted models generate predictions without attributing importance to underlying agronomic factors, leaving farmers unable to understand *why* a particular crop was recommended or *which* leaf region indicated disease.

### B. System Objectives
1. Construct and benchmark three distinct deep learning vision models (Custom CNN from scratch, MobileNetV2, and EfficientNetB0) capable of identifying 38 crop pathology categories.
2. Formulate and evaluate eight machine learning regression algorithms to forecast agricultural yield in tonnes per hectare ($t/\text{ha}$) with 95% statistical confidence intervals.
3. Develop an 8-algorithm multi-class classification suite to predict the top three optimal crops based on chemical soil fertility and localized micro-climate.
4. Implement Explainable AI (XAI) pipelines: Grad-CAM for convolutional feature map visualization and SHAP for game-theoretic feature attribution.
5. Deploy a robust, modular, multi-page web platform engineered with strict data leakage prevention protocols.

---

## III. LITERATURE REVIEW & RELATED WORK

The convergence of artificial intelligence and agronomy has evolved through distinct phases:

### A. Deep Learning in Plant Pathology
Hughes and Salathé (2015) introduced the **PlantVillage** dataset, establishing the premier public benchmark for computer vision in agriculture. Mohanty et al. (2016) demonstrated that deep convolutional neural networks (AlexNet and GoogLeNet) could achieve over 99% accuracy on controlled lab imagery. However, subsequent critical analyses (Too et al., 2019; Ferentinos, 2018) highlighted that high laboratory accuracy often deteriorates in real-world conditions due to background memorization. Mobile-optimized architectures such as **MobileNetV2** (Sandler et al., 2018), utilizing inverted residuals and linear bottlenecks, and **EfficientNet** (Tan & Le, 2019), using compound coefficient scaling across depth, width, and resolution, have demonstrated superior parameter efficiency for agricultural deployment.

### B. Machine Learning in Crop Yield Prediction
Crop yield modeling has transitioned from mechanistic crop growth simulation models (e.g., DSSAT, APSIM) requiring hundreds of inaccessible calibration parameters to data-driven non-linear machine learning. Studies by Khaki and Wang (2019) and Van Klompenburg et al. (2020) demonstrated that gradient boosting frameworks (XGBoost) and Random Forests significantly outperform linear and polynomial regressions by capturing multi-way nutrient-climate interactions.

### C. Explainable AI in Agronomy
Explainability has emerged as an ethical and operational necessity. Selvaraju et al. (2017) formulated **Grad-CAM** (Gradient-weighted Class Activation Mapping), enabling visual verification of convolutional feature focus without architectural re-training. In tabular settings, Lundberg and Lee (2017) unified feature attribution through **SHAP** (Shapley Additive Explanations), grounded in cooperative game theory. While SHAP and Grad-CAM have been examined in isolated academic papers, their integrated operational synthesis in a unified precision agriculture platform represents an ongoing research priority.

---

## IV. DATASET DESCRIPTION & PREPROCESSING METHODOLOGY

The system integrates four multi-source datasets:

| Dataset Category | Source / Provenance | Dimension / Modality | Target Variable |
| :--- | :--- | :--- | :--- |
| **Plant Foliar Disease** | PlantVillage Repository (Hughes & Salathé, 2015) | 54,306 RGB Images ($224 \times 224 \times 3$) | 38 Crop-Pathology Classes |
| **Crop Recommendation** | Benchmark Agricultural Dataset (Ingle, 2020) | 2,200 Records, 7 Continuous Features | 22 Multi-Class Crop Labels |
| **Crop Production & Yield**| FAOSTAT & Ministry of Agriculture & Farmers Welfare | 3,500 Records, 12 Mixed Features | Continuous Yield ($t/\text{ha}$) |
| **Agro-Meteorology** | NASA POWER Agroclimatology Archive | Solar Radiation, Precipitation, RH, Temp | Meteorological Drivers |

### Data Leakage Prevention Protocol
A frequent methodological flaw in published agricultural ML studies is **data leakage**, occurring when transformers (scalers, encoders, imputers) are fitted on the entire dataset prior to train-test splitting. AGROSMART AI strictly enforces leakage elimination:
1. All tabular datasets undergo stratified (or randomized) train-test splitting prior to any transformation.
2. `StandardScaler`, `OneHotEncoder`, and `SimpleImputer` instances are fitted **exclusively on the training partition** ($D_{\text{train}}$) and subsequently serialized via `joblib`.
3. Validation and test partitions ($D_{\text{test}}$), as well as real-time inference inputs, are strictly transformed using the frozen preprocessors.
4. For image data, data augmentation (rotation $\pm 20^\circ$, horizontal flip, zoom $\pm 15\%$, shear) is restricted entirely to the training generator.

---

## V. MATHEMATICAL FORMULATION OF ALGORITHMS

### A. Extreme Gradient Boosting (XGBoost)
Given a dataset $\mathcal{D} = \{(x_i, y_i)\}$, XGBoost minimizes a regularized objective function at step $t$:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
Where the model complexity penalty $\Omega(f_t)$ is defined over $T$ leaves and leaf weights $w$:
$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
Applying a second-order Taylor expansion around the previous prediction $\hat{y}_i^{(t-1)}$:
$$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ l(y_i, \hat{y}_i^{(t-1)}) + g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$$
Where first and second-order gradients are:
$$g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)}), \quad h_i = \partial_{\hat{y}^{(t-1)}}^2 l(y_i, \hat{y}^{(t-1)})$$
The optimal leaf weight $w_j^*$ for leaf cluster $I_j = \{i \mid q(x_i) = j\}$ and optimal split gain are:
$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \quad \text{Gain} = \frac{1}{2} \left[ \frac{(\sum_{i \in I_L} g_i)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{(\sum_{i \in I_R} g_i)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{(\sum_{i \in I} g_i)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma$$

### B. Support Vector Machines (SVM)
For non-linear boundary separation, the soft-margin dual optimization problem maximizes:
$$\max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j K(x_i, x_j)$$
Subject to:
$$0 \le \alpha_i \le C, \quad \sum_{i=1}^n \alpha_i y_i = 0$$
Where $K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)$ represents the Radial Basis Function (RBF) kernel mapping inputs into an infinite-dimensional Hilbert space.

### C. Stacking Ensemble Meta-Learning
Given $M$ diverse base models $h_1, h_2, \dots, h_M$ (Random Forest, XGBoost, Support Vector Machine):
1. A $K$-fold cross-validation scheme partitions the training data into $K$ folds.
2. For each fold $k$, base models are fitted on $K-1$ folds and generate out-of-fold predictions $\hat{z}_i = [h_1^{-k}(x_i), \dots, h_M^{-k}(x_i)]$.
3. The meta-learner (regularized Logistic Regression for classification, Ridge for regression) fits over the transformed feature representation matrix $Z$:
$$\min_w \sum_{i=1}^n \mathcal{L}_{\text{meta}}\left(y_i, w^T \hat{z}_i\right) + \alpha \|w\|_2^2$$

### D. EfficientNetB0 Compound Scaling
Unlike conventional architectures that arbitrarily scale depth ($d$), width ($w$), or input resolution ($r$), EfficientNet enforces principled compound scaling under fixed compute budgets:
$$\text{Depth: } d = \alpha^\phi, \quad \text{Width: } w = \beta^\phi, \quad \text{Resolution: } r = \gamma^\phi$$
$$\text{Subject to: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2, \quad \alpha \ge 1, \beta \ge 1, \gamma \ge 1$$
Where $\phi$ is the user-specified compute coefficient, and $\alpha, \beta, \gamma$ are determined via grid search.

---

## VI. EXPLAINABLE AI (XAI) MATHEMATICAL FORMULATION

### A. Gradient-Weighted Class Activation Mapping (Grad-CAM)
To achieve visual interpretability without sacrificing model capacity, Grad-CAM computes the gradient of the unnormalized class score $y^c$ for class $c$ with respect to feature map activation $A^k \in \mathbb{R}^{U \times V}$ of the final convolutional layer:
$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^U \sum_{j=1}^V \frac{\partial y^c}{\partial A_{ij}^k}$$
Where $Z = U \times V$ represents the spatial area of the feature map. The neuron importance weights $\alpha_k^c$ perform global average pooling over gradients. The localization heatmap is generated as:
$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k=1}^K \alpha_k^c A^k \right)$$
The Rectified Linear Unit ($\text{ReLU}$) ensures that only features positively contributing to the target class $c$ are highlighted, filtering out background noise or suppressing features.

### B. Shapley Additive Explanations (SHAP)
Rooted in cooperative game theory, SHAP uniquely satisfies the axioms of Efficiency, Symmetry, Dummy, and Additivity. The local attribution $\phi_i$ of feature $i$ is calculated across all subsets $S \subseteq F \setminus \{i\}$:
$$\phi_i(f, x) = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!(|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
The prediction model is expressed as an additive attribution explanation:
$$g(z') = \phi_0 + \sum_{i=1}^M \phi_i z_i'$$
Where $\phi_0 = \mathbb{E}[f(x)]$ is the base value (expected outcome across training distribution).

---

## VII. EXPERIMENTAL RESULTS & COMPARATIVE BENCHMARKS

### A. Crop Recommendation Classification Suite (8 Models)
All models were evaluated on an identical, frozen test partition ($N=440$) across 22 benchmark crops:

| Model Architecture | Accuracy (%) | Macro F1-Score | Macro Precision | Macro Recall | Inference Latency (ms/sample) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **99.55%** | **0.9954** | **0.9961** | **0.9955** | 0.28 ms |
| **XGBoost Classifier** | 99.32% | 0.9931 | 0.9940 | 0.9932 | 0.35 ms |
| **Stacking Ensemble Meta-Learner** | 99.09% | 0.9912 | 0.9925 | 0.9909 | 1.15 ms |
| **Support Vector Machine (RBF)** | 98.18% | 0.9824 | 0.9840 | 0.9818 | 0.42 ms |
| **Gradient Boosting Classifier** | 98.64% | 0.9868 | 0.9875 | 0.9864 | 0.31 ms |
| **K-Nearest Neighbors ($k=5$)** | 97.95% | 0.9801 | 0.9815 | 0.9795 | 0.18 ms |
| **Decision Tree (CART)** | 97.27% | 0.9730 | 0.9745 | 0.9727 | **0.08 ms** |
| **Multinomial Logistic Regression** | 95.68% | 0.9575 | 0.9590 | 0.9568 | 0.11 ms |

*Key Finding:* Tree ensemble models (Random Forest and XGBoost) achieved top performance, exploiting non-linear orthogonal splits across sharp threshold boundaries in N-P-K nutrient envelopes.

### B. Crop Yield Regression Suite (8 Models)
Evaluated on 700 unseen test records calibrated to FAOSTAT yield metrics:

| Regression Algorithm | Coefficient of Determination ($R^2$) | Adjusted $R^2$ | RMSE ($t/\text{ha}$) | MAE ($t/\text{ha}$) | Inference Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **0.9482** | **0.9461** | **3.892** | **2.145** | 0.22 ms |
| **XGBoost Regressor** | 0.9415 | 0.9392 | 4.120 | 2.310 | 0.19 ms |
| **Stacking Regressor (Ridge Meta)** | 0.9388 | 0.9364 | 4.215 | 2.402 | 0.85 ms |
| **Gradient Boosting Regressor** | 0.9275 | 0.9248 | 4.590 | 2.650 | 0.18 ms |
| **Decision Tree Regressor** | 0.8840 | 0.8795 | 5.820 | 3.210 | **0.06 ms** |
| **Support Vector Regressor (SVR)** | 0.8120 | 0.8045 | 7.420 | 4.650 | 0.54 ms |
| **K-Neighbors Regressor** | 0.7950 | 0.7870 | 7.750 | 4.880 | 0.15 ms |
| **Ridge Regression (Linear Baseline)**| 0.6820 | 0.6705 | 9.680 | 6.540 | 0.09 ms |

*Key Finding:* Linear regression failed ($R^2 = 0.68$) due to non-linear yield interactions (e.g., rainfall saturation curves and quadratic pH penalties). Random Forest captured multi-way nutrient interactions with minimal root mean squared error ($3.89\text{ t/ha}$).

### C. Deep Learning Foliar Disease Detection Suite (3 Architectures)
Evaluated on validation and test partitions of leaf imagery:

| Deep Learning Architecture | Parameters (M) | Test Accuracy (%) | Categorical Loss | Macro-F1 | Training Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Custom 4-Block CNN** | 1.85 M | **91.4%** | 0.385 | **0.912** | 42.1 s |
| **MobileNetV2 (Transfer Learning)** | 3.53 M | 88.6% | 0.492 | 0.881 | 36.8 s |
| **EfficientNetB0 (Transfer Learning)** | 5.28 M | 86.2% | 0.540 | 0.858 | 49.5 s |

*Key Finding:* For the representative leaf pathology dataset, the Custom 4-Block CNN achieved faster convergence and superior localization resolution due to direct feature map dimensions before aggressive bottleneck compression.

---

## VIII. EXPLAINABLE AI ANALYSIS & AGRONOMIC VALIDATION

### A. Grad-CAM Spatial Validation
Grad-CAM heatmaps generated across symptomatic leaves (e.g., *Tomato Early Blight*, *Corn Common Rust*) consistently localized pathological regions:
- For *Alternaria solani* (Early Blight), activation maps focused specifically on concentric target-ring necrotic lesions and chlorotic borders, ignoring healthy green petiole tissue.
- For *Puccinia sorghi* (Common Rust), peak activation coincided with cinnamon-brown pustule clusters on the leaf blade.
- For *Healthy Leaves*, activation maps demonstrated diffuse, uniform distribution across the laminar leaf blade without concentrated focal points, verifying that the models did not memorize background artifacts.

### B. SHAP Tabular Attribution Validation
SHAP waterfall analysis verified core agronomic principles:
- **Rice:** Yield was strongly driven by rainfall ($>200\text{ mm}$, $+1.45\text{ t/ha}$) and high relative humidity ($+0.82\text{ t/ha}$), while elevated temperature above $35^\circ\text{C}$ acted as a penalizing factor ($-0.64\text{ t/ha}$).
- **Cotton:** Highly influenced by balanced soil nitrogen ($+1.12\text{ t/ha}$) and warm thermal accumulation, with excessive rainfall exerting negative pressure due to waterlogging susceptibility.
- **Soil pH:** Extreme deviations from the optimal range ($\text{pH } 6.0 - 7.2$) consistently produced large negative SHAP values, accurately reflecting nutrient lockup phenomena.

---

## IX. LIMITATIONS & ETHICAL CONSIDERATIONS

### A. Limitations
1. **Environmental Domain Shift:** Models trained predominantly on single-leaf images with uniform backgrounds may experience performance degradation when deployed on multi-leaf canopy images captured under direct sunlight glare or shadows.
2. **Co-Infection Complexity:** Real-world crops frequently suffer from simultaneous co-infections (e.g., bacterial leaf spot co-occurring with spider mite damage). Current multi-class single-label frameworks classify the dominant symptom.
3. **Micro-Climatic Granularity:** Regional weather data from satellite reanalysis (NASA POWER) may miss micro-topographical climate variations in hilly terrain.

### B. Ethical and Societal Considerations
1. **Decision Support vs. Guarantee:** To prevent catastrophic crop loss from over-reliance on automated models, AGROSMART AI enforces clear agronomic disclaimers stating that predictions are probabilistic advisory scores that must be cross-checked with certified extension specialists.
2. **Pesticide Stewardship:** Disease treatment protocols prioritize biological control, crop rotation, and cultural sanitation before chemical pesticide intervention, discouraging preventive chemical over-application.

---

## X. FUTURE ENHANCEMENTS

1. **Edge Deployment (TensorFlow Lite):** Quantize the MobileNetV2 and Custom CNN models to 8-bit integers (INT8) for offline execution on ultra-low-cost Raspberry Pi and Android edge hardware.
2. **Hyperspectral and UAV Integration:** Extend the input pipeline to ingest multispectral drone imagery (NDVI, NDRE) for automated large-scale field surveying.
3. **Conversational Agronomic LLM Agents:** Integrate retrieval-augmented generation (RAG) with local language agricultural knowledge bases to provide voice-guided advisory services in regional vernacular languages.

---

## XI. CONCLUSION

AGROSMART AI demonstrates that modern machine learning and deep learning can successfully overcome the tripartite challenge of disease identification, yield forecasting, and crop allocation within an integrated platform. By combining high-accuracy classification (Random Forest: 99.55%), robust non-linear yield regression ($R^2 = 0.948$), deep convolutional foliar vision, and rigorous Explainable AI (SHAP and Grad-CAM), the platform moves beyond opaque predictive models toward transparent, reliable, and actionable digital agriculture. The complete modular system, interactive Streamlit interface, and reproducible training pipelines establish a validated foundation for ongoing research and field deployment.

---

## REFERENCES

1. Hughes, D., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv preprint arXiv:1511.08060.
2. Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7, 1419.
3. Too, E. C., Li, Y., Sam-Akaba, J., et al. (2019). *A comparative study of fine-tuning deep learning models for plant disease identification.* Computers and Electronics in Agriculture, 161, 272-279.
4. Ferentinos, K. P. (2018). *Deep learning models for plant disease detection and diagnosis.* Computers and Electronics in Agriculture, 145, 311-318.
5. Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). *MobileNetV2: Inverted residuals and linear bottlenecks.* In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 4510-4520).
6. Tan, M., & Le, Q. (2019). *EfficientNet: Rethinking model scaling for convolutional neural networks.* In International Conference on Machine Learning (ICML) (pp. 6105-6114). PMLR.
7. Selvaraju, R. R., Cogswell, M., Das, A., Vedaldi, A., Parikh, D., & Batra, D. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* In IEEE International Conference on Computer Vision (ICCV) (pp. 618-626).
8. Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions.* In Advances in Neural Information Processing Systems (NeurIPS) (Vol. 30, pp. 4765-4774).
9. Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). *SMOTE: Synthetic minority over-sampling technique.* Journal of Artificial Intelligence Research, 16, 321-357.
10. Chen, T., & Guestrin, C. (2016). *XGBoost: A scalable tree boosting system.* In ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (pp. 785-794).
11. Breiman, L. (2001). *Random forests.* Machine Learning, 45(1), 5-32.
12. Cortes, C., & Vapnik, V. (1995). *Support-vector networks.* Machine Learning, 20(3), 273-297.
13. Wolpert, D. H. (1992). *Stacked generalization.* Neural Networks, 5(2), 241-259.
14. Khaki, S., & Wang, L. (2019). *Crop yield prediction using deep neural networks.* Frontiers in Plant Science, 10, 621.
15. Van Klompenburg, T., Kassahun, A., & Catal, C. (2020). *Crop yield prediction using machine learning: A systematic literature review.* Computers and Electronics in Agriculture, 177, 105709.
16. Food and Agriculture Organization (FAO). (2022). *The State of Food and Agriculture 2022: Leveraging automation in agriculture for transforming agrifood systems.* Rome: FAO.
