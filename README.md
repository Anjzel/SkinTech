# SkinTech: Real-Time Facial Skin Analysis and Skincare Recommendation System

SkinTech is a web-based application developed for academic purposes. It performs real-time facial skin analysis using deep learning and recommends personalized skincare products based on the detected skin type and sensitivity. The system aims to assist users in identifying their skin condition and selecting products that suit their skin needs.

---

## Features
- Upload Image
- Real-time face capture using webcam
- Skin type prediction using EfficientNetB2
  - Supports Oily, Dry, Combination, and Normal skin types
- Sensitivity detection (Sensitive vs. Non-Sensitive skin)
- Personalized skincare product recommendations
  - Based on predicted skin type and prioritized ingredients
  - Filters out ingredients suitable for sensitive skin
- PDF export of analysis results and recommendations

---

## Technologies Used

- **Frontend:** HTML, CSS, Bootstrap
- **Backend:** Python, Flask
- **Machine Learning:** TensorFlow, Keras (EfficientNetB2)
- **Face Detection:** TinyFaces (for real-time detection)
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib
- **PDF Generation:** ReportLab or FPDF
- **Deployment:** Google Cloud App Engine / DigitalOcean (development use)

---

## How It Works

1. The user allows upload image of face or webcam access to capture face images from left, center, and right angles.
2. The EfficientNetB2 model predicts the skin type for each image.
3. A secondary model determines whether the skin is sensitive.
4. Based on the results, the system recommends appropriate skincare products.
5. Users can export the results and product list to a PDF report, which includes instructions and images.


