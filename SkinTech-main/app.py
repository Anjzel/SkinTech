from flask import Flask, render_template, request, redirect, url_for, flash, session, Response, send_from_directory
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import base64
import re
from io import BytesIO
import pandas as pd
import random
import requests
import io
import cv2
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

import google.generativeai as genai

# 🔑 Replace this with your Google API key
GENAI_API_KEY = "AIzaSyCH6StzM1z1eKNjthj24JagDpvx7CgiRE4"  

genai.configure(api_key=GENAI_API_KEY)


# Register custom objects (if any)
@tf.keras.utils.register_keras_serializable()
def custom_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

app = Flask(__name__, template_folder='main')
app.secret_key = os.urandom(24)

MODEL_PATH = "models/(4)efficientnet_final_trained.keras"
SENSITIVEMODEL = "models/efficientnet_model.keras"
try:
    model = load_model(MODEL_PATH, custom_objects={'loss': custom_loss})
    sensitive_model = load_model(SENSITIVEMODEL, custom_objects={'loss': custom_loss})  # Load the sensitivity model
except Exception as e:
    raise RuntimeError(f"Failed to load models: {e}")

# Load the face detection cascade classifier
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception as e:
    raise RuntimeError(f"Failed to load face cascade classifier: {e}")

CLOUD_CSV_URL = "https://drive.google.com/uc?export=download&id=1zHeYAs6RvPFZAW5C1LvFmCYbcz9dUL_w"


def load_recommendations():
    try:
        response = requests.get(CLOUD_CSV_URL)
        response.raise_for_status()
        recommendations_df = pd.read_csv(io.StringIO(response.text))
        return recommendations_df
    except Exception as e:
        raise RuntimeError(f"Failed to load recommendations from cloud CSV: {e}")

try:
    recommendations_df = load_recommendations()
except RuntimeError as e:
    print(e)

# Expanded ingredient dictionary
skin_type_ingredients = {
    "Oily": [
        "salicylic acid", "niacinamide", "zinc gluconate", "kaolin", "monolaurin", 
        "benzoyl peroxide", "tea tree oil", "witch hazel", "retinol", 
        "glycolic acid", "surfactants", "glycerin", "thermal spring water"
    ],
    "Dry": [
        "hyaluronic acid", "glycerin", "shea butter", "ceramides", "jojoba oil", 
        "urea", "sunflower oil", "omega fatty acids", "rhealba oat extract", 
        "thermal spring water", "avocado oil", "vitamin e", "paraffin", 
        "argan oil", "filaxerine", "zinc oxide", "copper sulfate", "aloe vera"
    ],
    "Normal": [
        "vitamin c", "peptides", "aloe vera", "green tea extract", 
        "squalane", "thermal spring water", "hyaluronic acid", "glycerin", 
        "vitamin e", "rhealba oat extract", "retinaldehyde", "micelles", 
        "zinc oxide", "copper sulfate", "red fruit extract", "vitamin b5"
    ],
    "Combination": [
        "glycolic acid", "lactic acid", "vitamin b5", "rosehip oil", 
        "witch hazel", "retinaldehyde", "thermal spring water", "glycerin", 
        "avocado oil", "titanium dioxide", "silica", "hyaluronic acid"
    ],
    "Sensitive": [
        "chamomile", "allantoin", "centella asiatica", "colloidal oatmeal", 
        "panthenol", "avocado oil", "thermal spring water", "aloe vera", 
        "glycerin", "rhealba oat extract", "zinc oxide", "copper sulfate", 
        "vitamin e", "licorice extract", "dextran sulfate", "imodium"
    ],
    "Non-Sensitive": [
        "retinol", "glycolic acid", "benzoyl peroxide", "salicylic acid", 
        "fragrance", "lactic acid", "rosehip oil", "jojoba oil", "ceramides", 
        "urea", "niacinamide", "kaolin", "witch hazel", "silicone", 
        "beeswax", "mineral oil"
    ]
}



IMAGE_SIZE = (260, 260)

skin_types = ["dry", "oily", "combination", "normal"]
sensitivity_types = ["sensitive", "non-sensitive"]


def detect_and_crop_face(image):
    """Detect face in image and crop it before resizing"""
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Convert RGB to BGR (OpenCV uses BGR format)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # Adjust scaleFactor for better detection
        minNeighbors=5,  # Increase minNeighbors to reduce false positives
        minSize=(30, 30)  # Minimum face size to detect
    )

    # If no faces are detected, return the resized original image
    if len(faces) == 0:
        print("No face detected, resizing full image.")
        return image.resize(IMAGE_SIZE)

    # Get the largest face detected (based on area)
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # Ensure the bounding box is within the image dimensions
    x1, y1, x2, y2 = max(0, x), max(0, y), min(img_array.shape[1], x + w), min(img_array.shape[0], y + h)

    # Crop the face region from the image
    face_img = img_array[y1:y2, x1:x2]

    # Convert the cropped face back to a PIL image
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

    # Resize the cropped face to the required size for the model
    return face_pil.resize(IMAGE_SIZE)

def detect_face_upload(image):
    """Detect face in image and crop it before resizing."""
    # Convert PIL image to numpy array
    img_array = np.array(image)

    # Convert RGB to BGR for OpenCV
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Detect faces using the Haar Cascade classifier
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # ✅ Return None if no faces are detected
    if len(faces) == 0:
        print("No face detected.")  # For debugging
        return None

    # Get the largest face detected
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # Crop the face region
    face_img = img_array[y:y + h, x:x + w]

    # Convert the cropped face back to PIL format
    face_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

    # Resize the face to the model's required input size
    return face_pil.resize(IMAGE_SIZE)



def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Detect and crop face
    face_image = detect_and_crop_face(image)

    # Convert to numpy array and normalize
    image_array = np.array(face_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def interpret_prediction(prediction):
    predicted_class_index = np.argmax(prediction, axis=1)[0] if prediction.ndim > 1 else prediction
    return skin_types[predicted_class_index]


def majority_vote(predictions):
    vote_counts = {skin_type: 0 for skin_type in skin_types}
    for pred in predictions:
        vote_counts[pred] += 1
    sorted_votes = sorted(vote_counts.items(), key=lambda x: x[1], reverse=True)
    if sorted_votes[0][1] > 1:  # If there is a majority
        return sorted_votes[0][0]
    return predictions[1]  # Default to center image prediction


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help_page():
    return render_template('help.html')

@app.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')



def get_gemini_analysis(skin_type):
    """Uses Gemini API to analyze the predicted skin type and provide skincare advice."""
    prompt = (
        f"The user's skin type is {skin_type}. "
        f"Provide a concise analysis of {skin_type} skin, including typical characteristics "
        f"(for example, if dry skin: flakiness, tightness, rough patches). "
        f"Then provide specific skincare recommendations for {skin_type} skin type. "
        f"Format your response as plain text without any markdown formatting like asterisks. "
        f"Focus on practical advice without disclaimers or general recommendations to consult professionals."
    )

    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content([prompt])

    # Get the response text
    text = response.text.strip() if response.text else "⚠️ No analysis received. Try again."

    # Clean up any remaining markdown formatting
    cleaned_text = text.replace("**", "").replace("*", "")

    return cleaned_text


@app.route('/results', methods=['POST'])
def results():
    image_data_urls = {
        'left': request.form.get('image_left'),
        'center': request.form.get('image_center'),
        'right': request.form.get('image_right')
    }

    uploaded_file = request.files.get('image')

    if not uploaded_file and not all(image_data_urls.values()):
        flash("Please upload an image or capture all three images.")
        return redirect(url_for('index'))

    predictions = []
    sensitivity_predictions = []
    image_path = 'static/images'
    os.makedirs(image_path, exist_ok=True)

    try:
        if uploaded_file:
            filename = 'uploaded_image.png'
            filepath = os.path.join(image_path, filename)
            uploaded_file.save(filepath)

            # ✅ Open the uploaded image
            img = Image.open(filepath)
                    # 🔥 Face detection
            detected_face = detect_face_upload(img)

            # ✅ Flash message when no face is detected
            if detected_face is None:
                flash("⚠️ No face detected in the uploaded image. Please try again.")
                return redirect(url_for('index'))

            # 🔥 Face detection for upload
            cropped_img = detect_face_upload(img)

            if cropped_img.size[0] < 30 or cropped_img.size[1] < 30:  # No face detected
                flash("❌ Invalid image: No face detected.")
                return redirect(url_for('index'))

            # Save the cropped image
            cropped_filepath = os.path.join(image_path, 'cropped_uploaded_image.png')
            cropped_img.save(cropped_filepath)

            # Preprocess the cropped image
            processed_img = preprocess_image(cropped_img)


            # ✅ Predict skin type
            prediction = model.predict(processed_img)
            final_skin_type = interpret_prediction(prediction)

            # ✅ Predict sensitivity & Debugging
            sensitivity_prediction = sensitive_model.predict(processed_img)
            print(f"DEBUG: Raw Sensitivity Prediction -> {sensitivity_prediction}")

            is_sensitive = "Sensitive" if sensitivity_prediction > 0.5 else "Non-Sensitive"
            print(f"DEBUG: Sensitivity Classification -> {is_sensitive}")

            session['skin_type'] = final_skin_type
            session['sensitivity'] = is_sensitive if is_sensitive else "Not Detected"
            session['image_filenames'] = {
                'cropped_uploaded': 'cropped_uploaded_image.png'
            }

        else:
            for position, image_data_url in image_data_urls.items():
                if not image_data_url:
                    continue

                image_data = re.sub('^data:image/.+;base64,', '', image_data_url)
                image_data = base64.b64decode(image_data)
                filename = f'{position}_image.png'
                filepath = os.path.join(image_path, filename)

                with open(filepath, 'wb') as f:
                    f.write(image_data)

                img = Image.open(BytesIO(image_data))

                # Keep existing logic for captured images (no face detection)
                processed_img = preprocess_image(img)

                # ✅ Predict skin type
                prediction = model.predict(processed_img)
                predictions.append(interpret_prediction(prediction))

                # ✅ Predict sensitivity & Debugging
                sensitivity_prediction = sensitive_model.predict(processed_img)
                print(f"DEBUG: Raw Sensitivity Prediction for {position} -> {sensitivity_prediction}")

                sensitivity_predictions.append(sensitivity_prediction > 0.5)

            if predictions:
                final_skin_type = majority_vote(predictions)
                is_sensitive = "Sensitive" if sum(sensitivity_predictions) > 1 else "Non-Sensitive"

                session['skin_type'] = final_skin_type
                session['sensitivity'] = is_sensitive if is_sensitive else "Not Detected"
                session['image_filenames'] = {
                    pos: f'{pos}_image.png' for pos in image_data_urls if image_data_urls[pos]
                }
            else:
                flash("No valid images were processed.")
                return redirect(url_for('index'))

        # 🔥 NEW: Get Gemini-generated skin analysis
        skin_analysis = get_gemini_analysis(final_skin_type)
        session['skin_analysis'] = skin_analysis  # Store the analysis

    except Exception as e:
        flash(f"An error occurred during prediction: {e}")
        return redirect(url_for('index'))

    return render_template(
        'index.html',
        skin_type=session.get('skin_type', "Unknown"),
        sensitivity=session.get('sensitivity', "Unknown"),
        skin_analysis=session.get('skin_analysis', "No analysis available.")
    )


def get_recommendations(skin_type, preferences):
    if recommendations_df is None or recommendations_df.empty:
        raise RuntimeError("Recommendations dataset not loaded or is empty.")

    # Normalize column names (lowercase, strip spaces)
    recommendations_df.columns = recommendations_df.columns.str.lower().str.strip()

    # Get key ingredients for the detected skin type
    key_ingredients = skin_type_ingredients.get(skin_type.capitalize(), [])
    
    # Filter by skin type
    filtered_df = recommendations_df[
        recommendations_df['skin_types'].str.lower().str.strip() == skin_type.lower().strip()]

    # If no products match the skin type, return empty list
    if filtered_df.empty:
        return []
    
    # Create a new column that counts how many key ingredients are in each product
    filtered_df['ingredient_match_count'] = filtered_df['ingredients'].apply(
        lambda x: sum(1 for ingredient in key_ingredients if ingredient.lower() in str(x).lower())
    )
    
    # Initialize empty list for final recommendations
    final_recommendations = []
    
    # If user selected preferences, get one product per preference
    if preferences:
        for pref in preferences:
            # Filter products by this specific preference
            pref_df = filtered_df[filtered_df['prodtypes'].str.lower() == pref.lower()]
            
            if not pref_df.empty:
                # Sort by ingredient match count
                pref_df = pref_df.sort_values('ingredient_match_count', ascending=False)
                
                # Take the top 3 products with highest ingredient matches
                top_matches = pref_df.head(min(3, len(pref_df)))
                
                # Randomly select 1 product from the top matches
                selected_product = top_matches.sample(1).to_dict(orient='records')[0]
                
                # Add this product to our recommendations
                final_recommendations.append(selected_product)
    else:
        # If no preferences selected, just pick top 5 products overall
        sorted_df = filtered_df.sort_values('ingredient_match_count', ascending=False)
        final_recommendations = sorted_df.head(min(5, len(sorted_df))).to_dict(orient='records')

    # Clean up and convert benefits string to list for each recommendation
    for product in final_recommendations:
        if isinstance(product.get('benefits'), str):
            product['benefits'] = [b.strip() for b in product['benefits'].split(',') if b.strip()]
    
    return final_recommendations

@app.route('/haarcascade_frontalface_default.xml')
def serve_cascade():
    return send_file(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@app.route('/submit_preferences', methods=['POST'])
def submit_preferences():
    skin_type = session.get('skin_type')
    sensitivity = session.get('sensitivity')
    image_filenames = session.get('image_filenames')
    skin_analysis = session.get('skin_analysis')
    preferences = request.form.getlist('preferences')

    if not skin_type:
        flash("Skin type not found. Please analyze your skin first.")
        return redirect(url_for('index'))

    # Clear old recommendations to force new ones
    session.pop('recommendations', None)

    # Fetch new recommendations based on the latest skin type
    recommendations = get_recommendations(skin_type, preferences)
    session['recommendations'] = recommendations

    return render_template(
        'results.html',
        skin_type=skin_type,
        sensitivity=sensitivity,
        recommendations=recommendations,
        image_filenames=image_filenames,
        skin_analysis=skin_analysis,
        skin_type_ingredients=skin_type_ingredients  # Pass the ingredients dictionary to the template
    )

@app.route('/download_pdf')
def download_pdf():
    recommendations = session.get('recommendations', [])
    skin_analysis = session.get('skin_analysis', "No skin analysis available.")
    skin_type = session.get('skin_type', "Unknown")
    sensitivity = session.get('sensitivity', "Unknown")

    if not recommendations:
        flash("No recommendations available to download.", "warning")
        return redirect(url_for('index'))

    # Get the filename from the request
    custom_filename = request.args.get('filename', '').strip()
    if not custom_filename:
        custom_filename = "Untitled"
    
    # Clean the filename to remove invalid characters
    # Replace invalid characters with underscores
    custom_filename = re.sub(r'[<>:"/\\|?*]', '_', custom_filename)
    
    # Create the full filename
    full_filename = f"{custom_filename}_Facial_Skincare_Report.pdf"

    # Set up PDF with proper margins
    pdf_buffer = BytesIO()
    width, height = letter
    margin = 50  # Increased margin for better spacing
    pdf = canvas.Canvas(pdf_buffer, pagesize=letter)
    pdf.setTitle(f"{custom_filename}'s Skincare Report")

    
    # Count how many pages we'll need (rough estimate)
    total_pages = 1 + len(recommendations) // 3  # Approximately 3 products per page

    # Function to add footer to each page
    def add_footer(page_num):
        footer_y = margin - 30
        pdf.setFont("Helvetica", 8)
        pdf.setFillColorRGB(0.5, 0.5, 0.5)  # Gray for footer
        from datetime import datetime
        today = datetime.now().strftime("%B %d, %Y")
        pdf.drawString(margin, footer_y, f"Generated on: {today}")
        pdf.drawString(width - margin - 100, footer_y, f"Page {page_num} of {total_pages}")
    
    current_page = 1

    # Add decorative elements
    # Header background
    pdf.setFillColorRGB(0.9, 0.9, 0.95)  # Light blue-gray
    pdf.rect(margin, height - 100, width - (2 * margin), 70, fill=1, stroke=0)
    
    # Title with improved styling
    pdf.setFillColorRGB(0.2, 0.2, 0.4)  # Dark blue for title
    pdf.setFont("Helvetica-Bold", 28)  # or try 32
    title_text = f"{custom_filename}'s Skincare Report"
    title_width = pdf.stringWidth(title_text, "Helvetica-Bold", 28)
    pdf.drawString((width - title_width) / 2, height - 60, title_text)


    
    # Add a small decorative line under title
    pdf.setStrokeColorRGB(0.5, 0.5, 0.7)
    pdf.setLineWidth(1.5)
    pdf.line((width - title_width) / 2, height - 65, (width + title_width) / 2, height - 65)

    # Reset colors for main content
    pdf.setFillColorRGB(0, 0, 0)
    
    # Starting position for content
    y_position = height - 120
    line_spacing = 20
    content_width = width - (2 * margin)
    left_margin = margin

    # Section: Skin Type & Sensitivity with section styling
    y_position -= 10  # Extra spacing before first section
    
    # Section header background
    pdf.setFillColorRGB(0.95, 0.95, 1.0)  # Very light blue
    pdf.rect(left_margin - 5, y_position - 5, content_width + 10, 30, fill=1, stroke=0)
    
    pdf.setFillColorRGB(0.2, 0.2, 0.5)  # Dark blue for headers
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y_position, "Skin Type & Sensitivity:")
    y_position -= line_spacing + 5

    pdf.setFillColorRGB(0, 0, 0)  # Back to black for regular text
    pdf.setFont("Helvetica", 12)
    pdf.drawString(left_margin + 10, y_position, f"Detected Skin Type: {skin_type}")
    y_position -= line_spacing
    pdf.drawString(left_margin + 10, y_position, f"Skin Sensitivity: {sensitivity}")
    y_position -= line_spacing * 1.5

    # Section: Beneficial Ingredients
    # Section header background
    pdf.setFillColorRGB(0.95, 0.95, 1.0)
    pdf.rect(left_margin - 5, y_position - 5, content_width + 10, 30, fill=1, stroke=0)
    
    pdf.setFillColorRGB(0.2, 0.2, 0.5)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y_position, "Beneficial Ingredients:")
    y_position -= line_spacing + 5

    beneficial_ingredients = skin_type_ingredients.get(skin_type.capitalize(), [])
    if sensitivity == "Sensitive":
        beneficial_ingredients.extend(skin_type_ingredients.get("Sensitive", []))

    ingredients_text = ", ".join(beneficial_ingredients)
    
    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica", 12)
    
    wrapped_ingredients = simpleSplit(ingredients_text, "Helvetica", 12, content_width - 20)
    
    for line in wrapped_ingredients:
        pdf.drawString(left_margin + 10, y_position, line)
        y_position -= line_spacing

        if y_position < margin + 50:  # Check page overflow with proper margin
            add_footer(current_page)
            pdf.showPage()
            current_page += 1
            
            pdf.setFillColorRGB(0, 0, 0)
            pdf.setFont("Helvetica", 12)
            y_position = height - margin

    # Section: Skin Analysis
    y_position -= 10  # Extra space before new section
    
    # Section header background
    pdf.setFillColorRGB(0.95, 0.95, 1.0)
    pdf.rect(left_margin - 5, y_position - 5, content_width + 10, 30, fill=1, stroke=0)
    
    pdf.setFillColorRGB(0.2, 0.2, 0.5)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y_position, "Skin Analysis:")
    y_position -= line_spacing + 5

    pdf.setFillColorRGB(0, 0, 0)
    pdf.setFont("Helvetica", 12)
    
    wrapped_analysis = simpleSplit(skin_analysis, "Helvetica", 12, content_width - 20)

    for line in wrapped_analysis:
        pdf.drawString(left_margin + 10, y_position, line)
        y_position -= line_spacing

        if y_position < margin + 50:
            add_footer(current_page)
            pdf.showPage()
            current_page += 1
            
            pdf.setFillColorRGB(0, 0, 0)
            pdf.setFont("Helvetica", 12)
            y_position = height - margin

    # Section: Recommended Products
    y_position -= 10  # Extra space
    
    # Section header background
    pdf.setFillColorRGB(0.95, 0.95, 1.0)
    pdf.rect(left_margin - 5, y_position - 5, content_width + 10, 30, fill=1, stroke=0)
    
    pdf.setFillColorRGB(0.2, 0.2, 0.5)
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(left_margin, y_position, "Recommended Products:")
    y_position -= line_spacing * 1.5

    pdf.setFillColorRGB(0, 0, 0)

    for i, product in enumerate(recommendations):
        if y_position < margin + 100:  # More space for product entries
            add_footer(current_page)
            pdf.showPage()
            current_page += 1
            
            pdf.setFillColorRGB(0, 0, 0)
            y_position = height - margin
        
        # Product box background
        pdf.setFillColorRGB(0.97, 0.97, 1.0)  # Very light purple
        pdf.rect(left_margin, y_position - 50, content_width, 60, fill=1, stroke=0)
        
        # Product border
        pdf.setStrokeColorRGB(0.7, 0.7, 0.8)
        pdf.setLineWidth(1)
        pdf.rect(left_margin, y_position - 50, content_width, 60, fill=0, stroke=1)
        
        # Product number circle
        pdf.setFillColorRGB(0.3, 0.3, 0.6)  # Purple for number
        pdf.circle(left_margin + 15, y_position - 15, 10, fill=1, stroke=0)
        
        # Number inside circle
        pdf.setFillColorRGB(1, 1, 1)  # White text
        pdf.setFont("Helvetica-Bold", 10)
        number_width = pdf.stringWidth(str(i+1), "Helvetica-Bold", 10)
        pdf.drawString(left_margin + 15 - (number_width/2), y_position - 18, str(i+1))
        
        # Product name and type
        pdf.setFillColorRGB(0.2, 0.2, 0.4)  # Dark blue
        pdf.setFont("Helvetica-Bold", 12)
        product_text = f"{product['name']} - {product.get('prodtypes', 'N/A')}"
        
        # Draw product name with proper indentation
        pdf.drawString(left_margin + 35, y_position - 15, product_text)
        y_position -= line_spacing + 10
        
        # Product ingredients
        if 'ingredients' in product and product['ingredients']:
            pdf.setFillColorRGB(0, 0, 0)  # Black for ingredients
            pdf.setFont("Helvetica", 10)
            ingredients_text = f"Ingredients: {product['ingredients']}"
            wrapped_ingredients = simpleSplit(ingredients_text, "Helvetica", 10, content_width - 40)

            for line in wrapped_ingredients:
                pdf.drawString(left_margin + 35, y_position - 5, line)
                y_position -= line_spacing - 5
        
        y_position -= line_spacing + 10  # Extra space after each product

    # Add footer to the last page
    add_footer(current_page)
    pdf.save()
    
    # Prepare response
    pdf_buffer.seek(0)
    
    response = Response(pdf_buffer.getvalue(), content_type="application/pdf")
    response.headers["Content-Disposition"] = f"attachment; filename={full_filename}"
    
    return response

if __name__ == '__main__':
    app.run(debug=True)