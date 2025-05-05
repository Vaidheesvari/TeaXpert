import os
import sqlite3
from flask import Flask, request, redirect, url_for, render_template, session, flash, g, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
from werkzeug.utils import secure_filename
app = Flask(__name__)
app.secret_key = 'supersecretkey'

DATABASE = 'leaf_db' 

# ---------------- DB Connection ---------------- #
def get_db_connection():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db_connection(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Upload settings
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model once
model = load_model('trained_model_DNN1.h5')

# Class labels
class_names = ['algal leaf', 'Anthracnose', 'bird eye spot', 'brown blight', 'gray light',
               'healthy', 'red leaf spot', 'white spot']

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

def predict(img_path):
    processed_img = preprocess_image(img_path)
    prediction = model.predict(processed_img)
    class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return class_names[class_index], confidence

@app.route('/detection', methods=['GET', 'POST'])
def detection():
    if 'email' not in session:
        flash('Please login to access detection.', 'warning')
        return redirect(url_for('login')) 

    if request.method == 'POST':
        file = request.files.get('image')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(img_path)

            relative_path = f'uploads/{filename}'
            predicted_class, confidence = predict(img_path)  
            formatted_prediction = predicted_class.lower().replace(' ', '_') 

            return render_template('detection.html', 
                                   image_path=relative_path, 
                                   prediction=predicted_class,  
                                   formatted_prediction=formatted_prediction,
                                   confidence=confidence)
        else:
            flash('Invalid file. Please upload a JPG/PNG/BMP image.', 'danger')

    return render_template('detection.html')
tips_data = {
    "anthracnose": [
        "Remove infected leaves and ensure proper air circulation.",
        "Use copper-based fungicides as a preventative measure.",
        "Maintain proper watering practices to avoid leaf diseases.",
        "Apply neem oil as a natural pesticide to combat pests.",
        "Ensure soil drainage to prevent fungal growth."
    ],
    "algal_leaf": [
        "Reduce watering frequency.",
        "Apply a suitable fungicide.",
        "Ensure good air circulation around the plants.",
        "Remove affected leaves promptly.",
        "Maintain proper soil pH levels."
    ],
    "bird_eye_spot": [
        "Use resistant plant varieties if available.",
        "Apply fungicides as recommended.",
        "Ensure proper spacing between plants.",
        "Remove fallen leaves and debris.",
        "Water plants in the morning to reduce humidity."
    ],
    "brown_blight": [
        "Prune infected areas of the plant.",
        "Avoid overhead watering.",
        "Apply fungicides as necessary.",
        "Ensure adequate sunlight for plants.",
        "Rotate crops to avoid soil-borne diseases."
    ],
    "gray_light": [
        "Ensure proper irrigation to avoid water stress.",
        "Prune affected areas to promote healthy growth.",
        "Maintain optimal humidity levels.",
        "Use fungicides to control fungal growth.",
        "Increase air circulation around plants."
    ],
    "healthy": [
        "Maintain proper watering practices.",
        "Ensure good soil drainage and healthy root systems.",
        "Provide adequate sunlight and proper nutrition.",
        "Monitor plants for early signs of disease.",
        "Keep the garden or field free from debris."
    ],
    "red_leaf_spot": [
        "Remove infected leaves to prevent spread.",
        "Apply appropriate fungicides for red leaf spot.",
        "Ensure proper spacing between plants for better air circulation.",
        "Avoid over-watering to reduce fungal growth.",
        "Use disease-resistant plant varieties if available."
    ],
    "white_spot": [
        "Remove affected leaves promptly.",
        "Use fungicides to control white spot disease.",
        "Maintain good air circulation and proper spacing.",
        "Water plants in the morning to reduce humidity.",
        "Ensure proper soil drainage."
    ]
}

@app.route('/tips/<disease_name>', methods=['GET', 'POST'] )
def get_tips(disease_name):
    """Fetch tips based on the disease name."""
    try:
        disease_name = disease_name.lower() 
        
      
        tips = tips_data.get(disease_name, ["No tips available for this disease."])
        
       
        return render_template('tips.html', disease_name=disease_name, tips=tips)
    
    except Exception as e:
        print(f"Error fetching tips: {e}")
        return jsonify({
            "status": "error",
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }), 500



# ---------------- Create DB/Table if Needed ---------------- #
def init_db():
    if not os.path.exists(DATABASE):
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                number TEXT NOT NULL,
                password TEXT NOT NULL,
                dob TEXT NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
        print("Database and users table created.")
    else:
        print("Database already exists.")

# ---------------- Routes ---------------- #
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        number = request.form['number']
        password = request.form['password']
        dob = request.form['dob']

        hashed_password = generate_password_hash(password)

        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute(
                'INSERT INTO users (name, email, number, password, dob) VALUES (?, ?, ?, ?, ?)',
                (name, email, number, hashed_password, dob)
            )
            conn.commit()
            flash('Registration successful. Please login.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Email already exists.', 'danger')
        finally:
            cursor.close()
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['email'] = user['email']
            session['name'] = user['name']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid email or password', 'danger')

    return render_template('login.html')

@app.route('/profile')
def profile():
    if 'email' not in session:
        flash('Please login to view your profile.', 'warning')
        return redirect(url_for('login'))

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM users WHERE email = ?', (session['email'],))
    user = cursor.fetchone()
    cursor.close()
    conn.close()

    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('login'))

    return render_template('profile.html', user=user)

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('login'))

# ---------------- Run Server ---------------- #
if __name__ == '__main__':
    init_db() 
    app.run(debug=True)

