from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    session,
)
from werkzeug.utils import secure_filename
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_session import Session
from datetime import timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from models import User, Prediction, db

app = Flask(__name__)


app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://flask:1234@localhost:5432/dbs"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.secret_key = "supersecretkey"

app.config["SESSION_TYPE"] = "filesystem"
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "supersecretkey")
app.config["SESSION_PERMANENT"] = True
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(minutes=30)
app.config["SESSION_USE_SIGNER"] = True

Session(app)

db.init_app(app)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


MODEL_PATH = '//home/vivekananda/Downloads/FLASK/flask-project (1)/apps/trained_models/PCOS.h5'

try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print("Error loading the model:", e)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/admin", methods=["GET", "POST"])
def admin():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()

        if user and check_password_hash(user.password, password) and user.is_admin:
            session["admin_id"] = user.id
            session["admin_name"] = user.name
            flash("Logged in successfully", "success")
            return redirect(url_for("admin_dashboard"))
        else:
            flash("Invalid credentials", "danger")
            return redirect(url_for("admin"))
    return render_template("admin.html")


@app.route("/admin_dashboard")
def admin_dashboard():
    users = User.query.all()
    return render_template("admin-dashboard.html", users=users)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        name = request.form.get("name")
        if request.form.get("is_admin"):
            is_admin = bool(request.form.get("is_admin"))
        else:
            is_admin = False

        if User.query.filter_by(email=email).first():
            flash("Email already registered", "warning")
            return redirect(url_for("signup"))
        hashed_password = generate_password_hash(
            password,
            method="pbkdf2:sha256",
        )
        new_user = User(
            email=email, password=hashed_password, name=name, is_admin=is_admin
        )
        db.session.add(new_user)
        db.session.commit()
        flash("Account created successfully", "success")
        if is_admin:
            return redirect(url_for("admin"))
        return redirect(url_for("signup"))

    return render_template("patient.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")
        user = User.query.filter_by(email=email).first()

        try:

            if user.is_admin:
                flash("Please Login as admin", "danger")
                return redirect(url_for("admin"))
            if user and check_password_hash(user.password, password):
                session["user_id"] = user.id
                session["user_name"] = user.name
                flash("Logged in successfully", "success")
                return redirect(url_for("services"))
            else:
                flash("Invalid credentials", "warning")
                return redirect(url_for("login"))
        except Exception:
            flash("Invalid credentials", "warning")
            return redirect(url_for("login"))
    return redirect(url_for("services"))


@app.route("/logout")
def logout():
    session.pop("user_id", None)
    session.pop("user_name", None)
    if "admin_id" in session:
        session.pop("admin_id", None)
        session.pop("admin_name", None)

    flash("Logged out successfully", "success")
    return redirect(url_for("home"))


@app.route("/services")
def services():
    if "user_id" not in session:
        flash("Please log in to continue", "warning")
        return redirect(url_for("signup"))
    return render_template("services.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/contact")
def contact():
    if "user_id" not in session:
        flash("Please log in to continue", "warning")
        return redirect(url_for("signup"))
    return render_template("contact.html")


@app.route("/pcos_checker")
def pcos_checker():
    return render_template("pcoschecker.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "user_id" not in session:
        flash("Please log in to continue", "warning")
        return redirect(url_for("signup"))
    try:
        form_data = request.form.to_dict()
        numeric_fields = [
            "age",
            "weight",
            "height",
            "bmi",
            "blood_group",
            "pulse_rate",
            "rr",
            "hb",
            "cycle",
            "cycle_length",
            "marriage_status",
            "pregnant",
            "abortions",
            "fsh",
            "lh",
            "hip",
            "waist",
            "tsh",
            "amh",
            "prl",
            "vit_d3",
            "prg",
            "rbs",
            "weight_gain",
            "hair_growth",
            "skin_darkening",
            "hair_loss",
            "pimples",
            "fast_food",
            "regular_exercise",
            "bp_systolic",
            "bp_diastolic",
            "follicle_L",
            "follicle_R",
            "avg_f_size_L",
            "avg_f_size_R",
            "endometrium",
        ]

        input_data = {key: float(form_data[key]) for key in numeric_fields}
        model = joblib.load(
            "trained_models/random_forest_pcos.joblib"
        )  # Assuming separate model for form data
        prediction = model.predict([list(input_data.values())])[0]
        print(prediction)
        if prediction == 1.0:
            Result = "Positive"
        else:
            Result = "Negative"

        user_id = session["user_id"]
        new_prediction = Prediction(
            result=Result,
            user_id=user_id,
            prediction_type="pcos_prediction",
        )
        db.session.add(new_prediction)
        db.session.commit()

        # Pass name, age, and prediction result to predict.html template
        name = form_data.get("name")
        age = form_data.get("age")
        return render_template(
            "predict.html",
            Result=Result,
            name=name,
            age=age,
        )
    except Exception as e:
        print("Error in predict function:", e)
        return "Failed to make a prediction. Please try again later."


@app.route("/image_scan", methods=["GET", "POST"])
def image_scan():
    if "user_id" not in session:
        flash("Please log in to continue", "warning")
        return redirect(url_for("signup"))
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        file = request.files["file"]
        if file.filename == "":
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            try:
                file.save(file_path)
                return filename
            except Exception as e:
                return str(e), 500
    return render_template("imagescan.html")


@app.route("/result/<filename>")
def result(filename):
    if "user_id" not in session:
        flash("Please log in to continue", "warning")
        return redirect(url_for("signup"))
    if filename == "No file part":
        return "No file uploaded"
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    if not os.path.exists(file_path):
        return "File not found"

    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)
    predicted_value = prediction[0][0]
    predicted_class = "Not Affected" if predicted_value == 1.0 else "Affected"

    # Retrieve user ID and name (assuming you have a way to get the user's name)
    user_id = session["user_id"]
    user_name = session.get("user_name", "User")  # Replace with actual method to get user's name

    new_prediction = Prediction(
        result=predicted_class,
        user_id=user_id,
        prediction_type="image_scan",
    )
    db.session.add(new_prediction)
    db.session.commit()

    return render_template(
        "result.html", filename=filename, prediction_result=predicted_class, name=user_name
    )


@app.route("/history")
def history():
    if "user_id" not in session:
        flash("Please log in to continue", "warning")
        return redirect(url_for("signup"))
    user_id = session["user_id"]
    predictions = (
        Prediction.query.filter_by(user_id=user_id)
        .order_by(Prediction.timestamp.desc())
        .all()
    )

    return render_template("history.html", predictions=predictions)


@app.route("/init_db")
def init_db():
    try:
        db.create_all()
        return "Database initialized!"
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    app.run(debug=True, port=8008)
