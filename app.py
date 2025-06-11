from flask import Flask, render_template, request, redirect, url_for, session, flash
import sqlite3
from datetime import datetime
from rapidfuzz import fuzz
import pickle
import numpy as np
import os
import csv
from functools import wraps

app = Flask(__name__)
app.secret_key = "super_secret_key"
SESSION_TIMEOUT = 10 * 60

# Load placement model
with open('placement_model.pkl', 'rb') as f:
    placement_model = pickle.load(f)

# ---------- Database Setup ----------
def init_db():
    conn = sqlite3.connect('chat.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_message TEXT, bot_response TEXT, timestamp TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS intents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        tag TEXT NOT NULL, pattern TEXT NOT NULL, response TEXT NOT NULL)''')

    conn.commit()
    conn.close()

init_db()

# Preload default intents
def preload_intents():
    conn = sqlite3.connect('chat.db')
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM intents")
    if c.fetchone()[0] == 0:
        default_intents = [
            ("student", "Student Info", "This section contains your personal student profile and academic performance."),
            ("exam", "Exam Dates", "Mid-semester: Oct 15 | End-semester: Dec 10"),
            ("calendar", "Academic Calendar", "You can download the calendar from the official college website."),
            ("class", "Classes", "Classes run from Monday to Saturday, 9 AM to 5 PM."),
            ("admission", "Admission", "For admission queries, visit the admin office or call 01234-567890."),
            ("complaint", "Complaints", "You can submit complaints in Room 102 or via student portal."),
            ("comment", "Comments", "We value your feedback! Mail to feedback@college.edu."),
            ("library", "Library Info", "Library is open 8 AM to 8 PM. Membership is mandatory."),
            ("faculty", "Faculty List", "Visit faculty directory on the portal or ask department office."),
            ("campus", "Campus Map", "Find the campus map at the entrance gate or website homepage."),
            ("menu", "Main Menu", "Student Info | Exam Dates | Academic Calendar | Classes | Admission | Complaints | Comments | Library Info | Faculty List | Campus Map")
        ]
        c.executemany("INSERT INTO intents (tag, pattern, response) VALUES (?, ?, ?)", default_intents)
        conn.commit()
    conn.close()

preload_intents()

def get_db():
    conn = sqlite3.connect('chat.db')
    conn.row_factory = sqlite3.Row
    return conn

# ---------- Session Timeout ----------
@app.before_request
def auto_logout():
    if 'admin_logged_in' in session:
        now = datetime.utcnow()
        last_active = session.get('last_active')
        if last_active:
            last_active = datetime.strptime(last_active, "%Y-%m-%d %H:%M:%S.%f")
            if (now - last_active).total_seconds() > SESSION_TIMEOUT:
                session.clear()
                flash("Session expired. Please login again.")
                return redirect(url_for('login'))
        session['last_active'] = now.strftime("%Y-%m-%d %H:%M:%S.%f")

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin_logged_in"):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ---------- Chatbot Logic ----------
def log_chat(msg, reply):
    conn = get_db()
    conn.execute("INSERT INTO chat_history (user_message, bot_response, timestamp) VALUES (?, ?, ?)",
                 (msg, reply, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()

def find_response(msg):
    conn = get_db()
    intents = conn.execute("SELECT pattern, response FROM intents").fetchall()
    conn.close()

    msg_lower = msg.lower()
    for intent in intents:
        if intent['pattern'].lower() == msg_lower:
            return intent['response']

    best_match, score = None, 0
    for intent in intents:
        match_score = fuzz.partial_ratio(msg_lower, intent['pattern'].lower())
        if match_score > score and match_score > 70:
            best_match, score = intent['response'], match_score
    return best_match or "Sorry, I didn't understand that. कृपया पुनः प्रयास करें।"

# ---------- Routes ----------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat_response", methods=["POST"])
def chat_response():
    msg = request.form.get("msg")
    response = find_response(msg)
    log_chat(msg, response)
    return response

@app.route("/placement_predict", methods=["POST"])
def placement_predict():
    data = request.form
    features = [float(data['CGPA']), int(data['Internship']), int(data['Communication']),
                int(data['Technical']), int(data['Certifications']),
                int(data['Projects']), int(data['ExtraActivities'])]
    prediction = placement_model.predict(np.array(features).reshape(1, -1))[0]
    result = "Yes" if prediction == 1 else "No"

    suggestions = []
    if float(data['CGPA']) < 7: suggestions.append("Improve CGPA")
    if int(data['Internship']) == 0: suggestions.append("Get internship experience")
    if int(data['Communication']) < 6: suggestions.append("Work on communication")
    if int(data['Technical']) < 6: suggestions.append("Improve technical skills")

    file_exists = os.path.exists('pl.csv')
    write_header = not file_exists or os.stat('pl.csv').st_size == 0

    with open('pl.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["Name", "Prediction", "Time", "Suggestions"])
        writer.writerow([data['Name'], result, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "; ".join(suggestions)])

    return render_template("index.html", prediction=result, suggestions=suggestions)

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        if request.form.get("password") == "Oversmart13":
            session["admin_logged_in"] = True
            session["last_active"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")
            return redirect(url_for("admin"))
        flash("Wrong password.")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out.")
    return redirect(url_for("home"))

@app.route("/admin")
@login_required
def admin():
    conn = get_db()
    intents = conn.execute("SELECT * FROM intents").fetchall()
    conn.close()
    return render_template("admin.html", intents=intents)

@app.route("/add_intent", methods=["POST"])
@login_required
def add_intent():
    tag = request.form['tag']
    pattern = request.form['pattern']
    response = request.form['response']
    if not tag or not pattern or not response:
        flash("All fields required.")
    else:
        conn = get_db()
        conn.execute("INSERT INTO intents (tag, pattern, response) VALUES (?, ?, ?)", (tag, pattern, response))
        conn.commit()
        conn.close()
        flash("Intent added.")
    return redirect(url_for("admin"))

@app.route("/update_intent/<int:intent_id>", methods=["POST"])
@login_required
def update_intent(intent_id):
    tag = request.form['tag']
    pattern = request.form['pattern']
    response = request.form['response']
    if not tag or not pattern or not response:
        flash("All fields are required for update.")
    else:
        conn = get_db()
        conn.execute("UPDATE intents SET tag=?, pattern=?, response=? WHERE id=?",
                     (tag, pattern, response, intent_id))
        conn.commit()
        conn.close()
        flash("Intent updated successfully.")
    return redirect(url_for("admin"))

@app.route("/delete_intent/<int:intent_id>")
@login_required
def delete_intent(intent_id):
    conn = get_db()
    conn.execute("DELETE FROM intents WHERE id=?", (intent_id,))
    conn.commit()
    conn.close()
    flash("Intent deleted.")
    return redirect(url_for("admin"))

if __name__ == "__main__":
    app.run(debug=True)