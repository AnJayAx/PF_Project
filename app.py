from flask import Flask, render_template, request, redirect, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, login_required, logout_user, current_user, LoginManager
import requests
import locale
import re
from werkzeug.security import generate_password_hash, check_password_hash
from uuid import uuid4
import threading
locale.setlocale(locale.LC_ALL, '')
import time


app = Flask(__name__)

# Google Cloud SQL
PASSWORD = "123"
PUBLIC_IP_ADDRESS = "34.87.87.11"
DBNAME = "database"
PROJECT_ID = "oceanic-hangout-435310"
INSTANCE_NAME = "database1"

# Database configuration
app.config[
    "SQLALCHEMY_DATABASE_URI"] = f"mysql+mysqldb://root:{PASSWORD}@{PUBLIC_IP_ADDRESS}:3306/{DBNAME}?unix_socket=/cloudsql/{PROJECT_ID}:{INSTANCE_NAME}"
db = SQLAlchemy(app)


class Users(db.Model, UserMixin):
    id = db.Column(db.String(255), primary_key=True, nullable=False)
    username = db.Column(db.String(20), nullable=False)
    email = db.Column(db.String(70), nullable=False, unique=True)
    password = db.Column(db.String(45), nullable=False)


# Session management
app.secret_key = '5791628bb0b13ce0c676dfde280ba245'
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Table data in memory
table_data_chunks = []

def fetch_data_from_api():
    completeAPICall = False
    global table_data_chunks
    local_data_chunks = []
    base_url = "https://data.gov.sg"
    first_api_res = "/api/action/datastore_search?resource_id=d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    hdb_api = base_url + first_api_res
    
    start = time.time()
    while not completeAPICall:
        response = requests.get(hdb_api)
        if response.status_code == 200:
            data = response.json()
            records = data["result"]["records"]

            for j in records:
                j["price_per_sq_metre"] = str(round(float(j["resale_price"]) / float(j["floor_area_sqm"]), 2))
                j["f_resale_price"] = locale.currency(float(j["resale_price"]), grouping=True)
                j["f_price_per_sq_metre"] = locale.currency(float(j["price_per_sq_metre"]), grouping=True)

            # Chunk data into groups of 10
            chunk_size = 10
            for i in range(0, len(records), chunk_size):
                local_data_chunks = [records[i:i + chunk_size]]
                local_data_chunks.reverse()

            try:
                if (data["result"]["total"] - data["result"]["offset"]) <= 100:
                    completeAPICall = True
                    end = time.time()
                    time_taken = (end - start) / 60

                    print("Loop End ++++++")
                    print("Time Taken for full API call: " + str(time_taken))
            except:
                print("Fetching data from API...")
                pass
            
            hdb_api = base_url + data["result"]["_links"]["next"]
            table_data_chunks.insert(0, local_data_chunks)

        else:
            print('Failed to fetch data from API')
            break

def fetch_data_async():
    fetch_data_from_api()

@app.before_request
def startup():
    app.before_request_funcs[None].remove(startup)
    thread = threading.Thread(target=fetch_data_async)
    thread.start()
    

@login_manager.user_loader
def load_user(id):
    return Users.query.get(id)


@app.route('/', methods=['GET', 'POST'])
def main():  # put application's code here
    # url for carpark availability
    # url = "https://api.data.gov.sg/v1/transport/carpark-availability"
    # url for hdb resale prices and details
    collectionId = 189
    url = "https://api-production.data.gov.sg/v2/public/api/collections/{}/metadata".format(collectionId)
    #response = requests.get(url)
    #print(response.json())

    return render_template('index.html', title='index', user=current_user)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect("/")
    if request.method == 'POST':
        email = request.form["email"]
        pwd = request.form["password"]
        user = Users.query.filter_by(email=email).first()

        def login_credentials():
            if user:
                if check_password_hash(user.password, pwd):
                    return True
                else:
                    flash("Incorrect Email or Password!", category='error')
            else:
                flash("Incorrect Email or Password!", category='error')

        if login_credentials():
            flash("Login Successful!", category='success')
            login_user(user, remember=True)
            return redirect("/")
        else:
            return redirect("login")

    return render_template('page-signin.html', title='index', user=current_user)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect("/")
    if request.method == 'POST':
        # attributes from register form
        userid = str(uuid4())
        username = request.form["username"]
        email = request.form["email"]
        password = request.form["password"]
        password2 = request.form["password2"]

        def validate_form():
            email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
            password_regex = r'^(?=.*[!@#$%^&*(),.?":{}|<>])[A-Za-z\d!@#$%^&*(),.?":{}|<>]{7,}$'
            email_valid = False
            password_valid = False
            user = Users.query.filter_by(email=email).first()

            # check for invalid email, password
            if re.match(email_regex, email):
                if user:
                    flash("Email is invalid! ", category='error')
                else:
                    email_valid = True

                if password == password2:
                    # Check if password meets the criteria
                    if re.match(password_regex, password):
                        password_valid = True
                    else:
                        flash(
                            "Password is invalid. It must contain at least 8 characters, including letters and numbers!",
                            category='error')
                else:
                    flash("Passwords do not match! ", category='error')
            else:
                flash("Email is invalid!", category='error')

            return email_valid and password_valid

        # validate if all fields are correct
        if validate_form():
            # commit new user details to DB
            new_user = Users(id=userid, username=username, email=email,
                             password=generate_password_hash(password, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            flash("Account has been created!", category='success')
            return redirect("login")
        else:
            return redirect("register")

    return render_template('page-signup.html', title='index', user=current_user)


@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    logout_user()
    return redirect("login")

@app.route("/get_data", methods=["GET"])
def get_data():
    # Get request parameters from Datatables
    draw = request.args.get('draw')  # For keeping track of requests
    start = int(request.args.get('start', 0))  # Starting index
    length = int(request.args.get('length', 10))  # Number of rows per page

    # [ [{1},{2},{3},{4},{5}] , [{1},{2},{3},{4},{5}] ]

    # Data chunking
    filtered_data = sum(table_data_chunks, [])
    total_records = len(filtered_data)
    data_to_send = filtered_data[start:start + length]
    #print(data_to_send[0])
    response = {
        'draw': draw,
        'recordsTotal': total_records,  # Total number of records
        'recordsFiltered': total_records,  # Number of records after filtering
        'data': data_to_send[0]  # Send only the current page's data
    }

    return jsonify(response)

@app.route('/search-hdb', methods=['GET', 'POST'])
def search_hdb():
    offsetTesting = "&offset=189600"
    datasetId = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc" + offsetTesting
    url = "https://data.gov.sg/api/action/datastore_search?resource_id=" + datasetId + offsetTesting

    responseJSON = requests.get(url).json()

    print("-------------------------------")
    records = responseJSON["result"]["records"]
    # Self-calculate Price per Sq Metre and insert into the records.
    for i in records:
        i["price_per_sq_metre"] = str(round(float(i["resale_price"]) / float(i["floor_area_sqm"]), 2))
        i["f_resale_price"] = locale.currency(float(i["resale_price"]), grouping=True)
        i["f_price_per_sq_metre"] = locale.currency(float(i["price_per_sq_metre"]), grouping=True)
    # print(records)
    print("-------------------------------")
    rows = range(len(records))
    # Filters required:
    # (DateTimePicker) - Date Range from Start to End, 
    # (Range Slider) - Price Range, 
    # (Single Select Box w/wo Search) - Flat Type(Rooms), Town
    # ---------------------------------------
    # RMB TO-DO:
    # hover over table headers to see what they mean
    # responsivity of the table could be better
    return render_template('search-hdb.html', title='index', user=current_user)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
