from flask import Flask, render_template, request, redirect, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, login_required, logout_user, current_user, LoginManager
from flask_socketio import SocketIO, emit
import requests
import locale
import re
from werkzeug.security import generate_password_hash, check_password_hash
from uuid import uuid4
import pandas as pd
import os
from datetime import datetime
import threading
locale.setlocale(locale.LC_ALL, '')
import time
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)
socketio = SocketIO(app)

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

path = os.getcwd()

# Session management
app.secret_key = '5791628bb0b13ce0c676dfde280ba245'
login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# Table DataFrames
df_api_data = pd.DataFrame()
df_older_data = pd.read_csv(path + '/static/csv/ResaleFlatPrices1990-2016.csv', low_memory=False)

def fetch_data_from_api():
    completeAPICall = False
    global df_api_data
    table_data_chunks = []
    base_url = "https://data.gov.sg"
    first_api_res = "/api/action/datastore_search?resource_id=d_8b84c4ee58e3cfc0ece0d773c8ca6abc"
    limit = "&limit=9999"
    hdb_api = base_url + first_api_res + limit

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

            table_data_chunks.extend(records)
            
            try:
                print(data["result"]["offset"] > data["result"]["total"])
                if data["result"]["offset"] > data["result"]["total"]:
                    completeAPICall = True
                    end = time.time()
                    time_taken = (end - start) / 60

                    df_api_data = pd.DataFrame(table_data_chunks)

                    if '_id' in df_api_data.columns:
                        start_id = df_api_data['_id'].max() + 1
                    else:
                        start_id = 1  # If JSON has no IDs, start from 1
                    df_older_data['_id'] = range(start_id, start_id + len(df_older_data))                    
                    df_api_data = pd.concat([df_api_data, df_older_data], ignore_index=True, sort=False)

                    # Sort by date
                    df_api_data['month'] = pd.to_datetime(df_api_data['month'], format='%Y-%m', errors='coerce')
                    df_api_data = df_api_data.sort_values(by='month', ascending=False)
                    df_api_data['month'] = df_api_data['month'].dt.strftime('%Y-%m')

                    socketio.emit('data_ready', {'message': 'Data is ready!'})

                    print("Loop End ++++++")
                    print("Time Taken for full API call: " + str(time_taken))
            except:
                print("Fetching data from API...")
                pass

            hdb_api = base_url + data["result"]["_links"]["next"]
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
    length = int(request.args.get('length', 20))  # Number of rows per page
    df_to_use = df_older_data
    #print(df_to_use)

    # Uses older data into the table until api data is fully retrieved. Then use that.
    if not df_api_data.empty:
        df_to_use = df_api_data

    # Total number of records in the DataFrame
    total_records = len(df_to_use)

    # Slice the DataFrame based on pagination parameters
    paginated_data = df_to_use.iloc[start:start + length]

    # Convert the sliced data to a list of dictionaries (records)
    data_to_send = paginated_data.to_dict(orient='records')

    response = {
        'draw': draw,
        'recordsTotal': total_records,  # Total number of records
        'recordsFiltered': total_records,  # Number of records after filtering
        'data': data_to_send  # Send only the current page's data
    }

    return jsonify(response)

@app.route('/filter-data', methods=['POST'])
def filter_data():
    data = request.json
    print("Data received is as such: ")
    print(data)

    #error validation not working... going to sleep zzz

    errors = []
    try:
        if data["remaining_lease"]["year"] != "":
            rem_lease_year = int(data["remaining_lease"]["year"])
        if data["remaining_lease"]["months"] != "":
            rem_lease_months = int(data["remaining_lease"]["months"])
        if data["storey"]["from"] != "":
            storey_from = int(data["storey"]["from"])
        if data["storey"]["to"] != "":
            storey_to = int(data["storey"]["to"])
        if data["lease_date"] != "":
            lease_date = int(data["lease_date"])
    except:
        errors.append("Please enter valid numbers.")
        return errors

    return jsonify(data)

@app.route('/search-hdb', methods=['GET', 'POST'])
def search_hdb():
    #print(df_api_data.columns)
    #data = df_api_data[['month', 'town', 'flat_type', 'block', 'street_name', 'storey_range', 'floor_area_sqm', 'lease_commence_date', 'resale_price', 'f_resale_price', 'f_price_per_sq_metre']].to_dict(orient='records')
    # Filters required:
    # (DateTimePicker) - Date Range from Start to End, 
    # (Range Slider) - Price Range, 
    # (Single Select Box w/wo Search) - Flat Type(Rooms), Town
    # ---------------------------------------
    # RMB TO-DO:
    # hover over table headers to see what they mean
    # responsivity of the table could be better
    return render_template('search-hdb.html', title='index', user=current_user)

# Load the CSV data into a global variable to avoid reloading
df = pd.read_csv(path + '/static/csv/ResaleFlatPrices.csv')

@app.route('/predict', methods=['GET'])
def prediction():
    date = datetime.now()
    tn_list = df['town'].unique().tolist()  # List of unique towns
    return render_template('prediction.html', user=current_user, tn_list=tn_list, date=date)

# Route to handle prediction
@app.route('/prediction', methods=['POST'])
def predict():
    data = request.json  # Get the data from the AJAX request
    print("Received data:", data)

    # Extract values
    month_year = data.get("month_year")
    town = data.get("town")
    block = data.get("block")
    street_name = data.get("street")
    flat_type = data.get("flat_type")
    storey_range = int(data.get("storey_range"))

    # Filter the DataFrame to find the matching row(s)
    filtered_df = df[
        (df['town'] == town) &
        (df['block'] == block) &
        (df['street_name'] == street_name) &
        (df['flat_type'] == flat_type)
        ]

    predicted_price = 0
    # Check if we found a matching row
    if not filtered_df.empty:
        # Extract the first matching row (you can handle multiple matches if necessary)
        result = filtered_df.iloc[0]

        # Prepare the data to return to the front-end
        output = {
            'town': town,
            'flat_type': result['flat_type'],
            'block': block,
            'street_name': street_name,
            'storey_range': storey_range,
            'floor_area_sqm': result['floor_area_sqm'],
            'lease_commence_date': result['lease_commence_date'],
            'remaining_lease': result['remaining_lease'],
            'max_floor_lvl': result['max_floor_lvl'],
            'year_completed': result['year_completed'],
            'nearest_mrt': result['nearest_mrt'],
            'nearest_distance_to_mrt': result['nearest_distance_to_mrt'],
            'month_numeric': convert_to_months_since_base(month_year),
        }

        print("Output", output)
        predicted_price = predicting(output)
        print("Predicted: ", predicted_price)
    else:
        print('No matching data found')

    # Return the predicted price as JSON
    return jsonify({"predicted_price": predicted_price})

# Define your function to convert 'year-month' to months since the base year
def convert_to_months_since_base(year_month, base_year=1960):
    year, month = map(int, year_month.split('-'))
    months_since_base = (year - base_year) * 12 + month
    print(months_since_base)
    return months_since_base

def predicting(input_dict):
    # Load the models and scalers
    model = load_model(path + '/static/prediction/resale_model_adjusted.h5')
    label_encoders = joblib.load(path + '/static/prediction/label_encoders.pkl')  # Load label encoders
    scaler_X = joblib.load(path + '/static/prediction/scaler_X.pkl')  # Load feature scaler
    scaler_y = joblib.load(path + '/static/prediction/scaler_y.pkl')  # Load target scaler

    print(input_dict)
    # Create a DataFrame from the input
    input_df = pd.DataFrame([input_dict])

    # Map categorical columns using the loaded encoders
    for col in label_encoders.keys():
        if col in input_df.columns:
            le = label_encoders[col]  # Get the correct mapping for the column
            print(len(le.keys()))
            # Check if the input value is valid for the LabelEncoder
            if input_dict[col] not in le.keys():
                print(f"Warning: '{input_dict[col]}' is not a recognized value for '{col}'.")

            # Transform the input using the dictionary mapping
            input_df[col] = le[input_dict[col]]

    # Scale the input
    input_scaled = scaler_X.transform(input_df)

    # Make the prediction
    prediction_result = model.predict(input_scaled)

    predicted_value = scaler_y.inverse_transform(prediction_result.reshape(-1, 1))

    return predicted_value.tolist()

@app.route('/get_blocks/<town>', methods=['GET'])
def get_blocks(town):
    blocks = df[df['town'] == town]['block'].unique().tolist()  # Get unique blocks for the selected town
    return jsonify({'blocks': blocks})

@app.route('/get_streets/<town>/<block>', methods=['GET'])
def get_streets(town, block):
    streets = df[(df['town'] == town) & (df['block'] == block)]['street_name'].unique().tolist()
    return jsonify({'streets': streets})

@app.route('/get_flats/<block>/<street>', methods=['GET'])
def get_flats(block, street):
    flat_types = df[(df['block'] == block) & (df['street_name'] == street)]['flat_type'].unique().tolist()
    return jsonify({'flat_types': flat_types})



if __name__ == '__main__':
    #app.run(debug=True, use_reloader=True)
    socketio.run(app, debug=True, use_reloader=True)
