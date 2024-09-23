from flask import Flask, render_template, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, login_required, logout_user, current_user, LoginManager
import requests
import locale
import re
from werkzeug.security import generate_password_hash, check_password_hash
from uuid import uuid4
locale.setlocale(locale.LC_ALL, '')



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
    response = requests.get(url)
    print(response.json())

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


@app.route('/search-hdb', methods=['GET', 'POST'])
def search_hdb():
    offsetTesting = "&offset=189600"
    offset = ""
    datasetId = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc" + offsetTesting
    url = "https://data.gov.sg/api/action/datastore_search?resource_id=" + datasetId

    responseJSON = requests.get(url).json()
    # To iterate through the API datasets:
    # Each api call will only return 100 rows from the dataset starting from the bottom (2017).
    # Api calls will return a ["_links"]["next"] obj containing the url to use for the next api call.
    # The next obj will be offset by the amount you already received, meaning you won't read what you already got.
    # Loop this until the ["total"] - ["offset"] is < 100 
    ######################################
    # completeAPICall = False
    # while completeAPICall == True:
    #     responseJSON = requests.get(url).json()
    #     #print(responseJSON)
    #     if "offset" in responseJSON["result"]["_links"]:
    #         if (responseJSON["result"]["total"] - responseJSON["result"]["_links"]["offset"]) > 100:
    #             url = responseJSON["_links"]["next"]
    #             print("loop API")
    #         else:
    #             completeAPICall = True

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
    # how to do a buncha api calls asynchronously... talking about tens of thousands of rows of data
    # may be better to download and store ALL the data in csvs and only api call for the most updated stuff via offset.
    # hover over table headers to see what they mean
    # responsivity of the table could be better
    return render_template('search-hdb.html', title='index', rows=rows, records=records, user=current_user)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
