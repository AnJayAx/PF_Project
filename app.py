from flask import Flask, render_template
from flask import Flask, render_template, request, redirect
import requests

app = Flask(__name__)


# Login route
@app.route('/', methods=['GET', 'POST'])
def login():  # put application's code here
    # url for carpark availability
    # url = "https://api.data.gov.sg/v1/transport/carpark-availability"
    # url for hdb resale prices and details
    collectionId = 189
    url = "https://api-production.data.gov.sg/v2/public/api/collections/{}/metadata".format(collectionId)
    response = requests.get(url)
    print(response.json())
    return render_template('base.html', title='base')


if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
