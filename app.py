from flask import Flask, render_template
from flask import Flask, render_template, request, redirect
import requests
import locale
locale.setlocale( locale.LC_ALL, '' )

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
    return render_template('index.html', title='index')

@app.route('/search-hdb', methods=['GET', 'POST'])
def search_hdb():  
    offsetTesting = "&offset=189600"
    offset = ""
    datasetId = "d_8b84c4ee58e3cfc0ece0d773c8ca6abc" + offsetTesting
    url = "https://data.gov.sg/api/action/datastore_search?resource_id="  + datasetId 
            
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
    #print(records)
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
    return render_template('search-hdb.html', title='index', rows = rows, records = records)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=True)
