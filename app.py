from flask import Flask, render_template, request
import json
import threading

FILE = 'data.json'
data = {}


app = Flask(__name__)
app_loaded = False

def load_data(FILE, data=None):
    """Returns json from the input file."""
    try:
        if data != None:
            data = data
        else:
            with open(FILE) as f:
                data = json.load(f)
    except:
        return {}
    if len(data) == 0:
    	return {}
    return data


@app.route('/')
def main():
    """Get request for root folder."""
    return render_template('index.php',
                           data=data,)


def start_app():
    global app_loaded, data
    data = load_data(FILE)

    app_loaded = True
    app.run(port=8080)

if __name__ == '__main__':
    start_app()
