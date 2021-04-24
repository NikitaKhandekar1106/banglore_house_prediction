from flask import Flask, request, jsonify,render_template
import functions
import test_model
import house_db
import config
app = Flask(__name__)

@app.route('/')
def index():
   return render_template(r'C:\Users\Shamali\Desktop\Velocity python 2jan21\Nikita Velocity\FLASK\banglore_hpp-main\banglore_hpp-main\templates\home.html')

@app.route('/register', methods=['GET','POST'])
def register():
    if request.method == 'POST':
        data = request.form
        response = house_db.register_user(data)

    return jsonify({'msg':response})

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        data = request.form
        response = house_db.login_user(data)
    return jsonify({'msg':response})

@app.route('/get_location_names', methods=['GET'])
def get_location_names():
    response = jsonify({
        'locations': test_model.get_location_names()
    })
    return response

@app.route('/predict_home_price', methods=['GET', 'POST'])
def predict_home_price():
    if request.method == 'POST':
        data = request.form
        total_sqft = float(data['sqft'])
        bhk = int(data['bhk'])
        bath = int(data['bath'])
        location = data.get('loc')
        print('location,total_sqft,bhk,bath',location,total_sqft,bhk,bath)
        prediction = test_model.get_predicted_price(location,total_sqft,bhk,bath)
        house_db.save_price_details(location,total_sqft,bhk,bath,prediction)

        return "The Predicted house price is Rs. {} lakhs".format(prediction)
    return render_template(r"C:\Users\Shamali\Desktop\Velocity python 2jan21\Nikita Velocity\FLASK\banglore_hpp-main\banglore_hpp-main\templates\home.html")


if __name__ == "__main__":
    print("Starting Python Flask Server For Home Price Prediction...")
    # functions.load_saved_artifacts()
    app.run(host='0.0.0.0',port=5004)