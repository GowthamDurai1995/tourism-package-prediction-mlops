
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Streamlit configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Tourism Package Prediction",
    page_icon="✈️",
    layout="centered"
)

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
MODEL_PATH = "/content/tourism_project/model_building/model/tourism_model.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("✈️ Tourism Package Purchase Prediction")

st.write(
    "Enter the customer details below to predict whether "
    "the customer is likely to purchase the tourism package."
)

# --------------------------------------------------
# User Inputs
# --------------------------------------------------

age = st.number_input("Age", min_value=18, max_value=100, value=35)

typeofcontact = st.selectbox(
    "Type of Contact",
    ["Self Enquiry", "Company Invited"]
)

citytier = st.selectbox("City Tier", [1, 2, 3])

durationofpitch = st.number_input(
    "Duration of Pitch",
    min_value=0,
    value=15
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried", "Small Business", "Large Business", "Free Lancer"]
)

gender = st.selectbox("Gender", ["Male", "Female"])

numberofpersonvisiting = st.number_input(
    "Number of Persons Visiting",
    min_value=1,
    value=2
)

numberoffollowups = st.number_input(
    "Number of Followups",
    min_value=0,
    value=4
)

productpitched = st.selectbox(
    "Product Pitched",
    ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
)

preferredpropertystar = st.selectbox(
    "Preferred Property Star",
    [3, 4, 5]
)

maritalstatus = st.selectbox(
    "Marital Status",
    ["Single", "Married", "Divorced", "Unmarried"]
)

numberoftrips = st.number_input(
    "Number of Trips",
    min_value=0,
    value=2
)

passport = st.selectbox(
    "Passport Available",
    [0, 1]
)

pitchsatisfactionscore = st.slider(
    "Pitch Satisfaction Score",
    min_value=1,
    max_value=5,
    value=3
)

owncar = st.selectbox(
    "Owns a Car",
    [0, 1]
)

numberofchildrenvisiting = st.number_input(
    "Number of Children Visiting",
    min_value=0,
    value=1
)

designation = st.selectbox(
    "Designation",
    ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
)

monthlyincome = st.number_input(
    "Monthly Income",
    min_value=0,
    value=30000
)

# --------------------------------------------------
# Create input DataFrame
# Column names match the trained model exactly
# --------------------------------------------------

input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeofcontact,
    "CityTier": citytier,
    "DurationOfPitch": durationofpitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": numberofpersonvisiting,
    "NumberOfFollowups": numberoffollowups,
    "ProductPitched": productpitched,
    "PreferredPropertyStar": preferredpropertystar,
    "MaritalStatus": maritalstatus,
    "NumberOfTrips": numberoftrips,
    "Passport": passport,
    "PitchSatisfactionScore": pitchsatisfactionscore,
    "OwnCar": owncar,
    "NumberOfChildrenVisiting": numberofchildrenvisiting,
    "Designation": designation,
    "MonthlyIncome": monthlyincome
}])

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔮 Predict Package Purchase"):

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success(
            "🎉 The customer is likely to purchase the tourism package!"
        )
    else:
        st.warning(
            "The customer is unlikely to purchase the tourism package."
        )

    st.subheader("Customer Input")
    st.dataframe(input_data)
