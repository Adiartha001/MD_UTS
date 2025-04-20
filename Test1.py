#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model dan encoder
model = joblib.load('XG_booking_status.pkl')
booking_status_encode = joblib.load('booking_status_encode.pkl')  # Dict: e.g., {'Canceled': 1, 'Not_Canceled': 0}
oneHot_encode_room = joblib.load('oneHot_encode_room.pkl')
oneHot_encode_meal = joblib.load('oneHot_encode_meal.pkl')
oneHot_encode_mark = joblib.load('oneHot_encode_mark.pkl')

# Judul dan Identitas
st.markdown("### Nama: Adiartha Wibisono Hasnan  ")
st.markdown("### NIM: 2702315236")
st.markdown("<h1 style='text-align: center;'>Hotel Booking Status Prediction</h1>", unsafe_allow_html=True)

def predict_booking_status(no_of_adults, no_of_children, no_of_weekend_nights,
                            no_of_week_nights, type_of_meal_plan, required_car_parking_space,
                            room_type_reserved, lead_time, arrival_year, arrival_month,
                            arrival_date, market_segment_type, repeated_guest,
                            no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
                            avg_price_per_room, no_of_special_requests):

    input_data = pd.DataFrame({
        'no_of_adults': [no_of_adults],
        'no_of_children': [no_of_children],
        'no_of_weekend_nights': [no_of_weekend_nights],
        'no_of_week_nights': [no_of_week_nights],
        'lead_time': [lead_time],
        'arrival_year': [arrival_year],
        'arrival_month': [arrival_month],
        'arrival_date': [arrival_date],
        'required_car_parking_space': [required_car_parking_space],
        'repeated_guest': [repeated_guest],
        'no_of_previous_cancellations': [no_of_previous_cancellations],
        'no_of_previous_bookings_not_canceled': [no_of_previous_bookings_not_canceled],
        'avg_price_per_room': [avg_price_per_room],
        'no_of_special_requests': [no_of_special_requests]
    })

    # Encode fitur kategori
    meal_encoded = oneHot_encode_meal.transform([[type_of_meal_plan]]).toarray()
    room_encoded = oneHot_encode_room.transform([[room_type_reserved]]).toarray()
    market_encoded = oneHot_encode_mark.transform([[market_segment_type]]).toarray()

    # Gabungkan semua fitur
    full_input = np.hstack([input_data.values, meal_encoded, room_encoded, market_encoded])
    prediction = model.predict(full_input)[0]

    # Konversi hasil prediksi dari angka ke label
    reverse_status = {v: k for k, v in booking_status_encode.items()}
    output = reverse_status.get(prediction, "Unknown")
    return output

# Input manual
st.subheader("Input Manual")

no_of_adults = st.number_input("No of Adults", 0, 100)
no_of_children = st.number_input("No of Children", 0, 100)
no_of_weekend_nights = st.number_input('No of Weekend Night', 0, 2)
no_of_week_nights = st.number_input('No of Week Night', 0, 5)
type_of_meal_plan = st.selectbox('Meal Plan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
required_car_parking_space = st.radio('Required Car Parking Space (0 for No, 1 for Yes)', [0, 1])
room_type_reserved = st.selectbox('Room Type Reserved', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 
                                                        'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
lead_time = st.number_input("Lead Time (days)", 0, 360)
arrival_year = st.selectbox("Arrival Year", [2017, 2018])
arrival_month = st.selectbox('Arrival Month', list(range(1, 13)))
arrival_date = st.selectbox("Arrival Date", list(range(1, 32)))
market_segment_type = st.selectbox('Market Segment Type', ['Offline', 'Online', 'Corporate', 'Aviation', 'Complementary'])
repeated_guest = st.radio('Repeated Guest (0 for No, 1 for Yes)', [0, 1])
no_of_previous_cancellations = st.number_input('Previous Cancellations', 0, 100)
no_of_previous_bookings_not_canceled = st.number_input('Previous Bookings Not Canceled', 0, 100)
avg_price_per_room = st.number_input('Average Price Per Room (in Euros)', 0.00, 10000.00)
no_of_special_requests = st.number_input('Number of Special Requests', 0, 100)

if st.button('Predict from Manual Input'):
    hasil = predict_booking_status(
        no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
        type_of_meal_plan, required_car_parking_space, room_type_reserved, lead_time,
        arrival_year, arrival_month, arrival_date, market_segment_type, repeated_guest,
        no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
        avg_price_per_room, no_of_special_requests
    )
    st.success(f"Prediction Result: {hasil}")

# --- Test Case 1 ---
st.subheader("Test Case 1 - Expected: Not_Canceled")
st.write("**Input:** 2 adults, 0 children, 2 weekend nights, 3 week nights, Meal Plan 1, Car Parking: Yes, Room_Type 1, Lead Time 20, 2017-7-15, Market: Online, Repeated Guest: Yes, No previous cancellations, 3 previous bookings not canceled, Price: 100.0, 1 special request")

if st.button("Run Test Case 1"):
    hasil = predict_booking_status(
        no_of_adults=2,
        no_of_children=0,
        no_of_weekend_nights=2,
        no_of_week_nights=3,
        type_of_meal_plan='Meal Plan 1',
        required_car_parking_space=1,
        room_type_reserved='Room_Type 1',
        lead_time=20,
        arrival_year=2017,
        arrival_month=7,
        arrival_date=15,
        market_segment_type='Online',
        repeated_guest=1,
        no_of_previous_cancellations=0,
        no_of_previous_bookings_not_canceled=3,
        avg_price_per_room=100.0,
        no_of_special_requests=1
    )
    st.success(f"Prediction Result: {hasil}")

# --- Test Case 2 ---
st.subheader("Test Case 2 - Expected: Canceled")
st.write("**Input:** 1 adult, 1 child, 0 weekend nights, 0 week nights, Meal Plan 2, Car Parking: No, Room_Type 3, Lead Time 150, 2018-5-10, Market: Offline, Repeated Guest: No, 2 previous cancellations, 0 previous bookings not canceled, Price: 250.0, 0 special request")

if st.button("Run Test Case 2"):
    hasil = predict_booking_status(
        no_of_adults=1,
        no_of_children=1,
        no_of_weekend_nights=0,
        no_of_week_nights=0,
        type_of_meal_plan='Meal Plan 2',
        required_car_parking_space=0,
        room_type_reserved='Room_Type 3',
        lead_time=150,
        arrival_year=2018,
        arrival_month=5,
        arrival_date=10,
        market_segment_type='Offline',
        repeated_guest=0,
        no_of_previous_cancellations=2,
        no_of_previous_bookings_not_canceled=0,
        avg_price_per_room=250.0,
        no_of_special_requests=0
    )
    st.success(f"Prediction Result: {hasil}")


# In[ ]:




