import streamlit as st
import joblib
import pandas as pd
import pickle

# Ganti warna button
st.markdown("""
<style>
    div.stButton > button:first-child {
        background-color: #4682B4;
        color: white;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #5F9EA0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

def preprocess_data(input_df):
    # Impute missing value
    with open('imputation_values.pkl', 'rb') as f:
        imputation_values_loaded = pickle.load(f)

    mode_typeMeal = imputation_values_loaded['mode_typeMeal']
    mode_parking = imputation_values_loaded['mode_parking']
    median_price = imputation_values_loaded['median_price']

    list_column = ['type_of_meal_plan', 'required_car_parking_space', 'avg_price_per_room']
    for column in list_column:
        if column == 'type_of_meal_plan':
            input_df[column].fillna(mode_typeMeal, inplace=True)
        elif column == 'required_car_parking_space':
            input_df[column].fillna(mode_parking, inplace=True)
        else:
            input_df[column].fillna(median_price, inplace=True)

    input_df = encode_tambahan(input_df) 
    input_df = encoding_scaling(input_df)
    return input_df

def encode_tambahan(input_df):
    df = input_df.copy()
    df['required_car_parking_space'] = df['required_car_parking_space'].apply(lambda x: 1 if x == 'Ya' else 0)
    df['repeated_guest'] = df['repeated_guest'].apply(lambda x: 1 if x == 'Ya' else 0)
    return df

def encoding_scaling(input_df):
    with open('encoders_and_scalers.pkl', 'rb') as f:
        encoders_and_scalers = pickle.load(f)

    train_encoded = encoders_and_scalers['train_encoded']
    minmax_scaler = encoders_and_scalers['minmax_scaler']
    robust_scaler = encoders_and_scalers['robust_scaler']

    cols_encode = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']
    x_enc = input_df[cols_encode]

    enc_column = pd.DataFrame(train_encoded.transform(x_enc), columns=train_encoded.get_feature_names_out())

    input_enc = input_df.reset_index(drop=True).drop(columns=cols_encode)
    input_df = pd.concat([input_enc, enc_column], axis=1)

    minmax_col = ['arrival_year', 'arrival_month', 'arrival_date']
    robust_col = ['no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'lead_time',
                  'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled', 'avg_price_per_room', 'no_of_special_requests']    
    
    input_df[minmax_col] = minmax_scaler.transform(input_df[minmax_col])
    input_df[robust_col] = robust_scaler.transform(input_df[robust_col])
    return input_df

def receive_input():
    no_of_adults = st.number_input('Jumlah Dewasa', min_value=0)
    no_of_children = st.number_input('Jumlah Anak-anak', min_value=0)
    no_of_weekend_nights = st.number_input('Jumlah Malam Weekend (Sabtu-Minggu)', min_value=0)
    no_of_week_nights = st.number_input('Jumlah Malam (Senin-Jumat)', min_value=0)
    type_of_meal_plan = st.selectbox('Jenis Paket Makanan', ['Meal Plan 1', 'Meal Plan 2', 'Meal Plan 3', 'Not Selected'])
    required_car_parking_space = st.selectbox('Apakah Membutuhkan Tempat Parkir?', ['Ya', 'Tidak'])
    room_type_reserved = st.selectbox('Tipe Kamar yang Dipesan', ['Room_Type 1', 'Room_Type 2', 'Room_Type 3', 'Room_Type 4', 'Room_Type 5', 'Room_Type 6', 'Room_Type 7'])
    lead_time = st.number_input('Lead Time (Waktu Antara Pemesanan dan Tanggal Check-In)', min_value=0)
    arrival_year = st.number_input('Tahun Kedatangan', min_value=2000, max_value=2100)
    arrival_month = st.number_input('Bulan Kedatangan', min_value=1, max_value=12)
    arrival_date = st.number_input('Tanggal Kedatangan', min_value=1, max_value=31)
    market_segment_type = st.selectbox('Tipe Segmen Pasar', ['Online', 'Offline', 'Corporate', 'Complementary', 'Aviation'])
    repeated_guest = st.selectbox('Apakah Sebelumnya Pernah Melakukan Booking dan Menginap?', ['Ya', 'Tidak'])
    no_of_previous_cancellations = st.number_input('Jumlah Pembatalan Sebelumnya', min_value=0)
    no_of_previous_bookings_not_canceled = st.number_input('Jumlah Pemesanan Sebelumnya yang Tidak Dibatalkan', min_value=0)
    avg_price_per_room = st.number_input('Harga Rata-rata Per Kamar', min_value=0.0)
    no_of_special_requests = st.number_input('Jumlah Permintaan Khusus', min_value=0)

    input_data = [
        no_of_adults, no_of_children, no_of_weekend_nights, no_of_week_nights,
        type_of_meal_plan, required_car_parking_space, room_type_reserved,
        lead_time, arrival_year, arrival_month, arrival_date, market_segment_type,
        repeated_guest, no_of_previous_cancellations, no_of_previous_bookings_not_canceled,
        avg_price_per_room, no_of_special_requests
    ]

    input_df = pd.DataFrame([input_data], columns=[
            'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights',
            'type_of_meal_plan', 'required_car_parking_space', 'room_type_reserved',
            'lead_time', 'arrival_year', 'arrival_month', 'arrival_date', 'market_segment_type',
            'repeated_guest', 'no_of_previous_cancellations', 'no_of_previous_bookings_not_canceled',
            'avg_price_per_room', 'no_of_special_requests'
        ])
    
    return input_df



model = joblib.load('best_model.pkl')

st.title('Prediksi Status Booking Hotel')
st.write('Masukkan data untuk memprediksi apakah booking akan dibatalkan atau tidak')

input_df = receive_input()

if st.button('Prediksi Status Booking', type='primary'):
    with st.spinner('Sedang memproses data...'):
        processed_data = preprocess_data(input_df)

    with st.spinner('Melakukan prediksi...'):
        hasil = model.predict(processed_data)

    st.subheader('Hasil Prediksi')
    if hasil[0] == 1:
        st.success("✅ Booking tidak akan dibatalkan (Not_Canceled)")
    else:
        st.error("❌ Booking akan dibatalkan (Canceled)")

    st.markdown("---")