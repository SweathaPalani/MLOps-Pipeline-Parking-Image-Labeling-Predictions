import gradio as gr
import tensorflow as tf
from xgboost import XGBClassifier, DMatrix
import numpy as np
import pickle


dnn_path = "DNN_predicting_validated_parking.h5"

xgboost_path = "xgboost_model.pkl"


dnn_model = tf.keras.models.load_model(dnn_path)
# xgboost_model = pickle.load(open(xgboost_path, "rb"))


def get_dnn_input(
    city: str,
    state: str,
    business_stars: float,
    business_review_count: int,
    is_open: int,
    BusinessAcceptsCreditCards: int,
    restaurant_price_level: int,
    BikeParking: int,
    parking_garage: int,
    parking_street: int,
    lot_parking: int,
    valet_parking: int,
    review_stars: int,
    review_useful: int,
    review_funny: int,
    review_cool: int,
    user_review_count: int,
    user_average_stars: float,
    checkin_review_count: int,
):
    return np.array(
        [
            [
                business_stars,
                business_review_count,
                is_open,
                BusinessAcceptsCreditCards,
                restaurant_price_level,
                BikeParking,
                parking_garage,
                parking_street,
                lot_parking,
                valet_parking,
                review_stars,
                review_useful,
                review_funny,
                review_cool,
                user_review_count,
                user_average_stars,
                checkin_review_count,
            ]
        ]
    )


def predict(
    city: str,
    state: str,
    business_stars: float,
    business_review_count: int,
    is_open: int,
    BusinessAcceptsCreditCards: int,
    restaurant_price_level: int,
    BikeParking: int,
    parking_garage: int,
    parking_street: int,
    lot_parking: int,
    valet_parking: int,
    review_stars: int,
    review_useful: int,
    review_funny: int,
    review_cool: int,
    user_review_count: int,
    user_average_stars: float,
    checkin_review_count: int,
):
    # Create input array for DNN model
    dnn_input = np.array(
        [
            [
                state,
                business_stars,
                business_review_count,
                is_open,
                BusinessAcceptsCreditCards,
                restaurant_price_level,
                BikeParking,
                parking_garage,
                parking_street,
                lot_parking,
                valet_parking,
                review_stars,
                review_useful,
                review_funny,
                review_cool,
                user_review_count,
                user_average_stars,
                checkin_review_count,
            ]
        ]
    )

    # xgboost_input = DMatrix(dnn_input)

    # Make predictions
    dnn_prediction = dnn_model.predict(dnn_input)[0][0]
    # xgboost_prediction = xgboost_model.predict(xgboost_input)[0]

    dnn_result = (
        "Validated parking likely"
        if dnn_prediction > 0.5
        else "Validated parking unlikely"
    )
    # xgboost_result = (
    #     "Validated parking likely"
    #     if xgboost_prediction > 0.5
    #     else "Validated parking unlikely"
    # )

    return f"DNN Prediction for validated parking: {dnn_input} \nXGBoost Prediction for validated parking: "


form = gr.Interface(
    fn=predict,
    inputs=[
        # gr.Textbox(label="City"),
        gr.Textbox(label="State"),
        gr.Slider(minimum=0.0, maximum=5.0, label="Business Stars", step=0.1),
        gr.Number(label="Business Review Count"),
        gr.Number(label="Is Open (1 for Yes, 0 for No)"),
        gr.Number(label="Business Accepts Credit Cards (1 for Yes, 0 for No)"),
        gr.Number(label="Restaurant Price Level (1-4)"),
        gr.Number(label="Bike Parking (1 for Yes, 0 for No)"),
        gr.Number(label="Parking Garage (1 for Yes, 0 for No)"),
        gr.Number(label="Parking Street (1 for Yes, 0 for No)"),
        gr.Number(label="Lot Parking (1 for Yes, 0 for No)"),
        gr.Number(label="Valet Parking (1 for Yes, 0 for No)"),
        gr.Number(label="Review Stars"),
        gr.Number(label="Review Useful"),
        gr.Number(label="Review Funny"),
        gr.Number(label="Review Cool"),
        gr.Number(label="User Review Count"),
        gr.Slider(minimum=0.0, maximum=5.0, label="User Average Stars", step=0.1),
        gr.Number(label="Checkin Review Count"),
    ],
    outputs="text",
    title="Business Parking Prediction",
    description="Please fill out the information about the business and its reviews.",
)

form.launch()
