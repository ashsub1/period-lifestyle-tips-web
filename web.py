import streamlit as st
import numpy as np
import pickle

# Load models
with open("exercise_model.pkl", "rb") as f:
    rf_exercise = pickle.load(f)

with open("food_model.pkl", "rb") as f:
    rf_food = pickle.load(f)

with open("selfcare_model.pkl", "rb") as f:
    rf_selfcare = pickle.load(f)

# Load encoders
with open("phase_encoder.pkl", "rb") as f:
    le_phase = pickle.load(f)

with open("energy_encoder.pkl", "rb") as f:
    le_energy = pickle.load(f)

with open("symptom_encoder.pkl", "rb") as f:
    le_symptom = pickle.load(f)

# Title and description
st.set_page_config(page_title="Period Lifestyle Tips", layout="centered")
st.title("🌸 Period Lifestyle Tips")
st.markdown("Get **personalized, detailed** tips on exercise, food, and self-care based on your cycle phase, energy, and symptoms.")

# Input
cycle_phase = st.selectbox("Cycle Phase", le_phase.classes_)
energy_level = st.selectbox("Energy Level", le_energy.classes_)
symptoms = st.multiselect("Symptoms (select one or more)", le_symptom.classes_)

# Prediction function
def predict_tips(phase, energy, symptoms):
    phase_encoded = le_phase.transform([phase])[0]
    energy_encoded = le_energy.transform([energy])[0]

    if not symptoms:
        st.warning("Please select at least one symptom.")
        return None, None, None

    symptom_encoded = [le_symptom.transform([s])[0] for s in symptoms]
    avg_symptom = np.mean(symptom_encoded)

    input_data = np.array([[phase_encoded, energy_encoded, avg_symptom]])

    exercise_tip = rf_exercise.predict(input_data)[0]
    food_tip = rf_food.predict(input_data)[0]
    selfcare_tip = rf_selfcare.predict(input_data)[0]

    return exercise_tip, food_tip, selfcare_tip

# Button and output
if st.button("Get Tips"):
    exercise, food, selfcare = predict_tips(cycle_phase, energy_level, symptoms)

    if exercise:
        st.markdown(f"### 💪 Exercise Tip\n{exercise}")
        st.markdown(f"### 🥗 Food Remedy\n{food}")
        st.markdown(f"### 🧘 Self-Care Tip\n{selfcare}")

