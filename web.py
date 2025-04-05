import streamlit as st
import numpy as np
import pickle

# Load models and encoders
rf_exercise = pickle.load(open("exercise_model.pkl", "rb"))
rf_food = pickle.load(open("food_model.pkl", "rb"))
rf_selfcare = pickle.load(open("selfcare_model.pkl", "rb"))

le_phase = pickle.load(open("phase_encoder.pkl", "rb"))
le_energy = pickle.load(open("energy_encoder.pkl", "rb"))
le_symptom = pickle.load(open("symptom_encoder.pkl", "rb"))

st.set_page_config(page_title="Menstrual Lifestyle Tips", layout="centered")

st.markdown("## 🌸 Menstrual Lifestyle Tips Predictor")
st.markdown("Personalized tips based on your cycle phase, energy level, and symptoms.")

# User input
phase = st.selectbox("Cycle Phase", le_phase.classes_)
energy = st.selectbox("Energy Level", le_energy.classes_)
symptoms = st.multiselect("Select Your Symptoms", le_symptom.classes_)

if st.button("Get Tips"):
    if len(symptoms) == 0:
        st.warning("Please select at least one symptom.")
    else:
        phase_encoded = le_phase.transform([phase])[0]
        energy_encoded = le_energy.transform([energy])[0]
        symptom_encoded = [le_symptom.transform([s])[0] for s in symptoms]
        symptom_avg = int(np.mean(symptom_encoded))

        input_data = np.array([[phase_encoded, energy_encoded, symptom_avg]])

        exercise_tip = rf_exercise.predict(input_data)[0]
        food_tip = rf_food.predict(input_data)[0]
        selfcare_tip = rf_selfcare.predict(input_data)[0]

        st.success("### 💪 Exercise Tip\n" + exercise_tip)
        st.success("### 🥗 Food Remedy\n" + food_tip)
        st.success("### 🧘 Self-Care Tip\n" + selfcare_tip)
