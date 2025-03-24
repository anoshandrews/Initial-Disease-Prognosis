import streamlit as st
import joblib
import numpy as np
from training import encoder


model = joblib.load('disease_prediction.pkl')

le = joblib.load('encoder.pkl')


# Title
st.title("Initial disease prognosis")
st.subheader("Select the symptoms you're experiencing, grouped by system.")

# -------------------------

all_symptoms = [
    'itching',
 'skin_rash',
 'nodal_skin_eruptions',
 'continuous_sneezing',
 'shivering',
 'chills',
 'joint_pain',
 'stomach_pain',
 'acidity',
 'ulcers_on_tongue',
 'muscle_wasting',
 'vomiting',
 'burning_micturition',
 'spotting_ urination',
 'fatigue',
 'weight_gain',
 'anxiety',
 'cold_hands_and_feets',
 'mood_swings',
 'weight_loss',
 'restlessness',
 'lethargy',
 'patches_in_throat',
 'irregular_sugar_level',
 'cough',
 'high_fever',
 'sunken_eyes',
 'breathlessness',
 'sweating',
 'dehydration',
 'indigestion',
 'headache',
 'yellowish_skin',
 'dark_urine',
 'nausea',
 'loss_of_appetite',
 'pain_behind_the_eyes',
 'back_pain',
 'constipation',
 'abdominal_pain',
 'diarrhoea',
 'mild_fever',
 'yellow_urine',
 'yellowing_of_eyes',
 'acute_liver_failure',
 'fluid_overload',
 'swelling_of_stomach',
 'swelled_lymph_nodes',
 'malaise',
 'blurred_and_distorted_vision',
 'phlegm',
 'throat_irritation',
 'redness_of_eyes',
 'sinus_pressure',
 'runny_nose',
 'congestion',
 'chest_pain',
 'weakness_in_limbs',
 'fast_heart_rate',
 'pain_during_bowel_movements',
 'pain_in_anal_region',
 'bloody_stool',
 'irritation_in_anus',
 'neck_pain',
 'dizziness',
 'cramps',
 'bruising',
 'obesity',
 'swollen_legs',
 'swollen_blood_vessels',
 'puffy_face_and_eyes',
 'enlarged_thyroid',
 'brittle_nails',
 'swollen_extremeties',
 'excessive_hunger',
 'extra_marital_contacts',
 'drying_and_tingling_lips',
 'slurred_speech',
 'knee_pain',
 'hip_joint_pain',
 'muscle_weakness',
 'stiff_neck',
 'swelling_joints',
 'movement_stiffness',
 'spinning_movements',
 'loss_of_balance',
 'unsteadiness',
 'weakness_of_one_body_side',
 'loss_of_smell',
 'bladder_discomfort',
 'foul_smell_of urine',
 'continuous_feel_of_urine',
 'passage_of_gases',
 'internal_itching',
 'toxic_look_(typhos)',
 'depression',
 'irritability',
 'muscle_pain',
 'altered_sensorium',
 'red_spots_over_body',
 'belly_pain',
 'abnormal_menstruation',
 'dischromic _patches',
 'watering_from_eyes',
 'increased_appetite',
 'polyuria',
 'family_history',
 'mucoid_sputum',
 'rusty_sputum',
 'lack_of_concentration',
 'visual_disturbances',
 'receiving_blood_transfusion',
 'receiving_unsterile_injections',
 'coma',
 'stomach_bleeding',
 'distention_of_abdomen',
 'history_of_alcohol_consumption',
 'fluid_overload.1',
 'blood_in_sputum',
 'prominent_veins_on_calf',
 'palpitations',
 'painful_walking',
 'pus_filled_pimples',
 'blackheads',
 'scurring',
 'skin_peeling',
 'silver_like_dusting',
 'small_dents_in_nails',
 'inflammatory_nails',
 'blister',
 'red_sore_around_nose',
 'yellow_crust_ooze',
]

# 1. Skin Symptoms
skin_symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'ulcers_on_tongue',
    'dischromic _patches', 'blister', 'pus_filled_pimples', 'blackheads',
    'scurring', 'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails',
    'inflammatory_nails', 'red_sore_around_nose', 'yellow_crust_ooze'
]

# 2. Neurological Symptoms
neuro_symptoms = [
    'headache', 'dizziness', 'loss_of_balance', 'unsteadiness',
    'weakness_of_one_body_side', 'slurred_speech', 'altered_sensorium',
    'coma', 'lack_of_concentration', 'visual_disturbances',
    'spinning_movements', 'depression', 'irritability'
]

# 3. Gastrointestinal Symptoms
gastro_symptoms = [
    'stomach_pain', 'acidity', 'vomiting', 'indigestion',
    'loss_of_appetite', 'abdominal_pain', 'diarrhoea', 'constipation',
    'bloody_stool', 'pain_during_bowel_movements', 'pain_in_anal_region',
    'irritation_in_anus', 'belly_pain', 'passage_of_gases',
    'distention_of_abdomen', 'stomach_bleeding'
]

# 4. Respiratory Symptoms
respiratory_symptoms = [
    'continuous_sneezing', 'breathlessness', 'cough', 'high_fever',
    'runny_nose', 'sinus_pressure', 'chest_pain', 'phlegm',
    'throat_irritation', 'redness_of_eyes', 'mucoid_sputum',
    'rusty_sputum', 'blood_in_sputum'
]

# 5. Cardiovascular Symptoms
cardio_symptoms = [
    'palpitations', 'fast_heart_rate', 'prominent_veins_on_calf',
    'painful_walking'
]

# 6. Musculoskeletal Symptoms
muscle_joint_symptoms = [
    'joint_pain', 'muscle_wasting', 'back_pain', 'knee_pain',
    'hip_joint_pain', 'muscle_weakness', 'movement_stiffness',
    'swelling_joints', 'stiff_neck', 'weakness_in_limbs'
]

# 7. Urinary Symptoms
urinary_symptoms = [
    'burning_micturition', 'spotting_ urination', 'yellow_urine',
    'continuous_feel_of_urine', 'bladder_discomfort',
    'foul_smell_of urine', 'polyuria'
]

# 8. Endocrine/Metabolic Symptoms
endocrine_symptoms = [
    'weight_gain', 'weight_loss', 'irregular_sugar_level', 'excessive_hunger',
    'increased_appetite', 'obesity', 'cold_hands_and_feets',
    'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties',
    'puffy_face_and_eyes'
]

# 9. General/Systemic Symptoms
general_symptoms = [
    'shivering', 'chills', 'fatigue', 'malaise', 'lethargy',
    'weakness_in_limbs', 'sweating', 'dehydration', 'mild_fever',
    'high_fever', 'acute_liver_failure', 'fluid_overload',
    'swelling_of_stomach', 'swelled_lymph_nodes', 'toxic_look_(typhos)',
    'family_history', 'history_of_alcohol_consumption', 'receiving_blood_transfusion',
    'receiving_unsterile_injections'
]

# 10. Eye Symptoms
eye_symptoms = [
    'blurred_and_distorted_vision', 'redness_of_eyes', 'watering_from_eyes',
    'pain_behind_the_eyes', 'yellowing_of_eyes'
]

# 11. Reproductive Symptoms
reproductive_symptoms = [
    'abnormal_menstruation', 'extra_marital_contacts'
]

# 12. Miscellaneous/Other
misc_symptoms = [
    'drying_and_tingling_lips', 'red_spots_over_body', 'yellowish_skin', 'dark_urine',
    'prognosis', 'fluid_overload.1'
]

# -------------------------
# Create Expanders for Each Symptom Category
selected_symptoms = []

with st.expander("Skin Symptoms"):
    selected = st.multiselect("Select skin symptoms:", skin_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Neurological Symptoms"):
    selected = st.multiselect("Select neurological symptoms:", neuro_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Gastrointestinal Symptoms"):
    selected = st.multiselect("Select gastrointestinal symptoms:", gastro_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Respiratory Symptoms"):
    selected = st.multiselect("Select respiratory symptoms:", respiratory_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Cardiovascular Symptoms"):
    selected = st.multiselect("Select cardiovascular symptoms:", cardio_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Musculoskeletal Symptoms"):
    selected = st.multiselect("Select musculoskeletal symptoms:", muscle_joint_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Urinary Symptoms"):
    selected = st.multiselect("Select urinary symptoms:", urinary_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Endocrine/Metabolic Symptoms"):
    selected = st.multiselect("Select endocrine/metabolic symptoms:", endocrine_symptoms)
    selected_symptoms.extend(selected)

with st.expander("General/Systemic Symptoms"):
    selected = st.multiselect("Select general/systemic symptoms:", general_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Eye Symptoms"):
    selected = st.multiselect("Select eye symptoms:", eye_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Reproductive Symptoms"):
    selected = st.multiselect("Select reproductive symptoms:", reproductive_symptoms)
    selected_symptoms.extend(selected)

with st.expander("Miscellaneous/Other Symptoms"):
    selected = st.multiselect("Select miscellaneous symptoms:", misc_symptoms)
    selected_symptoms.extend(selected)

# -------------------------
# Display selected symptoms
st.markdown("---")
st.subheader("Selected Symptoms")

if selected_symptoms:
    st.write(selected_symptoms)
else:
    st.write("No symptoms selected yet.")

# Optional: You can connect this list to a prediction model later

def predict_disease(selected_symptoms):
    # Create a binary input array for the model (1 if symptom is selected, 0 otherwise)
    input_data = [1 if symptom in selected_symptoms else 0 for symptom in all_symptoms]

    # Convert list to 2D array since scikit-learn expects 2D input
    input_array = np.array([input_data])

    # Make prediction
    predicted_label = model.predict(input_array)[0]

    predicted_disease = le.inverse_transform([predicted_label])[0]

    return predicted_disease

# -------------------------
# Prediction Button
if st.button("Predict Disease"):
    if selected_symptoms:
        disease = predict_disease(selected_symptoms)
        st.success(f"🔍 Based on the symptoms, the predicted disease is: **{disease}**")
    else:
        st.warning("Please select at least one symptom before predicting.")