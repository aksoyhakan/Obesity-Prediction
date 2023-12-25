import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import joblib
import time
from PIL import Image
from pytube import YouTube
import io
from moviepy.editor import VideoFileClip

image1 = Image.open(r'C:\Users\Meltem AKSOY\Desktop\Obesity-2\images.jpeg')
image2 = Image.open(r'C:\Users\Meltem AKSOY\Desktop\Obesity-2\TheDose_Ep99_FatimaStandord_Obesity-as-a-Disease_3x2.png')
st.set_page_config(page_title="Know Your Obesity Risk :question:", layout="wide")
st.header(":blue[Know Your Obesity] :red[Risk?]")
tab_home, tab_dataset, tab_vis, tab_model = st.tabs(["Home", "About Dataset", "Charts", "Obesity Level Prediction"])

# TAB HOME
inf, video = tab_home.columns(2, gap="large")
inf.markdown("The primary goal of this project is to develop a predictive model that can anticipate an individual's likelihood of becoming obese based on comprehensive data analysis. Obesity is associated with a range of adverse outcomes, including an elevated risk of cardiovascular diseases, type 2 diabetes, respiratory issues, and joint problems. Additionally, it can impact mental health, leading to increased rates of depression and anxiety. That is why addressing and preventing obesity is vital for overall health and well-being.")
inf.markdown("By using the dataset, encompassing crucial variables such as family history with overweight, dietary habits (including frequent consumption of high-caloric food, frequency of vegetable consumption, and consumption of food between meals), lifestyle factors (smoking habits, daily water consumption, calories consumption monitoring, physical activity frequency, time using technology devices, and alcohol consumption), and transportation type used, the aim is to uncover patterns and correlations that contribute to obesity. Through the exploration of these multifaceted factors, the project seeks to provide valuable insights for proactive interventions, enabling individuals to make informed lifestyle choices and potentially reduce the risk of obesity in the future.")
inf.markdown("Lastly, recognizing the importance of addressing obesity is crucial, given its profound implications for overall health and the increased risk of various health issues. By understanding and predicting obesity, we hope to contribute to the broader effort of promoting healthier lifestyles and preventing potential health risks associated with obesity.")

video_file = open(r'C:\Users\Meltem AKSOY\Desktop\Obesity-2\Recommendations on physical activity for older adults - Nutrition UP 65.mp4', 'rb')
video_bytes = video_file.read()
video.video(video_bytes)

#TAB DATASET
tab_dataset.subheader("About Dataset")
column_inf, column_dataset  = tab_dataset.columns(2, gap="large")

column_inf.image(image2, width=300, use_column_width=True, clamp=False, channels='RGB', output_format='auto')

column_dataset.markdown(
    """
       
    
    - The dataset used for the analysis was obtained from [UCI ML Repository](https://archive.ics.uci.edu/datasets).
    - It includes data for estimating obesity levels in people aged 14 to 61 with various eating habits and physical conditions in Mexico, Peru, and Colombia.
    - Data was collected using a web platform survey with 17 attributes and 2111 records.
    - Data was preprocessed including missing and atypical data deletion, and data normalization.
    - Eating habit attributes include FAVC, FCVC, NCP, CAEC, CH20, and CALC.
    - Physical condition attributes include SCC, FAF, TUE, and MTRANS. Other variables include Gender, Age, Height, and Weight.
    - Records are labeled with the NObesity class variable, allowing classification into 7 groups. Labeling process was performed based on WHO and Mexican Normativity. Balancing class was performed using the SMOTE filter using the tool Weka.
    - The dataset authors note that 23% of the records were collected directly, and the remaining 77% were generated synthetically.
    
    Value of the data:
    - This data presents information from different locations such as Mexico, Peru and Colombia, can be used to build estimation of the obesity levels based on the nutritional behavior of several regions. 
    - The data can be used for estimation of the obesity level of individuals using seven categories, allowing a detailed analysis of the affectation level of an individual. 
    - The structure and amount of data can be used for different tasks in data mining such as: classification, prediction, segmentation and association. 
    - The data can be used to build software tools for estimation of obesity levels. The data can validate the impact of several factors that propitiate the apparition of obesity problems.
    
    """)




column_dataset.subheader("Questions of the survey used for initial recollection of information")
def get_data2():
    df = pd.read_excel(r"C:\Users\Meltem AKSOY\Desktop\Obesity-2\Kitap1.xlsx")
    return df

df2 = get_data2()
column_dataset.dataframe(df2)

column_inf.subheader("Dataset")
def get_data():
    df = pd.read_csv(r"C:\Users\Meltem AKSOY\Desktop\Obesity-2\ObesityDataSet.csv")
    return df

df = get_data()
column_inf.dataframe(df)


df["Obesity Level"] = df["NObeyesdad"]
df["Physical Activity Frequency"] = df["FAF"]
df["Smoking"] = df["SMOKE"]
df["High Caloric Food Consumption"] = df["FAVC"]
df["Vegetables Consumption"] = df["FCVC"]
df["Number of Main Meals"] = df["NCP"]
df["Eating Between Meals"] = df["CAEC"]
df["Daily Water Consumption"] = df["CH2O"]
df["Calories Monitoring"] = df["SCC"]
df["Technological Devices Usage"] = df["TUE"]
df["Alcohol Consumption"] = df["CALC"]
df["Transportation_Preference"] = df["MTRANS"]
df["Family History with Overweight"] = df["family_history_with_overweight"]
df["BMI"] = df["Weight"]/(df["Height"]*df["Height"])





# TAB VIS
##Grafik-1
tab_vis.subheader("Numerical Variable Distributions by Gender")
numeric_columns = ["Age", "Height", "Weight", "BMI"]
selected_numerical_columns = tab_vis.multiselect(label="Select numerical columns", options=numeric_columns[0:], default=numeric_columns[0:])
for col in selected_numerical_columns:
    fig1 = px.histogram(df, x=col, marginal="box", color="Gender")
    fig1.update_layout(template='plotly_dark', title_x=0.5, yaxis_title='Counts', xaxis_title=f"{col}", title=f"{col} Distribution")
    tab_vis.plotly_chart(fig1, use_container_width=True)

##Grafik-2
tab_vis.subheader("Correlation Between Height and Weight")
fig2 = px.scatter(data_frame=df,y="Height",x="Weight",size="BMI",color="Gender",trendline="ols")
fig2.update_layout(template='plotly_dark')
tab_vis.plotly_chart(fig2, use_container_width=True)

##Grafik-3
def cross_plot(data, target_column, categorical_column, categories):
    fig3 = px.histogram(
        data[data[target_column].isin(categories)],
        x=target_column,
        color=categorical_column,
        barmode='group',
        category_orders={categorical_column: categories},

    )
    tab_vis.plotly_chart(fig3, use_container_width=True)
categorical_columns = ['Gender', 'Physical Activity Frequency', 'High Caloric Food Consumption', 'Vegetables Consumption', 'Number of Main Meals', 'Eating Between Meals','Smoking', 'Daily Water Consumption', 'Calories Monitoring', 'Alcohol Consumption', 'Technological Devices Usage', 'Family History with Overweight', 'Transportation_Preference']
selected_categorical_column = tab_vis.multiselect(label="Select a variable", options=categorical_columns, default=["Gender"])
categories = ["Insufficient_Weight", "Normal_Weight", "Overweight_Level_I", "Overweight_Level_II", "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"]
if selected_categorical_column:
    tab_vis.subheader(f"Cross-Plot: Distribution of Obesity Levels by {selected_categorical_column[0]}")
cross_plot(df, "Obesity Level", selected_categorical_column[0], categories)


## Grafik-4

tab_vis.subheader("Physical Activity Frequency by Obesity Level")

fig4 = px.box(df, x="Obesity Level", y="Physical Activity Frequency", points="all")
tab_vis.plotly_chart(fig4, use_container_width=True)

##Grafik-5
tab_vis.subheader("Scatter Plot of Age and BMI")

fig5 = px.scatter(df, x="Age", y="BMI", color="Obesity Level")
tab_vis.plotly_chart(fig5, use_container_width=True)

##Grafik-6
tab_vis.subheader("BMI Comparison Based on Gender and Transportation Preference")

selected_MTRANS = tab_vis.multiselect(label="Select a transportation type", options=df.Transportation_Preference.unique(), default=["Public_Transportation"])
filtered_df = df[df.Transportation_Preference.isin(selected_MTRANS)]

fig6 = px.bar(
    filtered_df,
    x="Obesity Level",
    y="BMI",
    color="Gender",
    facet_col="Transportation_Preference",
    labels={"BMI": "Body Mass Index"}
)
tab_vis.plotly_chart(fig6, use_container_width=True)



# TAB MODEL

df3 = pd.read_csv(r"C:\Users\Meltem AKSOY\Desktop\Obesity-2\ObesityDataSet.csv")
df3.columns = [col.upper() for col in df3.columns]
# loading the saved model
model = joblib.load(r"C:\Users\Meltem AKSOY\Desktop\Obesity-2\rta_model_deploy3.joblib")
encoder = joblib.load(r"C:\Users\Meltem AKSOY\Desktop\Obesity-2\onehot_encoder.joblib")

#creating option list for dropdown menu
options_GENDER = ["Select an option", "Female", "Male"]
options_FAMILY_HISTORY_WITH_OVERWEIGHT  = ["Select an option", 'No', 'Yes']
options_FAVC   = ["Select an option", 'No', 'Yes']
options_FCVC   = ["Select an option", 'Never', 'Sometimes', 'Always']
options_NCP   = ["Select an option", 'One', 'Two', 'Three', 'More than three']
options_CAEC   = ["Select an option", 'No', 'Sometimes', 'Frequently', 'Always']
options_SMOKE   = ["Select an option", 'No', 'Yes']
options_CH2O   = ["Select an option", 'Less than 1L', 'Between 1L and 2L', 'More than 2L']
options_SCC   = ["Select an option", 'No', 'Yes']
options_FAF   = ["Select an option", 'I do not have', '1 or 2 days', '3 or 4 days', '4 or 5 days']
options_TUE   = ["Select an option", '0-2 hours', '3-5 hours', 'More than 5 hours']
options_CALC  = ["Select an option", 'I do not drink', 'Sometimes', 'Frequently', 'Always']
options_MTRANS  = ["Select an option", 'Automobile', 'Motorbike', 'Bike', 'Public Transportation', 'Walking']


def main():
    tab_model.subheader("Please enter the following inputs:")
    GENDER = tab_model.selectbox('What is your gender?', options=options_GENDER)
    AGE = tab_model.text_input('Enter your age', placeholder='e.g., 25')

    # HEIGHT = tab_model.text_input('Enter your height (in cm)', placeholder='e.g., 1.75')
    # WEIGHT = tab_model.text_input('Enter your weight (in kg)', placeholder='e.g., 70')
    FAMILY_HISTORY_WITH_OVERWEIGHT = tab_model.selectbox('Has a family member suffered or suffers from overweight?',
                                                         options=options_FAMILY_HISTORY_WITH_OVERWEIGHT)
    FAVC = tab_model.selectbox('Do you eat high caloric food frequently?', options=options_FAVC)
    FCVC = tab_model.selectbox('Do you usually eat vegetables in your meals?', options=options_FCVC)
    NCP = tab_model.selectbox('How many main meals do you have daily?', options=options_NCP)
    CAEC = tab_model.selectbox('Do you eat any food between meals?', options=options_CAEC)
    SMOKE = tab_model.selectbox('Do you smoke?', options=options_SMOKE)
    CH2O = tab_model.selectbox('How much water do you drink daily?', options=options_CH2O)
    SCC = tab_model.selectbox('Do you monitor the calories you eat daily?', options=options_SCC)
    FAF = tab_model.selectbox('How often do you have physical activity?', options=options_FAF)
    TUE = tab_model.selectbox(
        'How much time do you use technological devices such as cell phone, videogames, television, computer and others?',
        options=options_TUE)
    CALC = tab_model.selectbox('How often do you drink alcohol?', options=options_CALC)
    MTRANS = tab_model.selectbox('Which transportation do you usually use?', options=options_MTRANS)

    # AGE = pd.to_numeric(AGE, errors='coerce')
    # WEIGHT = pd.to_numeric(AGE, errors='coerce')
    # HEIGHT = pd.to_numeric(AGE, errors='coerce')
    #def check_and_warn_selection(field_name, selected_option):
       # if selected_option == "Select an option":
            #column_prediction.warning(f"Please select a {field_name.lower()}.")
    st.cache(allow_output_mutation=True, hash_funcs={np.ndarray: lambda x: hash(x.tobytes())})
    if tab_model.button('Obesity Level Prediction Result'):
        input_array = np.array([GENDER, FAMILY_HISTORY_WITH_OVERWEIGHT, FAVC, CALC, SMOKE, SCC, CAEC, MTRANS, FCVC, NCP, CH2O, FAF, TUE],
            ndmin=2)
        encoded_matrix = encoder.transform(input_array)
        dense_array = encoded_matrix.toarray()
        encoded_arr = list(dense_array.ravel())

        num_arr = [AGE]
        pred_arr = np.array(num_arr + encoded_arr, dtype=np.float64).reshape(1, -1)
        prediction = model.predict(pred_arr)

        with st.spinner('Please wait...'):
            time.sleep(1)

        if prediction == 0:
            text = 'Insufficient Weight'
        elif prediction == 1:
            text = 'Normal Weight'
        elif prediction == 2:
            text = 'Obesity I'
        elif prediction == 3:
            text = 'Obesity II'
        elif prediction == 4:
            text = 'Obesity Type III'
        elif prediction == 5:
            text = 'Overweight Level I'
        elif prediction == 6:
            text = 'Overweight Level II'
        else:
            text = 'Invalid prediction value'
        tab_model.markdown(f'<p style="text-align:center; font-size:26px; font-weight:bold;">{text}</p>',
                           unsafe_allow_html=True)

    st.write("**Developed By: Meltem & Simay**", unsafe_allow_html=True)



if __name__ == '__main__':
    main()

    
    
    
    
    
    
    
    
    
    
    
    
  
    
  