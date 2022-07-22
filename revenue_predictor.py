import streamlit as st
import pickle

import numpy as np

xgb_model = pickle.load(open('xgbmodel.pkl','rb'))

@st.cache()
  
# defining the function which will make the prediction using the data which the user inputs 
def prediction(log_budget, male_crew_count, male_cast_count, cast_count, crew_count, female_crew_count, people_in_dept_Editing, people_in_dept_Production, people_in_job_Producer, people_in_job_Casting,
           people_in_dept_Visual_Effects, people_in_job_Editor, people_in_job_Art_Direction, female_cast_count, keywords_count, production_companies_count, count_of_tagline_words, people_in_job_Screenplay,
           have_collection, people_in_job_Production_Design, people_in_job_Executive_Producer, prodCountry_United_States_of_America):

    log_budget = np.log(log_budget)

    if have_collection == 'Yes':
        have_collection = 1
    else:
        have_collection = 0

    if prodCountry_United_States_of_America == 'Yes':
        prodCountry_United_States_of_America = 1
    else:
        prodCountry_United_States_of_America = 0

    features = np.array([log_budget, male_crew_count, male_cast_count, cast_count, crew_count, female_crew_count, people_in_dept_Editing, people_in_dept_Production, people_in_job_Producer, people_in_job_Casting,
           people_in_dept_Visual_Effects, people_in_job_Editor, people_in_job_Art_Direction, female_cast_count, keywords_count, production_companies_count, count_of_tagline_words, people_in_job_Screenplay,
           have_collection, people_in_job_Production_Design, people_in_job_Executive_Producer, prodCountry_United_States_of_America])
    # Making predictions 
    log_revenue = xgb_model.predict(features.reshape(1, -1))
    revenue = np.exp(log_revenue)
    return revenue

def main():
    st.markdown("<h1 style='text-align: center; color: yellow;'>TMDB BOX OFFICE REVENUE PREDICTOR</h1>", unsafe_allow_html=True)

    log_budget = st.number_input('Movie Budget (in Dollars):',
                        min_value=1.0, max_value=3.800000e+08,
                        help="please enter budget amount in dollars")

    cast_count = st.number_input('Total Casted People:',
                            min_value=0, max_value=156, step=1,
                            help="please enter total count of casted persons")

    male_cast_count = st.number_input('Number of Actors:',
                            min_value=0, max_value=84, step=1,
                            help="please enter count of casted male persons")

    female_cast_count = st.number_input('Number of Actresses:',
                            min_value=0, max_value=87, step=1,
                            help="please enter count of casted female persons")

    crew_count = st.number_input('Total Crew Members:',
                            min_value=0, max_value=194, step=1,
                            help="please enter total count of crew members")                       

    male_crew_count = st.number_input('Number of Male Crew Members:',
                            min_value=0, max_value=91, step=1, 
                            help="please enter count of male crew members")

    female_crew_count = st.number_input('Number of Female Crew Members:',
                            min_value=0, max_value=30, step=1,
                            help="please enter count of female crew members")

    people_in_dept_Production = st.number_input('Number of Crew members in Production Department:',
                            min_value=0, max_value=34, step=1,
                            help="please enter number of people in production department")

    people_in_job_Producer = st.number_input('Number of Producers:',
                            min_value=0, max_value=12, step=1,
                            help="please enter number of producers")

    people_in_job_Executive_Producer = st.number_input('Number of Executive Producers:',
                                min_value=0, max_value=18, step=1,
                                help="please enter count executive producers in movie")

    people_in_job_Production_Design = st.number_input('Number of Production Designers:',
                                min_value=0, max_value=4, step=1,
                                help="please enter count of Production Designers involved in movie")

    production_companies_count = st.number_input('Number of Production Companies involved:',
                                min_value=0, max_value=17, step=1,
                                help="please enter count of production companies involved in the movie")                            

    people_in_dept_Editing = st.number_input('Number of Crew members in Editing Department:',
                            min_value=0, max_value=14, step=1,
                            help="please enter number of people in editing department")

    people_in_job_Editor = st.number_input('Number of Editors:',
                            min_value=0, max_value=8, step=1,
                            help="please enter number of editors")                                                                                                         

    people_in_dept_Visual_Effects = st.number_input('Number of Crew members in Visual Effect Department:',
                        min_value=0, max_value=52, step=1,
                        help="please enter number of people in visual effects department")
                                           

    people_in_job_Art_Direction = st.number_input('Number of Art Directors:',
                            min_value=0, max_value=11, step=1,
                            help="please enter number of art directors")

    people_in_job_Screenplay = st.number_input('Number of people working on ScreenPlay:',
                                min_value=0, max_value=12, step=1,
                                help="please enter count of crew members with job in ScreenPlay")

    people_in_job_Casting = st.number_input('Number of people working on Casting:',
                            min_value=0, max_value=6, step=1,
                            help="please enter number of people in working on casting")                         

    keywords_count = st.number_input('Count of Keywords:',
                                min_value=0, max_value=150, step=1,
                                help="please enter count of keywords")

    count_of_tagline_words = st.number_input('Length of Tagline:',
                                min_value=0, max_value=43, step=1,
                                help="please enter count of words in movie tagline")
                           
    have_collection = st.selectbox(
        'Does the movie belongs to any Collection?',
        ('Yes', 'No'))

    prodCountry_United_States_of_America = st.selectbox(
        'Whether the movie is produced in USA',
        ('Yes', 'No'))

    if st.button("Predict"): 
        result = prediction(log_budget, male_crew_count, male_cast_count, cast_count, crew_count, female_crew_count, people_in_dept_Editing, people_in_dept_Production, people_in_job_Producer, people_in_job_Casting,
           people_in_dept_Visual_Effects, people_in_job_Editor, people_in_job_Art_Direction, female_cast_count, keywords_count, production_companies_count, count_of_tagline_words, people_in_job_Screenplay,
           have_collection, people_in_job_Production_Design, people_in_job_Executive_Producer, prodCountry_United_States_of_America) 
        st.success('Predicted Revenue of Movie: {} US Dollars'.format(result.item()))
        

if __name__=='__main__': 
    main()
