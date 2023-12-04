import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re


logo_path = "images/logo.png"  
st.image(logo_path, width=350)  

def load_data(data):
    df = pd.read_csv("data/udemy_courses_clean.csv")
    return df

def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat

def merge_sort(arr):
    if len(arr) > 1:
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0

        while i < len(left_half) and j < len(right_half):
            if left_half[i][1] > right_half[j][1]:
                arr[k] = left_half[i]
                i += 1
            else:
                arr[k] = right_half[j]
                j += 1
            k += 1

        while i < len(left_half):
            arr[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            arr[k] = right_half[j]
            j += 1
            k += 1

def get_recommendation(keyword, cosine_sim_mat, df, num_of_rec=10):
    
    tokens = re.findall(r'\b\w+\b', keyword.lower())
    keywords = [token for token in tokens if token not in ENGLISH_STOP_WORDS]

    if not keywords:
        st.warning(f"No significant keywords found in the input '{keyword}'.")
        return pd.DataFrame(columns=['course_title', 'similarity_score', 'url', 'price', 'num_subscribers'])

    keyword_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, keywords)) + r')\b', flags=re.IGNORECASE)

    filtered_df = df[df['course_title'].apply(lambda title: bool(re.search(keyword_pattern, title)))]

    if filtered_df.empty:
        st.warning(f"No courses found containing the significant keywords in the input '{keyword}'.")
        return pd.DataFrame(columns=['course_title', 'similarity_score', 'url', 'price', 'num_subscribers'])

    recommended_courses = set()
    
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(filtered_df['course_title'])
    cosine_sim_mat_filtered = cosine_similarity(cv_mat)

    filtered_df.reset_index(drop=True, inplace=True)

    results = []
    for idx in range(len(filtered_df)):
        sim_scores = list(enumerate(cosine_sim_mat_filtered[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        selected_course_indices = [i[0] for i in sim_scores[1:num_of_rec + 1]]

        for selected_idx in selected_course_indices:
            if selected_idx not in recommended_courses:
                rec_title = filtered_df.loc[selected_idx, 'course_title']
                rec_score = sim_scores[selected_idx][1]
                rec_url = filtered_df.loc[selected_idx, 'url']
                rec_price = filtered_df.loc[selected_idx, 'price']
                rec_num_sub = filtered_df.loc[selected_idx, 'num_subscribers']

                results.append([rec_title, rec_score, rec_url, rec_price, rec_num_sub])
                recommended_courses.add(selected_idx)

    if results:
        merge_sort(results)
        result_df = pd.DataFrame(results, columns=['course_title', 'similarity_score', 'url', 'price', 'num_subscribers'])
        return result_df
    else:
        st.warning(f"No recommendations found for courses containing the significant keywords in the input '{keyword}'.")
        return pd.DataFrame(columns=['course_title', 'similarity_score', 'url', 'price', 'num_subscribers'])


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #B5F0AB;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score: </span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link for the course</a></p>
<p style="color:blue;"><span style="color:black;">💲Price: </span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Number of students enrolled: </span>{}</p>
</div>
"""


def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term)]
    return result_df


def feedback_form():
    st.subheader("Feedback Form")
    feedback_text = st.text_area("Provide your feedback or review:", "")
    submit_button = st.button("Submit Feedback")

    if submit_button and feedback_text:
        
        feedback_data = pd.DataFrame({'Feedback': [feedback_text]})
        feedback_data.to_csv('feedback.csv', mode='a', header=False, index=False)
        st.success("Feedback submitted successfully!")


def main():
    menu = ["🏠 Home", "🎓 Recommend Online Courses", "✉️ Submit a Feedback", "📌 About"]
    choice = st.sidebar.selectbox("Menu", menu)

    df = load_data("data/udemy_course_data.csv")

    if choice == "🏠 Home":
        st.subheader("Welcome to CourseGuide!")
        st.dataframe(df.head(3690))

    elif choice == "🎓 Recommend Online Courses":
        st.subheader("Search a course you want to enroll in.")
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['course_title'])
        search_term = st.text_input("Search")
        num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)

        if st.button("Recommend"):
            if search_term is not None:
                results = get_recommendation(search_term, cosine_sim_mat, df, num_of_rec)
                if not results.empty:
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_score = row[1][1]
                        rec_url = row[1][2]
                        rec_price = row[1][3]
                        rec_num_sub = row[1][4]

                        st.write(RESULT_TEMP.format(rec_title, rec_score, rec_url, rec_price, rec_num_sub), unsafe_allow_html=True)
                else:
                    st.warning(f"No recommendations found for '{search_term}'.")
    
    elif choice == "✉️ Submit a Feedback":
        feedback_form()

    else:
        st.subheader("About")
        st.text("CourseGuide is a Massive Open Online Course Recommender System and is programmed by Jathniel Ramos & Fritzjan Marzan.")

if __name__ == '__main__':
    main()

