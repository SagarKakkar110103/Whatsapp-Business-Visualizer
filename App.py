import nltk
import streamlit as st
import pandas as pd
import preprocessor
import Helper
import matplotlib.pyplot as plt
import plotly.express as px
from streamlit_option_menu import option_menu

nltk.downloader.download('vader_lexicon')
st.sidebar.header('Whatsapp Bussiness Visualizer')

st.markdown(
    f'''
        <style>
            .sidebar .sidebar-content {{
                width: 375px;
            }}
        </style>
    ''',
    unsafe_allow_html=True
)

selected = option_menu(
    menu_title=None,
    options=["Home", "Dashboard"],
    icons=["house", "bar-chart-line-fill"],
    default_index=0,
    orientation="horizontal",
)
if selected == "Home":
    uploaded_file = st.sidebar.file_uploader("Choose a file")
    with st.sidebar:
        selected = option_menu(menu_title = '',
                               options=['User', 'Timeline','Words', "Emoji" ,'Wordcloud', 'Contribution'])
    # Main heading
    st. markdown("<h1 style='text-align: center; color: grey;'>Whatsapp Bussiness Visualizer</h1>", unsafe_allow_html=True)
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        # yeh data byte data ka stream hai isse string mein convert krna pdeega
        data = bytes_data.decode('utf-8')
        # ab file ka data screen pe dikhne lagega
        df = preprocessor.preprocess(data)
        df2 = preprocessor.preprocess2(data)
        df3 = preprocessor.preprocess3(data)
        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        def sentiment(d):
            if d["pos"] >= d["neg"] and d["pos"] >= d["nu"]:
                return 1
            if d["neg"] >= d["pos"] and d["neg"] >= d["nu"]:
                return -1
            if d["nu"] >= d["pos"] and d["nu"] >= d["neg"]:
                return 0


        # Object
        sentiments = SentimentIntensityAnalyzer()
        df["pos"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
        df["neg"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
        df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]
        df['value'] = df.apply(lambda row: sentiment(row), axis=1)
        st.dataframe(df)


        # fetch unique user
        user_list = df['user'].unique().tolist()
        user_list.remove('group_notification')
        user_list.sort()
        user_list.insert(0, 'Overall')
        selected_user = st.sidebar.selectbox('show analysis wrt', user_list)
        if st.sidebar.button('Show Analysis'):
            num_messages, words, num_media_messages, num_of_links = Helper.fetch_stats(selected_user, df2)
            st.title('Top Statistics')
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown("""
                            <style>
                            .big-font {
                                font-size:30px !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)

                st.markdown('<p class="big-font">Total Messages </p>', unsafe_allow_html=True)
                st.title(num_messages)
            with col2:
                st.markdown("""
                <style>
                .big-font {
                    font-size:30px !important;
                }
                </style>
                """, unsafe_allow_html=True)

                st.markdown('<p class="big-font">Total Words </p>', unsafe_allow_html=True)
                st.title(words)
            with col3:

                st.markdown("""
                            <style>
                            .big-font {
                                font-size:30px !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)

                st.markdown('<p class="big-font">Media Messages </p>', unsafe_allow_html=True)
                st.title(num_media_messages)
            with col4:
                st.markdown("""
                            <style>
                            .big-font {
                                font-size:30px !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)

                st.markdown('<p class="big-font">Links Shared </p>', unsafe_allow_html=True)
                st.title(num_of_links)

            #timeline
            # monthly
            if selected == 'Timeline':
                col1, col2 = st.columns(2)
                with col1:
                    timeline = Helper.day_timeline('Sagar', df)
                    fig = px.line(timeline, x='day_name', y='message', title='User Activity DayWise',
                     width=400, height=400)

                    fig
                # daily
                with col2:
                    timeline = Helper.day_timeline('Arihan Mech', df)
                    fig = px.line(timeline, x='day_name', y='message', title='User Activity DayWise',
                     width=400, height=400)
                    fig
            # finding the busiest users in the group (Group - level)
            if selected == 'User':
                if selected_user == 'Overall':
                    st.title('Most Busy Users')
                    x, new_df = Helper.most_busy_users(df)
                    fig, ax = plt.subplots()
                    #col1, col2 = st.columns(2)
                    names = new_df['names']
                    percentage = new_df['percentage']
                    fig = px.bar(new_df, x=names, y=percentage, color=names)
                    fig

            # WordCloud
            if selected == 'Wordcloud':
                df_wc = Helper.create_wordcloud(selected_user, df)
                fig, ax = plt.subplots()
                plt.imshow(df_wc)
                st.pyplot(fig)
            if selected == "Contribution":
            # Most Positive, Negitive, Neutral user...
                if selected_user == 'Overall':
                #    col1, col2, col3 = st.columns(3)
                #    with col1:
                        st.markdown("<h3 style='text-align: center; color: orange;'>Most Positive Users</h3>",unsafe_allow_html=True)
                        af = df['user'][df['value'] == 1]
                        x = af.value_counts()
                        fig = px.bar(af, y=x.values, x=x.index, color=x)
                        fig
                #    with col2:
                        st.markdown("<h3 style='text-align: center; color: blue;'>Most Neutral Users</h3>",unsafe_allow_html=True)
                        af = df['user'][df['value'] == 0]
                        x = af.value_counts()
                        fig = px.bar(af, y=x.values, x=x.index, color=x)
                        fig
                #    with col3:
                        st.markdown("<h3 style='text-align: center; color: green;'>Most Negitive Users</h3>",unsafe_allow_html=True)
                        af = df['user'][df['value'] == -1]
                        x = af.value_counts()
                        fig = px.bar(af, y=x.values, x=x.index, color=x)
                        fig
            # most common words
            if selected == 'Words':
                #col1, col2, col3 = st.columns(3)

                #with col1:
                    try:
                        st.markdown("<h3 style='text-align: center; color: orange;'>Most Positive Words</h3>",
                                    unsafe_allow_html=True)
                        most_common_df = Helper.most_common_words(selected_user, df3, 1)
                        fig, ax = plt.subplots()
                        word = most_common_df['word']
                        number =most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color = word)
                        fig
                    except:
                        pass
                #with col2:
                    try:
                        st.markdown("<h3 style='text-align: center; color: blue;'>Most Neutral words</h3>",
                                    unsafe_allow_html=True)
                        most_common_df = Helper.most_common_words(selected_user, df3, 0)
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word)
                        fig
                    except:
                        pass
                #with col3:
                    try:
                        st.markdown("<h3 style='text-align: center; color: green;'>Most Negitive words</h3>",
                                    unsafe_allow_html=True)
                        most_common_df = Helper.most_common_words(selected_user, df3, -1)
                        fig, ax = plt.subplots()
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word)
                        fig
                    except:
                        pass
            # emoji analysis
            if selected == 'Emoji':
                try:
                    emoji_df, p, neg, nu = Helper.emoji_helper(selected_user, df)
                    st.title("Emoji Analysis")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        try:
                            st.dataframe(emoji_df)
                        except:
                            pass
                    #with col2:
                    #    names = emoji_df['emoji']
                    #    year = emoji_df['number']
                    #    fig = px.pie(emoji_df, values=year, names= names)
                    #    fig.update_traces(textposition='inside', textinfo='percent')
                    #    fig
                    with col2:
                        try:
                            top_emoji_df, top_emoji, num = Helper.top_emoji(selected_user, emoji_df)
                            st.dataframe(top_emoji_df, width=40, height=400)
                        except:
                            pass
                    with col3:
                        try:
                            top_emoji_df, top_emoji, num = Helper.top_emoji(selected_user, emoji_df)
                            arr = [int((p / (p + neg + nu)) * 100), int((neg / (p + neg + nu)) * 100),
                                   int((nu / (p + neg + nu)) * 100)]
                            af = pd.DataFrame({'sentiment': ['positive', 'negitive', 'neutral'], 'percentage': arr, 'top_emoji':top_emoji})
                            fig = px.pie(af, values='percentage', names='sentiment',hover_data=['top_emoji'] ,labels={'top_emoji':'top_emoji' })
                            fig.update_traces(textposition='inside', textinfo='percent')
                            fig
                        except:
                            try:
                                arr = [int((p/(p+neg+nu))*100), int((neg/(p+neg+nu))*100), int((nu/(p+neg+nu))*100)]
                                af = pd.DataFrame({'sentiment': ['positive', 'negitive', 'neutral'], 'percentage': arr})
                                fig = px.pie(af, values='percentage', names='sentiment')
                                fig.update_traces(textposition='inside', textinfo='percent')
                                fig
                            except:
                                pass
                except:
                    pass

if selected == "Dashboard":

        import openai
        from streamlit_chat import message

        openai.api_key = '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'


        # This function uses the OpenAI Completion API to generate a
        # response based on the given prompt. The temperature parameter controls
        # the randomness of the generated response. A higher temperature will result
        # in more random responses,
        # while a lower temperature will result in more predictable responses.
        def generate_response(prompt):
            completions = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=1024,
                n=1,
                stop=None,
                temperature=0.5,
            )

            message = completions.choices[0].text
            return message


        st.title("chatBot")

        if 'generated' not in st.session_state:
            st.session_state['generated'] = []

        if 'past' not in st.session_state:
            st.session_state['past'] = []


        def get_text():
            input_text = st.text_input("You: ", "", key="input")
            return input_text


        user_input = get_text()
        if user_input[:4] == 'user':
            # Main heading
            a, b, c = user_input.split(",")
            selecte_user = [b, c]
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                # yeh data byte data ka stream hai isse string mein convert krna pdeega
                data = bytes_data.decode('utf-8')
                # ab file ka data screen pe dikhne lagega
                df = preprocessor.preprocess(data)
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                def sentiment(d):
                    if d["pos"] >= d["neg"] and d["pos"] >= d["nu"]:
                        return 1
                    if d["neg"] >= d["pos"] and d["neg"] >= d["nu"]:
                        return -1
                    if d["nu"] >= d["pos"] and d["nu"] >= d["neg"]:
                        return 0


                # Object
                sentiments = SentimentIntensityAnalyzer()
                df["pos"] = [sentiments.polarity_scores(i)["pos"] for i in df["message"]]  # Positive
                df["neg"] = [sentiments.polarity_scores(i)["neg"] for i in df["message"]]  # Negative
                df["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df["message"]]
                df['value'] = df.apply(lambda row: sentiment(row), axis=1)
                def sentiment2(d):
                    return d["pos"] - d["neg"]
                df['score'] = df.apply(lambda row: sentiment2(row), axis=1)
                # daily 1
                st.title('Timeline')
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        timeline = Helper.day_timeline(selecte_user[0], df)
                        fig = px.line(timeline, x='day_name', y='message', title = selecte_user[0] +' DayWise activity',
                                      width=400, height=400)
                        fig
                    except:
                        pass
                # daily 2
                with col2:
                    try:
                        timeline = Helper.day_timeline(selecte_user[1], df)
                        fig = px.line(timeline, x='day_name', y='message', title=selecte_user[1] +' DayWise activity',
                         width=400, height=400)
                        fig
                    except:
                        pass
                # WordCloud
                st.title('WordCloud')
                col1, col2 = st.columns(2)
                with col1:
                    df_wc = Helper.create_wordcloud(selecte_user[0], df)
                    fig, ax = plt.subplots()
                    plt.imshow(df_wc)
                    st.pyplot(fig)
                with col2:
                    df_wc = Helper.create_wordcloud(selecte_user[1], df)
                    fig, ax = plt.subplots()
                    plt.imshow(df_wc)
                    st.pyplot(fig)
                st.title('Most Positive Words')
                col1, col2= st.columns(2)
                with col1:
                    try:
                        most_common_df = Helper.most_common_words(selecte_user[0], df, 1)
                        fig, ax = plt.subplots()
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color = word)
                        fig
                    except:
                        pass
                with col2:
                    try:
                        most_common_df = Helper.most_common_words(selecte_user[1], df, 1)
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word)
                        fig
                    except:
                        pass
                st.title('Most Negitive Words')
                col1, col2 = st.columns(2)

                with col1:
                    try:
                        most_common_df = Helper.most_common_words(selecte_user[0], df, -1)
                        fig, ax = plt.subplots()
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word)
                        fig
                    except:
                        pass
                with col2:
                    try:
                        most_common_df = Helper.most_common_words(selecte_user[1], df, -1)
                        word = most_common_df['word']
                        number = most_common_df['number']
                        fig = px.bar(most_common_df, y=number, x=word, color=word)
                        fig
                    except:
                        pass
        elif user_input[:7] == 'product':
            a, b = user_input.split(",")
            data_points = b
            if uploaded_file is not None:
                bytes_data = uploaded_file.getvalue()
                # yeh data byte data ka stream hai isse string mein convert krna pdeega
                data = bytes_data.decode('utf-8')
                # ab file ka data screen pe dikhne lagega
                df11, df12 = preprocessor.preprocessor5(data, int(data_points))
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                def sentiment(d):
                    if d["pos"] >= d["neg"] and d["pos"] >= 0.1:
                        return 1
                    elif d["neg"] >= d["pos"] and d["neg"] >= 0.1:
                        return -1
                    else:
                        return 0
                # Object
                sentiments = SentimentIntensityAnalyzer()
                df11["pos"] = [sentiments.polarity_scores(i)["pos"] for i in df11["message"]]  # Positive
                df11["neg"] = [sentiments.polarity_scores(i)["neg"] for i in df11["message"]]  # Negative
                df11["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df11["message"]]
                df11['value'] = df11.apply(lambda row: sentiment(row), axis=1)

                df12["pos"] = [sentiments.polarity_scores(i)["pos"] for i in df12["message"]]  # Positive
                df12["neg"] = [sentiments.polarity_scores(i)["neg"] for i in df12["message"]]  # Negative
                df12["nu"] = [sentiments.polarity_scores(i)["neu"] for i in df12["message"]]
                df12['value'] = df12.apply(lambda row: sentiment(row), axis=1)


                def sentiment2(d):
                    return d["pos"] - d["neg"]


                df11['score'] = df11.apply(lambda row: sentiment2(row), axis=1)
                df12['score'] = df12.apply(lambda row: sentiment2(row), axis=1)


                st.title('')
                col1, col2 = st.columns(2)
                with col1:
                    p = len(df11[df11['value'] == 1])
                    neg = len(df11[df11['value'] == -1])
                    nu = len(df11[df11['value'] == 0])
                    arr = [int((p / (p + neg + nu)) * 100), int((neg / (p + neg + nu)) * 100),
                           int((nu / (p + neg + nu)) * 100)]
                    af11 = pd.DataFrame(
                        {'sentiment': ['positive', 'negitive', 'neutral'], 'percentage': arr})
                    fig = px.pie(af11, values='percentage', names='sentiment',
                                 )
                    fig.update_traces(textposition='inside', textinfo='percent')
                    fig

                with col2:
                    p = len(df12[df12['value'] == 1])
                    neg = len(df12[df12['value'] == -1])
                    nu = len(df12[df12['value'] == 0])
                    arr = [int((p / (p + neg + nu)) * 100), int((neg / (p + neg + nu)) * 100),
                           int((nu / (p + neg + nu)) * 100)]
                    af12 = pd.DataFrame(
                        {'sentiment': ['positive', 'negitive', 'neutral'], 'percentage': arr})
                    fig = px.pie(af12, values='percentage', names='sentiment',
                                 )
                    fig.update_traces(textposition='inside', textinfo='percent')
                    fig
                col1, col2 = st.columns(2)
                with col1:
                    try:
                        import plotly.express as px
                        fig = px.scatter(df11, x='Date', y='score', color='score');
                        fig.update_yaxes(tickvals=[-1, 0, 1])
                        fig
                    except:
                        pass
                with col2:
                    try:
                        import plotly.express as px
                        fig = px.scatter(df12, x='Date', y='score', color='score');
                        fig.update_yaxes(tickvals=[-1, 0, 1])
                        fig
                    except:
                        pass
        else:
            output = generate_response(user_input)
            st.session_state.past.append(user_input)
            st.session_state.generated.append(output)

            if st.session_state['generated']:

                for i in range(len(st.session_state['generated']) - 1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    
