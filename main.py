import re
from numpy import tile
import nltk
import pandas as pd
from nltk import *
from nltk.sentiment import SentimentIntensityAnalyzer
from regex import regex
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
nltk.download('vader_lexicon')


uploaded_file = st.sidebar.file_uploader("Upload the chat file:")

if uploaded_file is not None:

    def ValidDateAndTime(s):
        pattern = '[], []'
        result = re.match(pattern, s)
        if result:
            return True
        return False

    def SplitData(line):
        splitLine = line.split(']')
        dateTime = splitLine[0]
        dateTime = dateTime[1:]
        message = ' '.join(splitLine[1:])
        splitMessage = message.split(': ')
        sender = splitMessage[0]
        message = ' '.join(splitMessage[1:])

        return dateTime, sender, message

    # List to keep track of data so it can be used by a Pandas dataframe
    parsedData = []
    # Uploading exported chat file
    conversationPath = 'wh_chat.txt'  # chat file
    fp = uploaded_file

    # Skipping first line of the file because contains
    # information related to something about end-to-end encryption
    fp.readline()
    messageBuffer = []
    dateTime, sender = None, None

    while True:

        line = fp.readline().decode('utf-8')
        if not line:
            break
        line = line.strip()

        if ValidDateAndTime(line):
            if len(messageBuffer) >= 0:
                parsedData.append([dateTime, sender, ' '.join(messageBuffer)])
            messageBuffer.clear()
            dateTime, sender, message = SplitData(line)
            messageBuffer.append(message)

        else:
            messageBuffer.append(line)

    # Initialising a pandas Dataframe that contains the asnwer
    df_final_result = pd.DataFrame(
        parsedData, columns=['Date & Time', 'Sender', 'Message'])

    df_final_result = df_final_result.dropna()
    df_final_result = df_final_result.reset_index(drop=True)

    sentiments = SentimentIntensityAnalyzer()

    # Append the % of positive of the mesages
    df_final_result["positive"] = [sentiments.polarity_scores(
        i)["pos"] for i in df_final_result["Message"]]

    # Append the % of negative of the mesages
    df_final_result["negative"] = [sentiments.polarity_scores(
        i)["neg"] for i in df_final_result["Message"]]

    # Append the % of neutral of the mesages
    df_final_result["neutral"] = [sentiments.polarity_scores(
        i)["neu"] for i in df_final_result["Message"]]

    st.dataframe(df_final_result)

    total_num_messages = len(df_final_result)

    # Graph 1 show the number of messages and the sender
    df_groupby_Sender_number_messages = df_final_result.groupby(
        ["Sender"], sort=False)["Message"].count().reset_index(drop=False)

    fig_number_of_messages_per_sender = px.treemap(df_groupby_Sender_number_messages,
                                                   path=['Message'],
                                                   title="Number of Messages per Sender",
                                                   values='Message',
                                                   color='Sender')

    st.plotly_chart(fig_number_of_messages_per_sender,
                    use_container_width=True)

    # Graph #2 The global conversation percentage of positive negative and neutral.
    df_groupby_Sender = df_final_result.sum()

    data = {'values': [df_groupby_Sender["positive"]/total_num_messages, df_groupby_Sender["neutral"]/total_num_messages, df_groupby_Sender["negative"]/total_num_messages],
            'type': ["positive", "neutral", "negative"]}
    indexes = ['1', '2', '2']
    df = pd.DataFrame(data, index=indexes)

    fig_sentiment_global = px.pie(df,
                                  title="Percentage of sentiment of all the messages",
                                  values='values',
                                  names='type',
                                  color='type')

    st.plotly_chart(fig_sentiment_global)

    # Graph 2 The  conversation percentage of positive negative and neutral.
    df_groupby_Sender_avg = df_final_result.groupby(
        ["Sender"], sort=False).mean().reset_index(drop=False)

    fig_percentage_of_message_sentiment = go.Figure(data=[
        go.Bar(name='sentiment Positive',
               x=df_groupby_Sender_avg["Sender"], y=df_groupby_Sender_avg["positive"]),
        go.Bar(name='sentiment Negative',
               x=df_groupby_Sender_avg["Sender"], y=df_groupby_Sender_avg["negative"]),
        go.Bar(name='sentiment Neutral',
               x=df_groupby_Sender_avg["Sender"],  y=df_groupby_Sender_avg["neutral"]),

    ])

    #title="average of sentiment messages per person"

    fig_percentage_of_message_sentiment.update_layout(
        barmode='group',
        title="Average of sentiment messages per person")

    st.plotly_chart(fig_percentage_of_message_sentiment,
                    use_container_width=True)

    # line graph whith the timeline and the total amount of positive, neutral and negative.

    fig_line_graph = go.Figure()

    fig_line_graph.add_trace(go.Scatter(
        x=df_final_result["Date & Time"],
        y=df_final_result["positive"],
        mode='markers',
        name='sentiment positive'))

    fig_line_graph.add_trace(go.Scatter(
        x=df_final_result["Date & Time"],
        y=df_final_result["negative"],
        mode='markers',
        name='sentiment negative '))

    fig_line_graph.add_trace(go.Scatter(
        x=df_final_result["Date & Time"],
        y=df_final_result["neutral"],
        mode='markers',
        name='sentiment neutral'))

    fig_line_graph.update_layout(
        autosize=False,
        width=900,
        height=500,
        title="Evolution of the sentiments of the conversation",
        paper_bgcolor="LightSteelBlue",
    )
    st.plotly_chart(fig_line_graph)
