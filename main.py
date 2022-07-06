import re
import nltk
import pandas as pd
from nltk import *
from nltk.sentiment import SentimentIntensityAnalyzer
from regex import regex
import streamlit as st

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

    parsedData = []  # List to keep track of data so it can be used by a Pandas dataframe
    ### Uploading exported chat file
    conversationPath = 'wh_chat.txt'  # chat file
    fp=uploaded_file
    ### Skipping first line of the file because contains information related to something about end-to-end encryption
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
    df = pd.DataFrame(parsedData, columns=['Date & Time', 'Sender', 'Message'])  # Initialising a pandas Dataframe.

    df = df.dropna()
    df = df.reset_index(drop=True)
    print(df.head(5))


    sentiments=SentimentIntensityAnalyzer()
    df["positive"]=[sentiments.polarity_scores(i)["pos"] for i in df["Message"]]
    df["negative"]=[sentiments.polarity_scores(i)["neg"] for i in df["Message"]]
    df["neutral"]=[sentiments.polarity_scores(i)["neu"] for i in df["Message"]]

    st.dataframe(df.tail(20))