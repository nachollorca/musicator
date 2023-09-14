import streamlit as st
from src.song import Song
from src.model import NeuralNetworkModel

# setup page
st.set_page_config(layout="centered", page_title="Musicator")
st.write("# :sparkles::musical_note: Musicator :musical_note::sparkles:")
st.write("### Classify a song within 10 genres!")
st.write(
"""Musicator is the Streamlit addaptation of a little project I developed in 2021 attempting to use Machine Learning to classify songs by genre. The code is open source and available [here](https://github.com/nachollorca/musicator/)."""
)

st.write(
"""The characteristics of the dataset, neural network, training process and conclusions are explained in depth in [this notebook](https://github.com/nachollorca/musicator/blob/master/nb.ipynb)."""
)

st.markdown(
"""
In short:
- The numerical features of the GTZAN dataset are used to train a Neural Network
- A feature extractor that can replicate the exact values from the audio clips in GTZAN is used to extract the features from the song we want to predict
- Such numerical features are given to the neural network model to infer its genre
"""
)

st.image("media/diagram.png")

st.write("### Try yourself!")
upload = st.file_uploader("Upload a song", type=["wav"])

if upload is not None:
    song = Song(upload)
    song.load()
    song.extract_features()
    song.standardize_features()

    model = NeuralNetworkModel()
    model.load_checkpoint()
    st.write(model.infer(song.standardized_features))