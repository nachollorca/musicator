import streamlit as st
from src.song import Song
from src.model import NeuralNetworkModel
import time


st.set_page_config(layout="centered", page_title="Musicator")


# SIDEBAR
with st.sidebar:
    """
    ### :link: Project links
     - :rocket: [Full jupyter notebook](https://github.com/nachollorca/musicator/blob/master/nb.ipynb) describing in depth the characteristics of the dataset, neural network, training process and conclusions.
     - :smiley_cat: [Source code repository](https://github.com/nachollorca/musicator/)
     - :floppy_disk: [Original dataset](https://huggingface.co/datasets/marsyas/gtzan/blob/main/data/genres.tar.gz)

    ### :notes: Download Creative Common songs to classify
     - Nothing
     - More nothing

    ### :man: About me
    My name is Nacho, and I enjoy doing things with data. You can find some other projects in my [GitHub]().
    """


# MAIN PAGE
st.header(':sparkles::musical_note: Musicator :musical_note::sparkles:')
st.subheader('The _limited_ song genre classifier')
"""
Musicator is an OOP-Python program that uses a neural network trained on the GTZAN dataset to guess the genre of a song among blues, classical, country, disco, hiphop, jazz, metal, pop, reggae and rock.
"""

how, why, go = st.tabs(["How it works", "Why is it limited", "**:dizzy: Try it out!:dizzy:**"])

with how:
    st.markdown(
    """
     1. Musicator makes use of the **GTZAN dataset**, a compilation of 1000 audio files of 30 seconds. The audios are equally distributed (manually labeled) into 10 genres (100 audios per group). Each audio is represented through 57 numeric audio features, both from the time and frequency domains.
     2. The numerical features of each are **standardized**.
     3. The standardized values were used to **train a shallow Neural Network**.
     4. The **best performing checkpoint** state of the NN on the validation set was **stored**.
     5. A feature extractor that can **replicate the exact values from the audio clips in GTZAN** is used to **extract the features from the new song** that we want to classify.
     6. The extracted features are **standardized** through the scaler state saved in 3.
     7. The standardized values of the new song are given to the stored model **checkpoint** from 4. to **infer its genre**.
    """
    )
    st.image("media/diagram.png")

with why:
    st.markdown(
    """
    The model may fail to classify songs into the genre you believe it belongs because...
    
    ### GTZAN is far from perfect
    The GTZAN dataset has plenty of flaws. If it is used, it is because of the absence of other viable datasets, due to music labels being reticent to give their product free for research. Even for GTZAN, the ways in which the dataset was gathered could be brought in to question... To mention some of its defects:
     - 1000 tracks are quite few for machine learning problems
     - Some audios are distorted
     - Some audios are duplicated
     - Some audios are different extracts from the same song (or different versions)
     - Some artists are overrepresented in their genres (35% of the reggae labeled audios belong to Bob Marley!)
    
    ### Musicator answers an ill-posed problem
    The task here is inherently difficult to solve or lacks a unique solution due to various factors, mainly:
    - **Ambiguity of Genre Labels**: Music genre labels can be somewhat subjective and may not have universally agreed-upon definitions. Different people may classify the same piece of music into different genres based on their interpretations and criteria.
    - **Overlapping Genres**: Many songs can belong to multiple genres simultaneously.
    - **Non-represented Genres**: the 10 genres that GTZAN contemplates arguably cover a narrow span of the music spectrum. For example, there is no label for electronic or latin music. 
    
    In any case, seeing how the model reacts to these ambiguity factors is always interesting, i.e., seeing which kind of mistakes does the model make.
    """
    )

# ACTUAL CODE
with go:
    upload = st.file_uploader("Upload a song",type=["wav", "mp3"])
    st.write(upload)
    classify = st.button(":dizzy: **Classify** :dizzy:", use_container_width=True)

    if classify and upload is not None:

        with st.status("On it...", expanded=True) as status:
            st.write(":musical_note: Fetching song")
            song = Song(upload)
            song.load()
            time.sleep(1)

            st.write(":mag_right: Extracting numerical features (takes ca. 10s)")
            song.extract_features()

            st.write(":mag_right: Standardizing features")
            time.sleep(1)
            song.standardize_features()

            st.write(":brain: Loading model checkpoint")
            model = NeuralNetworkModel()
            model.load_checkpoint()
            time.sleep(1)

            st.write(":gear: Running inference")
            time.sleep(1)
            out = model.infer(song.standardized_features)

            status.update(label=":tada: **et voilà!** :tada:", state="complete", expanded=False)

        st.dataframe(out, use_container_width=True)
