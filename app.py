import streamlit as st

from apputil import *

# Load Titanic dataset
df = load_titanic()

st.write(
    '''
# Titanic Visualization 1
'''
)
st.write("Was survival most predicted by class, sex, or age?")

# Generate and display the figure
fig1 = visualize_demographic()
st.plotly_chart(fig1, use_container_width=True)

st.write(
    '''
# Titanic Visualization 2
'''
)
st.write("Does the average ticket fare increase with family size "
         "within each passenger class?")

# Generate and display the figure
fig2 = visualize_families()
st.plotly_chart(fig2, use_container_width=True)
