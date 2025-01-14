import streamlit as st
import random

class Sidebar():
    def _init_(self) -> None:
        self.model_name = None
        self.confidence_threshold = None
        self.title_img = None
        
        self._titleimage()
        self._model()
        self._confidencethreshold()
        
    def _titleimage(self):
        st.sidebar.title("X-Ray Diagnosis")
        
    def _model(self):
        st.sidebar.markdown('## Step 1: Choose model')

        self.model_name = st.sidebar.selectbox(
            label = 'Which Model would you like to choose from?',
            options = [
                'YoloV8',
                'FastRCNN with ResNet',
                'VGG16'
            ],
            index = 0,
        )
        
    def _confidencethreshold(self):
        st.sidebar.markdown('## Step 2: Choose Confidence Threshold')
        
        self.confidence_threshold = st.sidebar.slider(
            label = 'What should be the minimum confidence threshold?',
            min_value = 0.0,
            max_value = 1.0,
            value = 0.5,
            step = 0.1
        )
    
    def show(self):
        # Return the selected model name
        return self.model_name