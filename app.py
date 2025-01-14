import streamlit as st  
from ultralytics import YOLO
import random
import PIL
from PIL import Image, ImageOps
import numpy as np
import torchvision
import torch
import os

from sidebar import Sidebar
import rcnnres, vgg
# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# Add warning about model limitations
st.warning("""
⚠️ Important Note:
- The models currently use pretrained weights as custom weights are not available
- For accurate fracture detection, the models need to be trained specifically on X-ray images
""")

# Sidebar
sidebar = Sidebar()
model = sidebar.show()

#Main Page
st.title("X-Ray Diagnosis")
st.write("The Application provides X-Ray Diagnosis using multiple state-of-the-art computer vision models such as Yolo V8, ResNet, AlexNet, and CNN. To learn more about the app - Try it now!")

st.markdown("""
<style>
.reportview-container {
    background: #f0f2f6
}
</style>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["About", "Upload & Test"])

with tab1:
   st.markdown("### Overview")
   st.text_area(
    "TEAM MEMBERS",
    "Bhuvanesh Satish, Umesh Kumar N S, Abhilash R, Ashutosh Meshram",
    height=100
   )
   
   st.markdown("#### Network Architecture")
   st.markdown("""
   The X-Ray Diagnosis system uses three different deep learning models:
   1. YOLOv8 - For fast and accurate object detection
   2. Faster R-CNN with ResNet - For high accuracy detection with feature extraction
   3. VGG16 - For robust feature learning and classification
   """)
   
   st.markdown("#### Models Used")
   st.markdown("##### YoloV8")
   st.text_area(
       "Description",
       "In this X-Ray Diagnosis project, we're using a lightweight and efficient version of the YOLO v8 algorithm called YOLO v8 Nano (yolov8n). This algorithm is tailored for systems with limited computational resources. We start by training the model on a dataset of X-ray images that are labeled to show where fractures are. We specify the training settings in a YAML file. The model is trained for 50 epochs, and we save its progress every 25 epochs to keep the best-performing versions. YOLO v8 Nano is great at quickly and accurately spotting fractures, even on devices with lower computing power. After training, we test the model on a separate set of images to ensure it can reliably detect fractures. In practical use, the trained model automatically identifies and marks fractures on new X-ray images by drawing boxes around them. This helps doctors quickly and accurately diagnose fractures. We assess the model's effectiveness using performance metrics like confusion matrix and Intersection over Union (IoU) scores to understand how well it performs across different types of fractures.",
       height=200
    )
   
   st.markdown("##### FasterRCNN with ResNet")
   st.text_area(
        "Description",
        "The Faster R-CNN with ResNet backbone is a state-of-the-art object detection model that combines the Region Proposal Network (RPN) with a deep ResNet feature extractor. In our X-Ray Diagnosis project, we leverage this powerful architecture to detect and localize potential fractures in medical images. The ResNet backbone provides robust feature extraction capabilities, allowing the model to learn complex hierarchical representations of the input image. The RPN generates high-quality region proposals, which are then refined through subsequent convolutional layers. This approach enables precise fracture detection with improved accuracy compared to traditional methods.",
        height=200
    )
   
   st.markdown("##### VGG16")
   st.text_area(
        "Description", 
        "VGG16 is a deep convolutional neural network known for its simplicity and effectiveness in image classification tasks. In our X-Ray Diagnosis project, we utilize VGG16 to classify and analyze X-ray images. The network's architecture consists of multiple convolutional layers with small 3x3 filters, followed by max-pooling layers, which help in extracting meaningful features from medical images. By training VGG16 on a diverse dataset of X-ray images, we enable the model to learn intricate patterns and characteristics associated with different types of bone fractures.",
        height=200
    )
    
#weights 
yolo_path = os.path.join("weights", "yolov8.pt")

with tab2:
    st.markdown("### Upload & Test")
    st.markdown("#### Select Confidence Threshold")
    conf_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5)
    
    def set_clicked():
        st.session_state.clicked = True
    
    st.button('Upload Image', on_click=set_clicked)
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    if st.session_state.clicked:
        image = st.file_uploader("", type=["jpg", "png"])
        
        if image is not None:
            st.write("You selected the file:", image.name)
            
            if model == 'YoloV8':
                try:
                    yolo_detection_model = YOLO(yolo_path)
                    yolo_detection_model.load()
                except Exception as ex:
                    st.error(f"Unable to load model. Check the specified path: {yolo_path}")
                    st.error(ex)
                
                col1, col2 = st.columns(2)

                with col1:
                    uploaded_image = PIL.Image.open(image)
                        
                    st.image(
                        image=image,
                        caption="Uploaded Image",
                        use_container_width=True
                    )

                    if uploaded_image:
                        if st.button("Execution"):
                            with st.spinner("Running..."):
                                res = yolo_detection_model.predict(uploaded_image,
                                                    conf=conf_threshold, augment=True, max_det=1)
                                boxes = res[0].boxes
                                res_plotted = res[0].plot()[:, :, ::-1]
                                
                                if len(boxes)==1:
                                    names = yolo_detection_model.names
                                    probs = boxes.conf[0].item()
                                    
                                    for r in res:
                                        for c in r.boxes.cls:
                                            pred_class_label = names[int(c)]

                                    with col2:
                                        st.image(res_plotted,
                                                caption="Detected Image",
                                                use_container_width=True)
                                        try:
                                            with st.expander("Detection Results"):
                                                for box in boxes:
                                                    st.write(pred_class_label)
                                                    st.write(probs)
                                                    st.write(box.xywh)
                                        except Exception as ex:
                                            st.write("No image is uploaded yet!")
                                            st.write(ex)
                                
                                else:
                                    with col2:
                                        st.image(res_plotted,
                                                caption="Detected Image",
                                                use_container_width=True)
                                        try:
                                            with st.expander("Detection Results"):
                                                st.write("No Detection")
                                        except Exception as ex:
                                            st.write("No Detection")
                                            st.write(ex)
                                    
            elif model == 'FastRCNN with ResNet':
                resnet_model = rcnnres.get_model()
                device = torch.device('cpu')
                resnet_model.to(device)
                
                col1, col2 = st.columns(2)

                with col1:
                    uploaded_image = PIL.Image.open(image)
                        
                    st.image(
                        image=image,
                        caption="Uploaded Image",
                        use_container_width=True
                    )
                    
                    content = Image.open(image).convert("RGB")
                    to_tensor = torchvision.transforms.ToTensor()
                    content = to_tensor(content).unsqueeze(0)

                    if uploaded_image:
                        if st.button("Execution"):
                            with st.spinner("Running..."):
                                output = rcnnres.make_prediction(resnet_model, content, conf_threshold)
                                
                                print(output[0])

                                fig, _ax, class_name = rcnnres.plot_image_from_output(content[0].detach(), output[0])

                                with col2:
                                    st.image(rcnnres.figure_to_array(fig),
                                            caption="Detected Image",
                                            use_container_width=True)
                                    try:
                                        with st.expander("Detection Results"):
                                            st.write(class_name)
                                            st.write(output)
                                    except Exception as ex:
                                        st.write("No image is uploaded yet!")
                                        st.write(ex)

            elif model == 'VGG16':
                vgg_model = vgg.get_vgg_model()
                if vgg_model is not None:
                    device = torch.device('cpu')
                    vgg_model.to(device)
                    model = vgg_model
                else:
                    st.error("Failed to load VGG16 model. Please try another model.")
                    model = None
                
                col1, col2 = st.columns(2)

                with col1:
                    uploaded_image = PIL.Image.open(image)
                        
                    st.image(
                        image=image,
                        caption="Uploaded Image",
                        use_container_width=True
                    )
                    
                    content = Image.open(image).convert("RGB")
                    to_tensor = torchvision.transforms.ToTensor()
                    content = to_tensor(content).unsqueeze(0)

                    if uploaded_image:
                        if st.button("Execution"):
                            with st.spinner("Running..."):
                                output = vgg.make_prediction(model, content, conf_threshold)
                                
                                print(output[0])

                                with col2:
                                    fig, _ax, class_name, detection_info = vgg.plot_detection_results(content[0], output[0])
                                    if detection_info:
                                        st.image(vgg.figure_to_array(fig),
                                                caption="Detected Image",
                                                use_container_width=True)
                                        with st.expander("Detection Results"):
                                            st.write(f"Class: {detection_info['class']}")
                                            st.write(f"Confidence: {detection_info['confidence']}")
                                            st.write(f"Warning: {detection_info['warning']}")
                                    else:
                                        st.write("No detections found above confidence threshold.")
    else:
        st.write("Please upload an image to test")