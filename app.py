from pyexpat import model
from PIL import Image, ImageDraw, ImageFont
import streamlit as st
import numpy as np
import time
import math
import cv2
import os
import base64

st.set_page_config(page_title='Objek Deteksi YOLOv4',
page_icon="chart_with_upwards_trend")
@st.cache(allow_output_mutation=True)
def load_model(cfg_path, weights_path, labels_path):
    # lmemuat label kelas dan pelatihan dataset
    labels = open(labels_path).read().strip().split("\n")

    # menginisialisasi daftar warna untuk mewakili setiap label kelas yang mungkin
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    # memuat pre-trained model
    net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i-1] for i in net.getUnconnectedOutLayers()]

    return [net, layer_names, labels, colors]

def process_image(img, model_info, score_threshold=0.3, overlap_threshold=0.3):
    net, layer_names, labels, colors = model_info
    image = img
    H, W = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(layer_names)

    boxes = []
    scores = []
    classIDs = []
    for output in layerOutputs:
        # Untuk setiap objek yang terdeteksi, hitung kotak pembatas, temukan skornya, abaikan jika di bawah ambang batas
        for detection in output:
            confidences = detection[5:]
            classID = np.argmax(confidences)
            score = confidences[classID]
            if score >= score_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                centerX, centerY, width, height = box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                scores.append(float(score))
                classIDs.append(classID)

    # menerapkan penekanan non-maksimum untuk menekan kotak pembatas yang lemah dan tumpang tindih
    idxs = cv2.dnn.NMSBoxes(boxes, scores, score_threshold, overlap_threshold)

    # custom font for showing label on objects
    font = ImageFont.truetype(font='fonts/FiraMono-Medium.otf', size=np.floor(0.01 * (W + H) + 0.5).astype('int32'))
    # hitung ketebalan kotak pembatas yang sesuai
    thickness = (H + W) // (420 + len(idxs) // 10)
    
    img_pil = Image.fromarray(image)

    if len(idxs) > 0:
        for i in idxs.flatten():
            # ekstrak koordinat kotak pembatas, dapatkan label
            left, top, right, bottom = boxes[i]
            top = max(0, int(math.floor(top + 0.5)))
            left = max(0, int(math.floor(left + 0.5)))
            right = min(W, int(math.floor(right + left + 0.5)))
            bottom = min(H, int(math.floor(bottom + top + 0.5)))

            draw = ImageDraw.Draw(img_pil)
            color = tuple([int(c) for c in colors[classIDs[i]]])
            label = "{}: {:.3f}".format(labels[classIDs[i]], scores[i])
            labelSize = draw.textsize(label, font)

            origin = np.array([left, top + 1])
            if top - labelSize[1] >= 0:
                origin[1] = top - labelSize[1]
                
            # menggambar persegi panjang kotak pembatas dan beri label pada gambar
            for j in range(thickness):
                draw.rectangle([left + j, top + j, right - j, bottom - j], outline=color)
            draw.rectangle([tuple(origin), tuple(origin + labelSize)], fill=color)
            draw.text(origin, label, fill=(0, 0, 0), font=font)
            del draw

    image = np.array(img_pil)[:,:, [2, 1, 0]]
    classIDs = np.array(classIDs)[idxs].tolist()
    scores = np.array(scores)[idxs].tolist()

    return [image, classIDs, scores]


st.set_option('deprecation.showfileUploaderEncoding', False)

st.sidebar.title("About")
st.sidebar.text("aplikasi ini untuk mendeteksi jenis")
st.sidebar.text("ayam hias yang masuk kategori")
st.sidebar.text("ayam kate, ayam pelung, dan ayam poland")
# tambahkan judul dan sidebar
st.title("Objek Deteksi Jenis Ayam Hias YOLOv4 via Streamlit")

model_type = st.sidebar.markdown("  ")

#PATH ke file config, file bobot, nama kelas
if model_type == "Objek Deteksi":
    start = time.time()
    model_info = load_model("configs/yolov4-tiny-custom.cfg", "weights/yolov4-tiny-custom_best.weights", "obj.names")
    end = time.time()
    model_info[-1] = np.array([[0, 235, 43], [0, 0, 255]], dtype="uint8")
else:
    start = time.time()
    model_info = load_model("configs/yolov4-tiny-custom.cfg", "weights/yolov4-tiny-custom_best.weights", "obj.names")
    end = time.time()
# styling
st.markdown(
f"""
<style>
    .my-progress-bar {{
        width: 60%;
        height: 10px;
        background-color: #cddefc;
    }}
    .my-progress-bar-fill {{
        display: block;
        height: 10px;
        background-color: #4c7adb;
        transition: width 500ms ease-in-out;
    }}
    .reportview-container .main .block-container{{
        width: {80}%;
        max-width: {900}px;
        margin: 0 auto;
    }}
</style>
""",
    unsafe_allow_html=True,
)

st.subheader("Upload file gambar untuk melakukan objek deteksi")
img_stream = st.file_uploader("", ['jpg', 'jpeg', 'png'])

if img_stream:
    # read image from stream and swap RGB to BGR
    img = cv2.imdecode(np.frombuffer(img_stream.read(), np.uint8), 1)
    st.subheader("INPUT:")
    st.image(img[:,:, [2, 1, 0]], use_column_width=True)
    start = time.time()
    image, classIDs, scores = process_image(img, model_info)
    end = time.time()
    # st.balloons()
    st.subheader("Hasil:")
    st.image(image, use_column_width=True)
    st.success("Waktu Deteksi: {:.2f} detik".format(end - start))
    st.markdown("**Prediksi:**")
    if model_type == "Objek Deteksi":
        st.write("Deteksi ", classIDs.count(0), " Objek ", classIDs.count(1), " Ayam Hias")
    else:
        st.write("Deteksi ", len(classIDs), " Objek Ayam Hias")
    labels, colors = model_info[2:]
    for i, j in enumerate(classIDs):
        color = tuple([int(c) for c in colors[j]])[1::1]
        st.markdown("""
        <div style='display: flex; justify-content: space-between; width: 50%; align-items:center;'>
            <div style='text-align: left; flex:1; color:rgb{}; font-weight: 500; text-shadow: 1px 0px;'>{}:</div>
            <div style='display:flex; justify-content: space-between; align-items:center; flex:1'>
                <div class="my-progress-bar" style='text-align: left'>
                    <span class="my-progress-bar-fill" style="width: {}%;"></span>
                </div>
                <div style='text-align: right'>{:.2f}</div>
            </div>
        </div>
        """.format(color, labels[j], scores[i] * 100, scores[i])
        , unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""

    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        height: 100%;
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('ayam3.jpg')
