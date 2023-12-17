import streamlit as st
from PIL import Image, ImageDraw
from io import BytesIO
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas


def numpy_to_pil_image(numpy_image):
    return Image.fromarray(numpy_image)


def watetSketch(inp_img):
    inp_img = np.array(inp_img, dtype="uint8")
    img_hsv = cv2.cvtColor(inp_img, cv2.COLOR_BGR2HSV)
    adjust_v = (img_hsv[:, :, 2].astype("uint") + 5) * 3
    adjust_v = ((adjust_v > 255) * 255 + (adjust_v <= 255) * adjust_v).astype("uint8")
    img_hsv[:, :, 2] = adjust_v
    img_soft = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
    img_soft = cv2.GaussianBlur(img_soft, (51, 51), 0)

    img_gray = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.equalizeHist(img_gray)
    invert = cv2.bitwise_not(img_gray)
    blur = cv2.GaussianBlur(invert, (21, 21), 0)
    invertblur = cv2.bitwise_not(blur)
    sketch = cv2.divide(img_gray, invertblur, scale=265.0)
    sketch = cv2.merge([sketch, sketch, sketch])

    img_water = ((sketch / 255.0) * img_soft).astype("uint8")

    st.subheader("Drawing Water")
    result_water = st.columns(1)
    with result_water[0]:
        st.image(img_water, width=500, caption="Final Result")

    return img_water

def pencilsketch(inp_img):
    gray_image = cv2.cvtColor(inp_img, cv2.COLOR_BGR2GRAY)

    inverted_image = 255 - gray_image

    inverted_blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)

    final = cv2.divide(gray_image, 255 - inverted_blurred, scale=240)

    st.subheader("Drawing Pencil")
    final_result = st.columns(1)
    with final_result[0]:
        st.image(final, width=500, caption="Final Result")

    list_img = [inp_img, gray_image, inverted_image, inverted_blurred, final]
    steps = 5

    columns = st.columns(steps)
    for i in range(steps):
        with columns[i]:
            st.image(list_img[i], width=125, caption=f"Step {i+1}")

    return final


def apply_noise_reduction(image):
    return cv2.medianBlur(image, 9)


def load_an_image(image_path):
    img = Image.open(image_path)
    return img


def cartoonizePhoto(img):
    image = np.asarray(img)
    K = 9
    scale_ratio = 0.95

    # Tính toán kích thước mới
    new_width = int(image.shape[1] * scale_ratio)
    new_height = int(image.shape[0] * scale_ratio)
    new_dimensions = (new_width, new_height)
    image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)

    pixels = image.reshape((-1, 3))
    pixels = np.float32(pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    # Lấy màu của từng pixel dựa trên nhãn
    res2 = centers[labels.flatten()]
    # Reshape ảnh đã phân loại
    res2 = res2.reshape(image.shape)

    contoured_image = np.copy(res2)
    gray = cv2.cvtColor(contoured_image, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(gray, 100, 200, L2gradient=True)
    contours, hierarchy = cv2.findContours(edged,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(contoured_image, contours, contourIdx=-1, color=1, thickness=1)

    st.subheader("Cartoonize")
    result = st.columns(1)
    with result[0]:
        st.image(contoured_image, width=500, caption="Final Result")

    list_img = [image, res2, gray, edged, contoured_image]
    steps = 5
    columns = st.columns(steps)
    for i in range(steps):
        with columns[i]:
            st.image(list_img[i], width=125, caption=f"Step {i + 1}")

    return contoured_image


def oilPaint(img):
    res = cv2.xphoto.oilPainting(img, 7, 1)
    st.subheader("Oil Paint")
    result = st.columns(1)
    with result[0]:
        st.image(res, width=500, caption="Final Result")
    return res


def main():

    st.title('Drawing From Photos')
    st.write("This is an application developed for applying drawing.")
    st.subheader("Choose an option")

    option = st.selectbox('Choose Options',
                          (
                           'Drawing Pencil',
                           'Drawing Water Color',
                           'Cartoon',
                           'Oil Paint',
                           'Apply Noise',
                           'Free Drawing'))

    if option != 'Free Drawing':
        st.subheader("Upload image")
        image_file = st.file_uploader("Upload Images", type=["jpg", "jpeg"])

        if image_file is not None:

            input_image = load_an_image(image_file)
            st.subheader("Original Image")

            st.image(load_an_image(image_file), width=500)

            if option == 'Drawing Water Color':
                result_image = watetSketch(input_image)
                buf = BytesIO()
                final_pil_image = numpy_to_pil_image(result_image)
                final_pil_image.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="watercolor.png",
                    mime="image/png"
                )

            if option == 'Drawing Pencil':
                image = Image.open(image_file)
                final_sketch = pencilsketch(np.array(image))
                im_pil = numpy_to_pil_image(final_sketch)

                buf = BytesIO()
                im_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="pencil.png",
                    mime="image/png"
                )

            if option == 'Cartoon':
                image = Image.open(image_file)
                final_sketch = cartoonizePhoto(image)
                im_pil = numpy_to_pil_image(final_sketch)

                buf = BytesIO()
                im_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="cartoonize.png",
                    mime="image/png"
                )

            if option == 'Oil Paint':
                image = Image.open(image_file)
                final_sketch = oilPaint(np.array(image))
                im_pil = numpy_to_pil_image(final_sketch)

                buf = BytesIO()
                im_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="oilPaint.png",
                    mime="image/png"
                )

            if option == 'Apply Noise':
                image = Image.open(image_file)
                img_array = np.array(image)
                img_array = apply_noise_reduction(img_array)
                im_pil = Image.fromarray(img_array)
                st.subheader("Image with Applied Noise")
                st.image(im_pil, width=500)

                buf = BytesIO()
                im_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                st.download_button(
                    label="Download Image",
                    data=byte_im,
                    file_name="noise.png",
                    mime="image/png"
                )

    if option == 'Free Drawing':
        st.subheader("Free Drawing")

        color = st.color_picker("Choose color", "#000000")

        drawing_mode = st.radio(
            "Choose drawing tool",
            ('Free Draw', 'Line', 'Rectangle'))

        if drawing_mode == 'Free Draw':
            canvas_result = st_canvas(
                fill_color=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                stroke_width=5,
                stroke_color=color,
                background_color="#eee",
                height=500,
                drawing_mode="freedraw",
                key="canvas", )

        elif drawing_mode == 'Line':
            canvas_result = st_canvas(
                fill_color=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                stroke_width=5,
                stroke_color=color,
                background_color="#eee",
                height=500,
                drawing_mode="line",
                key="canvas", )
        elif drawing_mode == 'Rectangle':
            canvas_result = st_canvas(
                fill_color=f"rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.3)",
                stroke_width=5,
                stroke_color=color,
                background_color="#eee",
                height=500,
                drawing_mode="rect",
                key="canvas", )


if __name__ == '__main__':
    main()
