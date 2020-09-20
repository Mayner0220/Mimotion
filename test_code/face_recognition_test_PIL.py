import FR
from PIL import Image, ImageDraw

test_img = FR.load_image_file("test.jpg")

test_face_locations = FR.face_locations(test_img)
test_face_encoding = FR.face_encodings(test_img, test_face_locations)

pil_img = Image.fromarray(test_img)
draw = ImageDraw.Draw(pil_img)

for (top, right, bottom, left), face_encoding in zip(test_face_locations, test_face_encoding):
    name = "Test"

    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

del draw

pil_img.show()