import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Rescaling, Flatten, Dense
import requests

# Список з назвами класів
class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

img_width, img_height = 180, 180
# Шлях до збереженої моделі
model_path = '/Users/innate/Desktop/flowersModel.keras'

# Завантаження моделі
model = tf.keras.models.load_model(model_path)

# Завантаження зображення
    flower_url = "https://salisburygreenhouse.com/wp-content/uploads//Salisbury-at-Enjoy-Floral-Studio_-1.png"
response = requests.get(flower_url, verify=False)
#
with open('/Users/innate/Desktop/flower.jpg', 'wb') as f:
    f.write(response.content)
    f.close()

img = tf.keras.utils.load_img(
    '/Users/innate/Desktop/flower.jpg', target_size=(img_height, img_width)
)
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)  # Створюємо пакет з одного зображення

# Прогнозування
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])

predicted_class_index = np.argmax(score)
predicted_class_name = class_names[predicted_class_index]
predicted_probability = 100 * np.max(score)
print(f"An image is {predicted_class_name} ({predicted_probability:.2f}% accuracy)")

# Показ зображення
img.show()