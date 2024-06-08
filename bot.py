import telebot
from pyowm import OWM

API_TOKEN = '7044590208:AAF4IloeQ5CCbVGQvz--vinaUhbKEUgxIEQ'

owm = OWM('4f4dbaf5fcf613e9375a46764e2c4025')
mgr = owm.weather_manager()

bot = telebot.TeleBot(API_TOKEN)


@bot.message_handler(content_types=['text'])
def send_welcome(message):
    # bot.reply_to(message, message.text)
    observation = mgr.weather_at_place(message.text)
    w = observation.weather.temperature
    h = observation.weather.temperature('celsius')
    bot.send_message(message.chat.id, h)


bot.polling(none_stop=True)

# # Перевірка інформації про бота
# user = bot.get_me()
# if user is None:
#     raise Exception("Failed to get bot information")
#
# print(f"Bot username: @{user.username}")
#
# bot.polling()
