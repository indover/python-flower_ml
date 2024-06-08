from pyowm import OWM

place = input('Enter location: ')
owm = OWM('4f4dbaf5fcf613e9375a46764e2c4025')
mgr = owm.weather_manager()

# print(mgr.weather_at_place(place))

observation = mgr.weather_at_place(place)
w = observation.weather.temperature
h = observation.weather.temperature('celsius')

print(observation.weather.temperature('celsius')["temp"])


# from pyowm.owm import OWM
# from pyowm.utils.config import get_default_config
# config_dict = get_default_config()
# config_dict['language'] = 'Ua'  # your language here, eg. French
# owm = OWM('4f4dbaf5fcf613e9375a46764e2c4025', config_dict)
# mgr = owm.weather_manager()
# observation = mgr.weather_at_place('Ukraine, UA')
# print(observation.weather.detailed_status)  # Nuageux