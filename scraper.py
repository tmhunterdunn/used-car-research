from random import randint
from time import sleep
import pandas as pd
import re
import datetime
import json
import os
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile


geckodriver_path = 'geckodriver/geckodriver'

options = Options()
#options.headless = False

profile = FirefoxProfile('/home/travis/snap/firefox/common/.mozilla/firefox/pq2s2mg1.selenium')
service = Service(geckodriver_path)

driver = webdriver.Firefox(service=service, options=options)


with open('config/province_codes.json', 'r') as f:
    province_codes = json.load(f)

url = "https://www.kijijiautos.ca/cars/"
car_type = "toyota/corolla/"
url += car_type

if not os.path.exists("data/" + car_type):
    os.makedirs("data/" + car_type)


driver.get(url)
