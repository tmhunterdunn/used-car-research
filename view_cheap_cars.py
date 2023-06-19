from random import randint
from time import sleep
from selenium import webdriver
import pandas as pd
import re

df = pd.read_csv('cheap-cars-kms.csv')
driver = webdriver.Firefox('geckodriver')

sleep(2)
url = "https://www.kijijiautos.ca/cars/toyota/corolla/automatic/#vip="

cheap_cars = df[df.cheap]

for i, vip in enumerate(cheap_cars.vip[15:]):
    driver.get(url + str(vip))
    print(i)
    driver.switch_to.new_window('tab')
    sleep(1)
