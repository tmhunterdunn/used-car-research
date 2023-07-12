from random import randint
from time import sleep
from selenium import webdriver
import pandas as pd
import re
import datetime
import json
import os

with open('config/province_codes.json', 'r') as f:
    province_codes = json.load(f)

url = "https://www.kijijiautos.ca/cars/"
car_type = "hyundai/kona/"
url += car_type

if not os.path.exists("data/" + car_type):
    os.makedirs("data/" + car_type)


driver = webdriver.Firefox('geckodriver')
driver.get(url)
sleep(2)

car_count_text = driver.find_elements('xpath', '//h2[@data-testid="searchResultHeadline"]')[0].text
res = re.search('(\d+(,\d+)*) cars for sale', car_count_text)
car_count = int(res.group(1).replace(',',''))



location_button = driver.find_elements('xpath', '//button[@data-testid="LocationLabelLink"]')[1]
location_button.click()
sleep(2)

location_input = driver.find_elements('xpath', '//input[@id="LocationAutosuggest"]')[0]
location_input.clear()
sleep(2)

location_submit_button = driver.find_elements('xpath', '//button[@data-testid="LocationModalSubmitButton"]')[0]
location_submit_button.click()
sleep(2)

items = []
rows = [] 
vips = set()

for i in range(100):

    if len(vips) >= car_count:
        break



    new_items = driver.find_elements('xpath', '//div[@data-testid="VehicleListItem"]')

    for item in new_items:

        car_id = item.get_attribute('data-test-ad-id')

        text_list = item.text.split('\n')

        if text_list[0]=='Details':
            text_list.pop(0)
        if text_list[0]=='Promoted':
            text_list.pop(0)

        new_row = {}
        new_row['vip'] = car_id
        vips.add(car_id)
        new_row['title'] = text_list[0]

        for text in text_list:

            # Year
            res = re.search('(\d\d\d\d)', new_row['title'])
            if res:
                new_row['year'] = int(res.group(1))

            # Kms
            res = re.search('(\d+(,\d+)*) km', text)
            if res:
                new_row['kms'] = float(res.group(1).replace(',',''))
            
            # Price
            res = re.search('\$(\d+(,\d+)*)', text)
            if res:
                new_row['price'] = float(res.group(1).replace(',',''))


            # Location
            res = re.search('(.+), (' + '|'.join(province_codes) +')', text)
            if res:
                new_row['city'] = res.group(1)
                new_row['province'] = res.group(2)


            # Transmission
            res = re.search('(Manual|Automatic)', text)
            if res:
                new_row['transmission'] = res.group(1)

            res = re.search('(CVT)', new_row['title'])
            if res:
                new_row['transmission'] = res.group(1)


        res = re.search('hybrid', new_row['title'], flags=re.IGNORECASE)

        if not res:
            rows.append(new_row)

    items = items + new_items
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    sleep(0.5)

df = pd.DataFrame(rows)
df.set_index('vip', inplace=True)
df.drop_duplicates(inplace=True)
df.kms.fillna(0, inplace=True)
df.dropna(0, subset=['price'], inplace=True)
df.to_csv('data/{}/{}.csv'.format(car_type, datetime.datetime.now()).replace(':', '.'))