import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By

BASE_DIR = os.path.dirname(os.path.realpath(__file__))


def initialize():
    driver_path = BASE_DIR + "/driver/chromedriver"
    driver = webdriver.Chrome(executable_path=driver_path)
    driver.get("https://m.dailyhunt.in/news/india/hindi/travel?mode=pwa&action=click")
    time.sleep(3)
    heading = driver.find_elements(By.XPATH, '//*[@id="app"]/div/div/div/section[1]/div/div/div/div[1]/section/section/figcaption/h2')
    head_text = heading[0].text
    button = driver.find_element(By.XPATH, '//*[@id="app"]/div/div/div/section[1]/div/div/div/div[1]/section/section/div/section/a/div')
    driver.execute_script("arguments[0].click();", button)
    time.sleep(2)
    return head_text
