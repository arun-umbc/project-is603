import csv
import os
import time

from selenium import webdriver
from http.client import BadStatusLine

from selenium.common.exceptions import StaleElementReferenceException
from selenium.webdriver.common.by import By

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
url_mapper = {
    'sport': "https://m.dailyhunt.in/news/india/hindi/sports-topics-bd090a7ae8c65533f30004b78668ffe4/sports-subtopics-17?topicTitle=खेल%20&mode=pwa",
    'crime': "https://m.dailyhunt.in/news/india/hindi/crime-topics-2330c83b209099931e02432866cb3d1b/crime-subtopics-502?topicTitle=अपराध&mode=pwa",
    'politics': "https://m.dailyhunt.in/news/india/hindi/politics?mode=pwa&action=click",
    'automobile': "https://m.dailyhunt.in/news/india/hindi/automobile-topics-80a9814b26dfeaed8bdcb25e13753414/automobile-subtopics-101?topicTitle=मोटर&mode=pwa",
    'health': "https://m.dailyhunt.in/news/india/hindi/health+tips",
    'entertainment': "https://m.dailyhunt.in/news/india/hindi/entertainment"
}


class ArticleDetailsScraper(object):
    parent_div_xpath = '//*[@id="app"]/div/div/div/section[1]/div/div/div/div'
    scroll_limit = 500000

    def __init__(self, category):
        self.category = category
        self.url = url_mapper[category]
        self.title = None
        self.content = None
        self._initialize_driver()

    def _initialize_driver(self):
        driver_path = BASE_DIR + '/driver/chromedriver'
        option = webdriver.ChromeOptions()
        option.add_argument('headless')
        self.driver = webdriver.Chrome(executable_path=driver_path, options=option)
        self.driver.get(self.url)

    def scroll_to_page_end(self):
        scroll_pause_time = 5
        last_height = self.driver.execute_script("return document.body.scrollHeight")

        while True:
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(scroll_pause_time)
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height or new_height > self.scroll_limit:
                break
            last_height = new_height
            print('scroll')

    def click_button(self, index):
        try:
            button = self.driver.find_element(By.XPATH, f'//*[@id="app"]/div/div/div/section[1]/div/div/div/div[{index}]/section/section/div/section/a/div')
            self.driver.execute_script("arguments[0].scrollIntoView();", button)
            self.driver.execute_script("arguments[0].click();", button)
            time.sleep(4)
            print('click')
        except Exception as e:
            print(str(e))

    def write_to_csv(self, data):
        keys = data[0].keys()
        with open(f'data/{self.category}.csv', 'w', newline='') as output_file:
            dict_writer = csv.DictWriter(output_file, keys)
            dict_writer.writeheader()
            dict_writer.writerows(data)

    def get_article_data(self):
        try:
            self.scroll_to_page_end()
            count_of_divs = len(self.driver.find_elements(By.XPATH, self.parent_div_xpath))
            print(f"Total documents: {count_of_divs}")
            data = list()
            try:
                for i in range(0, count_of_divs):
                    index = i+1
                    title_xpath = f'//*[@id="app"]/div/div/div/section[1]/div/div/div/div[{index}]/section/section/figcaption/h2'
                    content_xpath_1 = f'//*[@id="app"]/div/div/div/section[1]/div/div/div/div[{index}]/section/section/div[1]/div/p'
                    content_xpath_2 = f'//*[@id="app"]/div/div/div/section[1]/div/div/div/div[{index}]/section/section/div[2]/div[1]/p'
                    try:
                        self.click_button(index)
                        title_element = self.driver.find_element(By.XPATH, title_xpath)
                        content_element_1 = self.driver.find_elements(By.XPATH, content_xpath_1)
                        content_element_2 = self.driver.find_elements(By.XPATH, content_xpath_2)
                        content_elements = content_element_1 + content_element_2

                        self.title = self.get_true_text(title_element).strip()
                        self.content = '\n'.join([self.get_true_text(elem) for elem in content_elements])
                        if self.content:
                            data.append({'title': self.title, 'content': self.content.replace('\n', " "), 'category': self.category})
                        else:
                            print("Skipped")
                    except (BadStatusLine, StaleElementReferenceException):
                        continue
            except Exception as e:
                print(str(e))
            self.write_to_csv(data)
        finally:
            self.driver.close()

    @staticmethod
    def get_true_text(tag):
        children = tag.find_elements_by_xpath('*')
        original_text = tag.text
        if not original_text:
            for child in children:
                original_text = original_text.replace(child.text, '', 1)
        return original_text


for k, v in url_mapper.items():
    a = ArticleDetailsScraper(category=k)
    a.get_article_data()
