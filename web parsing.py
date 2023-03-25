#use selenium to load linkedin.com login page

user = "randiveshubham3@gmail.com"
pas = "Knk2mhyh"

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

#load firefox 
driver = webdriver.Firefox()
driver.get("https://www.linkedin.com/uas/login")

#find username and password fields
username = driver.find_element(By.ID, "username")
password = driver.find_element(By.ID, "password")

#enter username and password
username.send_keys(user)
password.send_keys(pas)

#find element for login button
login = driver.find_element(By.XPATH, "//button[@type='submit']")
login.click()

import time
time.sleep(5)

#find element that contains text "Who's viewed your profile"
who_viewed = driver.find_element(By.XPATH, "//a[@href='/me/profile-views/']")
who_viewed.click()


#load content of page into variable
content = driver.page_source

print(content)

#close browser
driver.close()

