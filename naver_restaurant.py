from selenium.webdriver.common.by import By
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
from openpyxl import Workbook
from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import time
import datetime
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options as ChromeOptions
from webdriver_manager.chrome import ChromeDriverManager

num = [1980713467,1214681781,1020814763,1634516053,1939769414,
       1011487904,1883610032,1160705436, 1050788601,1730426040,
       13444926, 1809575566, 35696852]

# Webdriver headless mode setting
options = webdriver.ChromeOptions()
#options.add_argument('headless')
options.add_argument('window-size=1920x1080')
options.add_argument("disable-gpu")

# BS4 setting for secondary access
session = requests.Session()
headers = {
    "User-Agent": "user value"}

retries = Retry(total=5,
                backoff_factor=0.1,
                status_forcelist=[500, 502, 503, 504])

session.mount('http://', HTTPAdapter(max_retries=retries))

# New xlsx file
now = datetime.datetime.now()
xlsx = Workbook()
list_sheet = xlsx.create_sheet('output')
list_sheet.append(['title', 'sub_title', 'nickname', 'content', 'date'])

# Start crawling/scraping!
for n in num:
    
    try:
        # url
        url = f'https://m.place.naver.com/restaurant/{n}/review/visitor?entry=ple'
        
        # 크롬 드라이버 최신 버전 설정
        service = ChromeService(executable_path=ChromeDriverManager().install())
                
        # chrome driver
        driver = webdriver.Chrome(service=service, options=options) # <- options로 변경
        res = driver.get(url)
        driver.implicitly_wait(30)

        # Pagedown
        for _ in range(20):  # 페이지 다운을 20번 반복
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.PAGE_DOWN)
            time.sleep(0.4)

        try:
            while True:
                driver.find_element(By.XPATH, '//*[@id="app-root"]/div/div/div/div[6]/div[2]/div[3]/div[2]/div/a').click()
                time.sleep(0.4)
        except Exception as e:
            print('finish')

        time.sleep(25)
        html = driver.page_source
        bs = BeautifulSoup(html, 'html.parser')
        reviews = bs.select('li.owAeM')

        for r in reviews:
            nickname = r.select_one('div.qgLL3')
            content = r.select_one('div.vg7Fp>a>span.zPfVt')
            date = r.select('div.D40bm>span.CKUdu>time')[0]
            

            # exception handling
            nickname = nickname.text if nickname else ''
            content = content.text if content else ''
            date = date.text if date else ''
            time.sleep(0.06)

            print(nickname, '/', content, '/', date)
            list_sheet.append([nickname, content, date])
            time.sleep(0.06)
        # Save the file
        file_name = 'naver_review_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.xlsx'
        xlsx.save(file_name)

    except Exception as e:
        print(e)
        # Save the file(temp)
        file_name = 'naver_review_' + now.strftime('%Y-%m-%d_%H-%M-%S') + '.xlsx'
        xlsx.save(file_name)