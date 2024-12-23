from selenium import webdriver
from bs4 import BeautifulSoup
import time
import numpy as np
import csv

driver = webdriver.Chrome()

URL = 'https://www.trendyol.com/erkek-giyim-x-g2-c82'

def get_content(url):
    driver.get(url)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    return soup

def get_page_no(start, end, url):
    page_nums = []

    for i in range(start, end+1):
        page_nums.append(url + "?&pi=" + str(i))

    return page_nums

def get_products(page_nos):
    products = []
    a = 0

    for no in page_nos:
        html = get_content(no)

        for product in html.find_all("div",{"class":"p-card-wrppr"}):        
            products.append("https://www.trendyol.com" + product.find("div",{"class":"p-card-chldrn-cntnr"}).a["href"])
            a += 1

    return products, a

def get_comments(url):
    driver.get(url)
    SCROLL_PAUSE_TIME = 0.5

    last_height = driver.execute_script("return document.body.scrollHeight")

    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        time.sleep(SCROLL_PAUSE_TIME)

        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    
    return soup 

def get_pr_features(products):

    features = [
            "brand", "product_name", "category", 
            "price", "rating", "info", "comments"
        ] 

    with open("dataset2.csv", "a", encoding="utf-8") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(features)

        for pr in products:
            html = get_content(pr)

            try:
                pr_brand = html.find("h1", {"class":"pr-new-br"}).a.text

            except:
                pr_brand = html.find("span", {"class":"product-brand-name-without-link"})
                if pr_brand:
                    pr_brand = pr_brand.text
                else:
                    pr_brand = np.nan
                
            try:
                h1_tag = html.find("h1", {"class": "pr-new-br"})
                pr_name = h1_tag.find("span").text

                if pr_name == pr_brand:
                    spans = h1_tag.find_all("span")
                    pr_name = spans[1].text
            except:
                pr_name = np.nan


            try:
                categories =  html.find_all("a" , {"class":"product-detail-breadcrumb-item"})[2]
                category = categories.span.text
            except: 
                category = np.nan  

            try:
                price = html.find("span", {"class": "prc-dsc"}).text
            except:
                price = np.nan

            try:
                rating_div = html.find("div", {"class": "product-rating-score"})

                if rating_div:
                    star = rating_div.find("div", {"class": "value"}).text.strip()
                else:
                    star = np.nan  
            except AttributeError:
                star = np.nan

            try:
                pr_details = {}
                product_desc = ""
                info = html.find_all("ul", {"class":"detail-desc-list"})
                desc = html.find_all("div", {"class":"product-desc-content"})
                detail_items = html.find_all("ul", {"class": "detail-attr-container"})
                
                list_items = info[0].find_all('li')
                if len(info) >= 1 and len(list_items) > 6: 
                    product_desc = " ".join([part.text.strip() for part in list_items[7:]])

                elif len(desc) != 0:
                    product_desc = " ".join([part.text.strip() for part in desc if part.text.strip()])

                elif len(detail_items) != 0:
                    for item in detail_items[0].find_all("li"):
                        key = item.find("span", {"class":"attr-key-name-w"}).text.strip()
                        value = item.find("span", {"class":"attr-value-name-w"}).text.strip()
            
                        pr_details[key] = value
        
                    product_desc = pr_details

                elif len(detail_items) == 0:
                    product_desc = np.nan

            except:
                product_desc = np.nan
            comments_arr = []
            try:
                if '?' in pr:
                    pr_href0 = pr.split('?')[0]  
                    pr_href1 = pr.split('?')[1]  
                else:
                    pr_href0 = pr  
                    pr_href1 = ''

                com_link = f"{pr_href0}/yorumlar?{pr_href1}"
                
                if com_link:
                    comments = get_comments(com_link)
                else: 
                    print("Comments link not found.")
                
                for com in comments.find_all("div", {"class":"comment"}):
                    c = com.find("div", {"class": "comment-text"}).p.text
                    comments_arr.append(c.strip())

            except Exception as e:
                print(f"Error occurred: {e}") 

            row = [
                pr_brand, pr_name, category,
                price, star, product_desc, comments_arr
            ]

            csv_writer.writerow(row)
            


if __name__=="__main__":

    nos = get_page_no(132, 200, URL) 
    products , num = get_products(nos)
    get_pr_features(products)
    
    driver.quit()
    