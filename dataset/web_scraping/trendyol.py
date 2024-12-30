from selenium import webdriver
from bs4 import BeautifulSoup
import time
import numpy as np
import csv
from threading import Thread, Lock
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import math

# Create a lock for thread-safe CSV writing
csv_lock = Lock()

def create_driver():
    return webdriver.Chrome()

def get_content(driver, url):
    driver.get(url)
    html = driver.page_source
    return BeautifulSoup(html, 'html.parser')

def get_page_no(start, end, url):
    return [f"{url}?&pi={i}" for i in range(start, end+1)]

def get_products(driver, page_nos):
    products = []
    for no in page_nos:
        html = get_content(driver, no)
        for product in html.find_all("div", {"class": "p-card-wrppr"}):
            products.append("https://www.trendyol.com" + product.find("div", {"class": "p-card-chldrn-cntnr"}).a["href"])
    return products

def get_comments(driver, url):
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
    return BeautifulSoup(html, "html.parser")

def process_product_chunk(products_chunk):
    
    driver = create_driver()
    
    try:
        for pr in products_chunk:
            try:
                html = get_content(driver, pr)
                
                # Extract product details
                try:
                    pr_brand = html.find("h1", {"class":"pr-new-br"}).a.text
                except:
                    pr_brand = html.find("span", {"class":"product-brand-name-without-link"})
                    pr_brand = pr_brand.text if pr_brand else np.nan
                
                try:
                    h1_tag = html.find("h1", {"class": "pr-new-br"})
                    pr_name = h1_tag.find("span").text
                    if pr_name == pr_brand:
                        spans = h1_tag.find_all("span")
                        pr_name = spans[1].text
                except:
                    pr_name = np.nan

                try:
                    categories = html.find_all("a", {"class":"product-detail-breadcrumb-item"})[2]
                    category = categories.span.text
                except:
                    category = np.nan

                try:
                    price = html.find("span", {"class": "prc-dsc"}).text
                except:
                    price = np.nan

                try:
                    rating_div = html.find("div", {"class": "product-rating-score"})
                    star = rating_div.find("div", {"class": "value"}).text.strip() if rating_div else np.nan
                except:
                    star = np.nan

                # Extract product description
                try:
                    product_desc = extract_product_description(html)
                except:
                    product_desc = np.nan

                # Get comments
                comments_arr = []
                try:
                    pr_href0 = pr.split('?')[0] if '?' in pr else pr
                    pr_href1 = pr.split('?')[1] if '?' in pr else ''
                    com_link = f"{pr_href0}/yorumlar?{pr_href1}"
                    
                    if com_link:
                        comments = get_comments(driver, com_link)
                        for com in comments.find_all("div", {"class":"comment"}):
                            c = com.find("div", {"class": "comment-text"}).p.text
                            comments_arr.append(c.strip())
                except Exception as e:
                    print(f"Error getting comments: {e}")

                # Thread-safe writing to CSV
                with csv_lock:
                    with open("dnm.csv", "a", encoding="utf-8", newline='') as file:
                        csv_writer = csv.writer(file)
                        row = [pr_brand, pr_name, category, price, star, product_desc, comments_arr]
                        csv_writer.writerow(row)

            except Exception as e:
                print(f"Error processing product {pr}: {e}")
                continue
    
    finally:
        driver.quit()

def extract_product_description(html):
    pr_details = {}
    info = html.find_all("ul", {"class":"detail-desc-list"})
    desc = html.find_all("div", {"class":"product-desc-content"})
    detail_items = html.find_all("ul", {"class": "detail-attr-container"})
    
    if len(info) >= 1:
        list_items = info[0].find_all('li')
        if len(list_items) > 6:
            return " ".join([part.text.strip() for part in list_items[7:]])
    
    if len(desc) != 0:
        return " ".join([part.text.strip() for part in desc if part.text.strip()])
    
    if len(detail_items) != 0:
        for ul in detail_items:
            for item in ul.find_all("li"):
                if item:
                    key_span = item.find("span", {"class": "attr-key-name-w"})
                    value_span = item.find("span", {"title": True})
                    
                    if key_span and value_span:
                        key = key_span.text.strip()
                        value_div = value_span.find("div", {"class": "attr-value-name-w"})
                        value = value_div.text.strip() if value_div else value_span.get("title", "").strip()
                        
                        if key and value:
                            pr_details[key] = value
        
        return pr_details if pr_details else np.nan
    
    return np.nan

def main():
    URL = 'https://www.trendyol.com/erkek-ayakkabi-x-g2-c114'
    
    main_driver = create_driver()
    
    try:
        page_nos = get_page_no(101, 200, URL)
        products = get_products(main_driver, page_nos)
    finally:
        main_driver.quit()
    
    with open("dnm.csv", "w", encoding="utf-8", newline='') as file:
        csv_writer = csv.writer(file)
        headers = ["brand", "product_name", "category", "price", "rating", "info", "comments"]
        csv_writer.writerow(headers)
    
    num_threads = 4  
    chunk_size = math.ceil(len(products) / num_threads)
    product_chunks = [products[i:i + chunk_size] for i in range(0, len(products), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(process_product_chunk, product_chunks)

if __name__ == "__main__":
    main()