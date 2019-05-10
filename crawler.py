#!/usr/bin/python
# -*- coding: utf-8 -*-
import urllib.request
import urllib.error
from bs4 import BeautifulSoup
import ssl
import json

def get_review():
    # For ignoring SSL certificate errors
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url=input("Enter Amazon Product Url- ")
    html = urllib.request.urlopen(url, context=ctx).read()
    soup = BeautifulSoup(html, 'html.parser')
    html = soup.prettify('utf-8')
    product_json = {}
    # This block of code will help extract the Brand of the item

    # This block of code will help extract the average star rating of the product
    for i_tags in soup.findAll('i',
                               attrs={'data-hook': 'average-star-rating'}):
        for spans in i_tags.findAll('span', attrs={'class': 'a-icon-alt'}):
            product_json['star-rating'] = spans.text.strip()
            break

    # This block of code will help extract the long reviews of the product
    product_json['long-reviews'] = []
    for divs in soup.findAll('div', attrs={'data-hook': 'review-collapsed'
                             }):
        long_review = divs.text.strip()
        product_json['long-reviews'].append(long_review)

    # Saving the scraped data in json format
    with open('product.json', 'w') as outfile:
        json.dump(product_json, outfile, indent=4)
    print('----------Extraction of data is complete. Check json file.----------')
    return float(product_json['star-rating'][:4]), product_json['long-reviews']