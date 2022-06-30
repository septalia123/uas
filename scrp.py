import requests
from bs4 import BeautifulSoup

web = requests.get('https://www.kompas.com/')
#print(web.content)

webfix = BeautifulSoup( web.content)
#print(webfix)

populer = webfix.find('div', {'class', 'most__wrap clearfix'})

a = populer.find_all('div', {'class', 'most__list clearfix'})

for each in a:
    no = each.find('span', {'class', 'most__count'})
    judul = each.find('a', {'class', 'most__tittle'})
    baca = each.find('div', {'class', 'most__read'})
    print (no)
    print (judul)
    print (baca)
    print ('='*20)