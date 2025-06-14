import requests
from bs4 import BeautifulSoup
import pandas as pd 
def fetch_investor_trend(stock_code: str, date: str):
    url = f"https://finance.naver.com/item/frgn.naver?code={stock_code}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, 'html.parser')

    table = soup.select_one("table.type2")
    rows = table.find_all('tr') if table else []

    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 7 and date[-2:] in row.text:
            try:
                기관 = cols[5].text.strip().replace(',', '')
                외국인 = cols[6].text.strip().replace(',', '')
                return {
                    '기관': int(기관) if 기관 not in ['-', ''] else 0,
                    '외국인': int(외국인) if 외국인 not in ['-', ''] else 0
                }
            except:
                continue
    return {'기관': 0, '외국인': 0}
def match_news_before_events(news_list, events_df):
    result = {}
    for i, row in events_df.iterrows():
        event_time = row["datetime"]
        matched = []
        for news in news_list:
            try:
                pub_time = pd.to_datetime(news['pub_date'])
                if pub_time.tzinfo is not None:
                    pub_time = pub_time.tz_convert(None)
                if pub_time <= event_time:
                    matched.append(news)
            except Exception as e:
                continue
        result[event_time] = matched[:3]
    return result