import requests
import json
from datetime import datetime, timedelta
import time
from bs4 import BeautifulSoup
from urllib.parse import quote
import pytz
import pandas as pd 

class NaverNewsSearcher:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.base_url = "https://openapi.naver.com/v1/search/news.json"

    def search_news(self, company_name, target_date, display=10, sort="date"):
        query = f"{company_name} {target_date}"
        encoded_query = quote(query)
        url = f"{self.base_url}?query={encoded_query}&display={display}&sort={sort}"
        headers = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API 요청 오류: {e}")
            return None

    def filter_news_by_date(self, news_data, target_date):
        """API 응답에서 특정 날짜의 뉴스만 필터링"""
        if not news_data or 'items' not in news_data:
            return []
        filtered_news = []
        for item in news_data['items']:
            pub_date = item.get('pubDate', '')
            try:
                date_obj = datetime.strptime(pub_date, "%a, %d %b %Y %H:%M:%S %z")
                news_date = date_obj.strftime("%Y-%m-%d")
                if news_date == target_date:
                    filtered_news.append(item)
            except ValueError:
                continue
        return filtered_news

    def format_news_data(self, news_items):
        """
        뉴스 데이터를 보기 좋게 포맷팅하며 pub_date를 datetime 형식으로 유지
        """
        formatted_news = []

        for item in news_items:
            try:
                # API 응답 형식에서 datetime 객체로 변환
                if isinstance(item.get('pubDate'), str):
                    pub_date = datetime.strptime(item.get('pubDate', ''), "%a, %d %b %Y %H:%M:%S %z")
                else:
                    # 이미 datetime 객체인 경우
                    pub_date = item.get('pubDate')
            except Exception:
                continue

            # HTML 태그 제거
            title = item.get('title', '').replace('<b>', '').replace('</b>', '')
            description = item.get('description', '').replace('<b>', '').replace('</b>', '')

            formatted_item = {
                'title': title,
                'description': description,
                'link': item.get('link', ''),
                'pub_date': pub_date,
                'original_link': item.get('originallink', '')
            }

            formatted_news.append(formatted_item)

        return formatted_news

def search_news_advanced(searcher, company_name, target_date):
    """고급 뉴스 검색 - 여러 쿼리 형식으로 검색"""
    search_queries = [
        f"{company_name} {target_date}",
        f"{company_name} {target_date.replace('-', '.')}",
        f"{company_name} {target_date.replace('-', '/')}"
    ]
    all_news = []
    
    for query in search_queries:
        print(f"검색 쿼리: {query}")
        news_data = searcher.search_news(query.split()[0], target_date, display=100)
        if news_data and 'items' in news_data:
            # API 응답에서 해당 날짜 뉴스만 필터링
            filtered_news = searcher.filter_news_by_date(news_data, target_date)
            all_news.extend(filtered_news)
    
    # 중복 제거 (링크 기준)
    unique_news = []
    seen_links = set()
    for news in all_news:
        link = news.get('link', '')
        if link not in seen_links:
            seen_links.add(link)
            unique_news.append(news)
    
    return unique_news

def format_news_data(news_items):
    """뉴스 아이템들을 포맷팅 (전역 함수)"""
    formatted = []
    for item in news_items:
        try:
            # pubDate가 문자열인 경우 datetime으로 변환
            if isinstance(item.get("pubDate"), str):
                pub_date = datetime.strptime(item["pubDate"], "%a, %d %b %Y %H:%M:%S %z")
            else:
                pub_date = item.get("pubDate")
        except Exception:
            pub_date = None

        formatted.append({
            "title": item.get("title", "").replace("<b>", "").replace("</b>", ""),
            "description": item.get("description", ""),
            "pub_date": pub_date,
            "link": item.get("link", ""),
            "original_link": item.get("originallink", "")
        })
    return formatted

def filter_news_by_date(news_items, target_date_str):
    """
    이미 포맷팅된 뉴스에서 특정 날짜만 필터링
    """
    result = []
    for item in news_items:
        try:
            pub_date = item.get("pub_date")
            if isinstance(pub_date, datetime):
                if pub_date.strftime("%Y-%m-%d") == target_date_str:
                    result.append(item)
        except Exception as e:
            continue
    return result

def match_news_before_events(news_list, event_df):
    """
    각 이벤트 시점보다 과거에 발생한 뉴스만 연결해 리턴 (타임존 일치 포함)

    Args:
        news_list (List[Dict]): 전체 뉴스 데이터 (title, pub_date 포함, datetime 객체)
        event_df (DataFrame): 이벤트 시점 목록 (datetime 컬럼 포함)

    Returns:
        Dict[datetime: List[뉴스]]
    """
    matched_results = {}
    korea_tz = pytz.timezone("Asia/Seoul")

    for _, event_row in event_df.iterrows():
        # 타임존이 없는 datetime → KST 기준으로 타임존 부여
        event_time = pd.to_datetime(event_row["datetime"])
        if event_time.tzinfo is None:
            event_time = korea_tz.localize(event_time)

        matched_news = []

        for news in news_list:
            pub_date = news.get("pub_date")
            if isinstance(pub_date, datetime) and pub_date < event_time:
                matched_news.append({
                    "title": news["title"],
                    "pub_date": pub_date.strftime("%Y-%m-%d %H:%M"),
                    "link": news.get("link", "")
                })

        # 보완 조건: 이전 뉴스가 전혀 없을 경우, 상위 일부 포함
        if not matched_news and len(news_list) > 0:
            for news in news_list[:3]:
                pub_date = news.get("pub_date")
                matched_news.append({
                    "title": news["title"],
                    "pub_date": pub_date.strftime("%Y-%m-%d %H:%M") if isinstance(pub_date, datetime) else "",
                    "link": news.get("link", "")
                })

        matched_results[event_time.strftime("%Y-%m-%d %H:%M")] = matched_news

    return matched_results