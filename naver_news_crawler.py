
import requests
import json
import time
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import re
from urllib.parse import quote
import os

class EnhancedNewsCollector:
    """향상된 뉴스 수집기"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.headers = {
            "X-Naver-Client-Id": client_id,
            "X-Naver-Client-Secret": client_secret
        }
        self.base_url = "https://openapi.naver.com/v1/search/news.json"
        
        # 종목별 키워드 매핑 (관련성 향상)
        self.stock_keywords = {
            "삼성전자": ["삼성전자", "삼전", "반도체", "갤럭시", "엑시노스", "파운드리"],
            "SK하이닉스": ["SK하이닉스", "하이닉스", "D램", "낸드", "메모리반도체"],
            "LG에너지솔루션": ["LG에너지솔루션", "LG에너지", "배터리", "이차전지", "전기차배터리"],
            "현대차": ["현대자동차", "현대차", "아이오닉", "제네시스", "전기차"],
            "NAVER": ["네이버", "NAVER", "라인", "웹툰", "클로바"],
            "카카오": ["카카오", "kakao", "카카오톡", "카카오페이", "카카오뱅크"],
            "바이오": ["바이오", "신약", "임상", "FDA", "품목허가"],
            "엔터": ["엔터테인먼트", "엔터", "아이돌", "콘텐츠", "IP"]
        }
        
        # 뉴스 품질 필터링 키워드
        self.quality_keywords = {
            "positive": ["상승", "급등", "호재", "신고가", "성장", "수주", "계약", "승인", "출시"],
            "negative": ["하락", "급락", "악재", "저가", "감소", "손실", "리콜", "소송", "지연"],
            "neutral": ["전망", "분석", "평가", "예상", "계획", "검토", "추진", "협의"]
        }
    
    def search_news_multi_strategy(self, stock_name: str, date: str, 
                                 days_before: int = 3, days_after: int = 1) -> List[Dict]:
        """
        다양한 전략으로 뉴스 검색
        
        Args:
            stock_name: 종목명
            date: 기준 날짜 (YYYY-MM-DD)
            days_before: 며칠 전까지 검색
            days_after: 며칠 후까지 검색
            
        Returns:
            수집된 뉴스 리스트
        """
        all_news = []
        seen_titles = set()  # 중복 제거용
        
        # 날짜 범위 설정
        base_date = datetime.strptime(date, "%Y-%m-%d")
        start_date = base_date - timedelta(days=days_before)
        end_date = base_date + timedelta(days=days_after)
        
        # 전략 1: 종목명 직접 검색
        print(f"🔍 전략 1: '{stock_name}' 직접 검색")
        news1 = self._search_news(
            query=stock_name,
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            display=50
        )
        all_news.extend(news1)
        
        # 전략 2: 종목 관련 키워드 조합 검색
        related_keywords = self._get_related_keywords(stock_name)
        if related_keywords:
            for keyword in related_keywords[:3]:  # 상위 3개 키워드
                if keyword != stock_name:  # 중복 방지
                    print(f"🔍 전략 2: '{stock_name} {keyword}' 조합 검색")
                    news2 = self._search_news(
                        query=f"{stock_name} {keyword}",
                        start_date=start_date.strftime("%Y-%m-%d"),
                        end_date=end_date.strftime("%Y-%m-%d"),
                        display=30
                    )
                    all_news.extend(news2)
                    time.sleep(0.1)  # API 제한 방지
        
        # 전략 3: 특정 날짜 집중 검색
        print(f"🔍 전략 3: {date} 날짜 집중 검색")
        news3 = self._search_news(
            query=stock_name,
            start_date=date,
            end_date=date,
            display=30,
            sort="sim"  # 정확도순
        )
        all_news.extend(news3)
        
        # 전략 4: 주요 이벤트 키워드 조합
        event_keywords = ["실적", "공시", "발표", "계약", "수주"]
        for event_keyword in event_keywords[:2]:
            print(f"🔍 전략 4: '{stock_name} {event_keyword}' 이벤트 검색")
            news4 = self._search_news(
                query=f"{stock_name} {event_keyword}",
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                display=20
            )
            all_news.extend(news4)
            time.sleep(0.1)
        
        # 중복 제거 및 정렬
        unique_news = []
        for news in all_news:
            if news['title'] not in seen_titles:
                seen_titles.add(news['title'])
                unique_news.append(news)
        
        # 관련성 점수 계산 및 정렬
        scored_news = self._calculate_relevance_scores(unique_news, stock_name, date)
        scored_news.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        print(f"✅ 총 {len(scored_news)}개 뉴스 수집 완료")
        return scored_news
    
    def _search_news(self, query: str, start_date: str, end_date: str, 
                    display: int = 50, sort: str = "date") -> List[Dict]:
        """
        네이버 뉴스 API 호출
        
        Args:
            query: 검색어
            start_date: 시작 날짜 (YYYY-MM-DD) - 참고용
            end_date: 종료 날짜 (YYYY-MM-DD) - 참고용
            display: 검색 결과 개수 (최대 100)
            sort: 정렬 방식 (date: 날짜순, sim: 정확도순)
        """
        news_items = []
        
        params = {
            "query": query,
            "display": min(display, 100),
            "start": 1,
            "sort": sort
        }
        
        print(f"  API 호출: {query} (display={params['display']})")
        
        try:
            response = requests.get(self.base_url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            items = data.get("items", [])
            
            print(f"  API 응답: {len(items)}개 항목")
            
            # 날짜 필터링 기준 설정
            start_datetime = datetime.strptime(start_date + " 00:00:00", "%Y-%m-%d %H:%M:%S")
            end_datetime = datetime.strptime(end_date + " 23:59:59", "%Y-%m-%d %H:%M:%S")
            
            for item in items:
                # 날짜 파싱
                pub_date = self._parse_date(item.get("pubDate", ""))
                
                if pub_date:
                    try:
                        pub_datetime = datetime.strptime(pub_date, "%Y-%m-%d %H:%M:%S")
                        
                        # 날짜 필터링 (선택적)
                        # 네이버 API는 최신 뉴스부터 제공하므로, 너무 오래된 뉴스만 제외
                        days_diff = abs((pub_datetime - start_datetime).days)
                        if days_diff <= 7:  # 7일 이내 뉴스만
                            news_item = {
                                "title": self._clean_html(item.get("title", "")),
                                "link": item.get("link", ""),
                                "description": self._clean_html(item.get("description", "")),
                                "pubDate": pub_date,
                                "originallink": item.get("originallink", "")
                            }
                            news_items.append(news_item)
                    except Exception as e:
                        print(f"  날짜 파싱 오류: {e}")
                        # 날짜 파싱 실패해도 뉴스는 포함
                        news_item = {
                            "title": self._clean_html(item.get("title", "")),
                            "link": item.get("link", ""),
                            "description": self._clean_html(item.get("description", "")),
                            "pubDate": item.get("pubDate", ""),
                            "originallink": item.get("originallink", "")
                        }
                        news_items.append(news_item)
            
            print(f"  필터링 후: {len(news_items)}개 뉴스")
            
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 오류: {e}")
            if hasattr(e, 'response') and e.response:
                print(f"   상태 코드: {e.response.status_code}")
                print(f"   응답 내용: {e.response.text[:200]}")
        except Exception as e:
            print(f"❌ 뉴스 검색 오류: {e}")
            import traceback
            traceback.print_exc()
        
        return news_items
    
    def _get_related_keywords(self, stock_name: str) -> List[str]:
        """종목명과 관련된 키워드 반환"""
        # 직접 매칭
        if stock_name in self.stock_keywords:
            return self.stock_keywords[stock_name]
        
        # 부분 매칭
        for key, keywords in self.stock_keywords.items():
            if key in stock_name or stock_name in key:
                return keywords
        
        # 업종별 키워드
        if "바이오" in stock_name or "제약" in stock_name:
            return self.stock_keywords.get("바이오", [])
        elif "엔터" in stock_name:
            return self.stock_keywords.get("엔터", [])
        
        return []
    
    def _calculate_relevance_scores(self, news_list: List[Dict], 
                                  stock_name: str, target_date: str) -> List[Dict]:
        """
        뉴스 관련성 점수 계산
        
        점수 기준:
        - 제목에 종목명 포함: +10점
        - 본문에 종목명 포함: +5점
        - 날짜 근접성: 최대 10점
        - 품질 키워드: +3점씩
        - 관련 키워드: +2점씩
        """
        target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
        related_keywords = self._get_related_keywords(stock_name)
        
        for news in news_list:
            score = 0
            
            title = news.get("title", "").lower()
            description = news.get("description", "").lower()
            combined_text = title + " " + description
            
            # 1. 종목명 포함 여부
            if stock_name.lower() in title:
                score += 10
            if stock_name.lower() in description:
                score += 5
            
            # 2. 날짜 근접성 (최대 10점)
            news_date = datetime.strptime(news["pubDate"], "%Y-%m-%d %H:%M:%S")
            date_diff = abs((news_date - target_datetime).days)
            date_score = max(0, 10 - date_diff * 2)
            score += date_score
            
            # 3. 품질 키워드
            for category, keywords in self.quality_keywords.items():
                for keyword in keywords:
                    if keyword in combined_text:
                        score += 3
            
            # 4. 관련 키워드
            for keyword in related_keywords:
                if keyword.lower() in combined_text:
                    score += 2
            
            # 5. 부정적 신호 (광고, 홍보성)
            if any(word in title for word in ["광고", "제공", "보도자료"]):
                score -= 5
            
            news["relevance_score"] = score
            news["date_diff"] = date_diff
        
        return news_list
    
    def _parse_date(self, date_str: str) -> str:
        """네이버 API 날짜 형식 파싱"""
        try:
            # 예: "Mon, 10 Jun 2024 15:30:00 +0900"
            from email.utils import parsedate_to_datetime
            dt = parsedate_to_datetime(date_str)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return ""
    
    def _clean_html(self, text: str) -> str:
        """HTML 태그 제거"""
        # <b>, </b> 등 태그 제거
        text = re.sub('<.*?>', '', text)
        # &quot; 등 HTML 엔티티 변환
        text = text.replace('&quot;', '"').replace('&amp;', '&')
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        return text.strip()
    
    def analyze_news_impact(self, news_list: List[Dict], stock_name: str) -> Dict:
        """
        뉴스 영향력 분석
        
        Returns:
            {
                "positive_count": int,
                "negative_count": int,
                "neutral_count": int,
                "key_events": List[Dict],
                "sentiment_score": float,
                "impact_summary": str
            }
        """
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        key_events = []
        
        for news in news_list[:20]:  # 상위 20개 분석
            title = news.get("title", "").lower()
            description = news.get("description", "").lower()
            combined = title + " " + description
            
            # 감성 분석
            pos_score = sum(1 for word in self.quality_keywords["positive"] if word in combined)
            neg_score = sum(1 for word in self.quality_keywords["negative"] if word in combined)
            
            if pos_score > neg_score:
                positive_count += 1
                sentiment = "positive"
            elif neg_score > pos_score:
                negative_count += 1
                sentiment = "negative"
            else:
                neutral_count += 1
                sentiment = "neutral"
            
            # 주요 이벤트 추출
            if news.get("relevance_score", 0) >= 15:  # 고관련성 뉴스
                key_events.append({
                    "title": news["title"],
                    "date": news["pubDate"],
                    "sentiment": sentiment,
                    "relevance_score": news.get("relevance_score", 0)
                })
        
        # 종합 감성 점수 (-1 ~ 1)
        total_news = positive_count + negative_count + neutral_count
        if total_news > 0:
            sentiment_score = (positive_count - negative_count) / total_news
        else:
            sentiment_score = 0
        
        # 영향력 요약
        if sentiment_score > 0.3:
            impact_summary = f"{stock_name}에 대한 긍정적인 뉴스가 우세합니다. 호재 중심의 보도가 이어지고 있습니다."
        elif sentiment_score < -0.3:
            impact_summary = f"{stock_name}에 대한 부정적인 뉴스가 많습니다. 투자 심리에 악영향을 줄 수 있습니다."
        else:
            impact_summary = f"{stock_name} 관련 뉴스는 중립적이거나 혼재된 상황입니다."
        
        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "key_events": key_events[:5],  # 상위 5개
            "sentiment_score": round(sentiment_score, 3),
            "impact_summary": impact_summary,
            "total_analyzed": total_news
        }
    
    def get_competitor_news(self, stock_name: str, date: str) -> List[Dict]:
        """경쟁사 관련 뉴스도 함께 수집"""
        competitors = {
            "삼성전자": ["SK하이닉스", "인텔", "TSMC"],
            "SK하이닉스": ["삼성전자", "마이크론", "웨스턴디지털"],
            "LG에너지솔루션": ["삼성SDI", "SK온", "CATL"],
            "현대차": ["기아", "테슬라", "폭스바겐"],
            "NAVER": ["카카오", "구글", "쿠팡"],
            "카카오": ["네이버", "라인", "쿠팡"]
        }
        
        competitor_news = []
        if stock_name in competitors:
            print(f"\n🏢 경쟁사 뉴스 수집 중...")
            for competitor in competitors[stock_name][:2]:  # 상위 2개 경쟁사
                news = self._search_news(
                    query=competitor,
                    start_date=date,
                    end_date=date,
                    display=10
                )
                for item in news:
                    item["competitor"] = competitor
                    item["original_stock"] = stock_name
                competitor_news.extend(news)
        
        return competitor_news


# 사용 예시 및 테스트
def test_enhanced_news_collector():
    """향상된 뉴스 수집기 테스트"""
    
    # API 키 설정 (실제 사용 시 환경변수 사용)
    client_id = os.getenv("NAVER_CLIENT_ID", "JEuS9xkuWGpP40lsI9Kz")
    client_secret = os.getenv("NAVER_CLIENT_SECRET", "I6nujCm0xF")
    
    collector = EnhancedNewsCollector(client_id, client_secret)
    
    # 테스트 케이스
    test_cases = [
        ("삼성전자", "2024-06-10"),
        ("SK하이닉스", "2024-06-10"),
        ("LG에너지솔루션", "2024-06-10")
    ]
    
    for stock_name, date in test_cases:
        print(f"\n{'='*70}")
        print(f"📊 {stock_name} - {date} 뉴스 수집 테스트")
        print(f"{'='*70}")
        
        # 다중 전략 뉴스 수집
        news_list = collector.search_news_multi_strategy(
            stock_name=stock_name,
            date=date,
            days_before=3,
            days_after=0
        )
        
        print(f"\n📰 수집된 뉴스 상위 10개:")
        for i, news in enumerate(news_list[:10], 1):
            print(f"{i}. [{news['relevance_score']}점] {news['title']}")
            print(f"   날짜: {news['pubDate']} | 차이: {news.get('date_diff', 0)}일")
        
        # 영향력 분석
        impact_analysis = collector.analyze_news_impact(news_list, stock_name)
        print(f"\n📈 뉴스 영향력 분석:")
        print(f"- 긍정: {impact_analysis['positive_count']}개")
        print(f"- 부정: {impact_analysis['negative_count']}개")
        print(f"- 중립: {impact_analysis['neutral_count']}개")
        print(f"- 감성점수: {impact_analysis['sentiment_score']}")
        print(f"- 요약: {impact_analysis['impact_summary']}")
        
        # 주요 이벤트
        if impact_analysis['key_events']:
            print(f"\n🎯 주요 이벤트:")
            for event in impact_analysis['key_events']:
                print(f"- {event['title']} ({event['sentiment']})")
        
        # 경쟁사 뉴스
        competitor_news = collector.get_competitor_news(stock_name, date)
        if competitor_news:
            print(f"\n🏢 경쟁사 뉴스:")
            for news in competitor_news[:5]:
                print(f"- [{news['competitor']}] {news['title']}")
        
        time.sleep(1)  # API 제한 방지


if __name__ == "__main__":
    print("🚀 향상된 뉴스 수집 시스템 테스트")
    print("="*70)
    
    test_enhanced_news_collector()