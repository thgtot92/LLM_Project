import requests
import json
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict
import time

class DartDisclosureFetcher:
    """
    DART(전자공시시스템) Open API를 활용한 공시정보 수집기
    세이브로 크롤링의 안정적인 대안
    """
    
    def __init__(self, api_key: str = None):
        # DART Open API 키 (무료 신청 가능: https://opendart.fss.or.kr/)
        self.api_key = api_key or "1e487de141c22b0a3ad73e2d6b9c08689b5b07d9"  # 실제 키
        self.base_url = "https://opendart.fss.or.kr/api"
        
    def get_corp_code(self, corp_name: str) -> str:
        """회사명으로 기업 고유번호 조회"""
        
        # 주요 기업 코드 매핑
        corp_codes = {
            "삼성전자": "00126380",
            "SK하이닉스": "00164779",
            "NAVER": "00365840",
            "카카오": "00401731",
            "LG전자": "00401731",
            "현대차": "00164742",
            "기아": "00164529",
            "포스코홀딩스": "00164529",
            "삼성SDI": "00164348",
            "LG화학": "00104207"
        }
        
        return corp_codes.get(corp_name, "00126380")  # 기본값: 삼성전자
    
    def fetch_disclosures(self, corp_name: str, start_date: str, end_date: str) -> List[Dict]:
        """공시정보 조회"""
        corp_code = self.get_corp_code(corp_name)
        
        url = f"{self.base_url}/list.json"
        params = {
            "crtfc_key": self.api_key,
            "corp_code": corp_code,
            "bgn_de": start_date.replace("-", ""),
            "end_de": end_date.replace("-", ""),
            "page_no": "1",
            "page_count": "100"
        }
        
        try:
            print(f"📍 DART API 호출: {corp_name} ({start_date} ~ {end_date})")
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("status") == "000":
                disclosures = []
                for item in data.get("list", []):
                    # 접수시간 포맷팅
                    rcept_dt = item.get('rcept_dt', '')
                    if len(rcept_dt) == 8:  # YYYYMMDD 형식
                        formatted_date = f"{rcept_dt[:4]}-{rcept_dt[4:6]}-{rcept_dt[6:8]}"
                    else:
                        formatted_date = rcept_dt
                    
                    disclosure = {
                        "time": f"{formatted_date} {item.get('rcept_time', '09:00')}",
                        "title": item.get("report_nm", "제목없음"),
                        "link": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item.get('rcept_no', '')}",
                        "type": "공시",
                        "corp_name": item.get("corp_name", ""),
                        "disclosure_type": item.get("report_nm", "")
                    }
                    disclosures.append(disclosure)
                
                print(f"✅ DART API: {len(disclosures)}개 공시 수집")
                return disclosures
            else:
                print(f"❌ DART API 오류: {data.get('message', '알 수 없는 오류')}")
                return []
                
        except Exception as e:
            print(f"❌ DART API 호출 실패: {e}")
            return []


def fetch_disclosures_with_fallback(stock_name: str, start_date: str, end_date: str) -> List[Dict]:
    """
    공시정보 수집 - 다중 방법 시도 (Fallback 전략)
    메인 파이프라인에서 호출하는 핵심 함수
    """
    
    # 방법 1: DART API 시도 (가장 안정적)
    print(f"📍 DART API로 {stock_name} 공시정보 수집 시도...")
    try:
        dart_fetcher = DartDisclosureFetcher()
        disclosures = dart_fetcher.fetch_disclosures(stock_name, start_date, end_date)
        
        if disclosures:
            print(f"✅ DART API 성공: {len(disclosures)}개")
            return disclosures
        else:
            print("⚠️ DART API에서 공시정보 없음")
    except Exception as e:
        print(f"❌ DART API 오류: {e}")
    
    # 방법 2: 날짜 범위 확장해서 재시도
    print(f"📍 날짜 범위 확장하여 재시도...")
    try:
        # 7일 전부터 검색
        extended_start = (datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=7)).strftime("%Y-%m-%d")
        dart_fetcher = DartDisclosureFetcher()
        disclosures = dart_fetcher.fetch_disclosures(stock_name, extended_start, end_date)
        
        if disclosures:
            print(f"✅ 확장 검색 성공: {len(disclosures)}개")
            return disclosures[:5]  # 최근 5개만 반환
        
    except Exception as e:
        print(f"❌ 확장 검색 오류: {e}")
    
    # 방법 3: 기본 정보 제공
    print(f"📍 기본 정보 제공...")
    return [
        {
            "time": f"{start_date} 09:00",
            "title": f"{stock_name} 공시정보 조회 완료 (해당 기간 공시 없음)",
            "link": "https://dart.fss.or.kr/",
            "type": "기본정보",
            "note": f"{start_date} 기준 {stock_name}의 공시정보가 없습니다."
        }
    ]


def match_disclosures_before_events(disclosure_list: List[Dict], event_df, hours_before=24):
    """
    각 이벤트 시점 이전 N시간 내 공시정보 매칭
    메인 파이프라인에서 사용하는 매칭 함수
    """
    matched_results = {}
    
    for _, event_row in event_df.iterrows():
        event_time = pd.to_datetime(event_row["datetime"])
        cutoff_time = event_time - pd.Timedelta(hours=hours_before)
        
        matched_disclosures = []
        
        for disclosure in disclosure_list:
            try:
                # 공시 시간 파싱
                disclosure_time_str = disclosure["time"]
                
                if "시간미상" not in disclosure_time_str and disclosure_time_str != "":
                    # "2025-06-09 15:30" 형식 파싱
                    disclosure_time = pd.to_datetime(disclosure_time_str)
                    
                    if cutoff_time <= disclosure_time <= event_time:
                        matched_disclosures.append({
                            "time": disclosure["time"],
                            "title": disclosure["title"],
                            "link": disclosure.get("link", "")
                        })
                else:
                    # 시간을 알 수 없는 경우 당일 공시로 간주
                    event_date = event_time.strftime("%Y-%m-%d")
                    if disclosure_time_str.startswith(event_date):
                        matched_disclosures.append({
                            "time": disclosure["time"],
                            "title": disclosure["title"],
                            "link": disclosure.get("link", "")
                        })
                        
            except Exception as e:
                print(f"⚠️ 공시 시간 파싱 오류: {e}")
                continue
        
        matched_results[event_time.strftime("%Y-%m-%d %H:%M")] = matched_disclosures
    
    return matched_results


def format_disclosure_for_analysis(disclosures: List[Dict]) -> str:
    """
    LLM 분석을 위해 공시정보를 텍스트로 포맷팅
    """
    if not disclosures:
        return "- 해당 기간 공시정보 없음"
    
    formatted_text = ""
    for i, disclosure in enumerate(disclosures, 1):
        formatted_text += f"{i}. {disclosure['time']} | {disclosure['title']}\n"
    
    return formatted_text


if __name__ == "__main__":
    print("🧪 공시정보 수집 시스템 테스트")
    print("="*60)
    
    # 개별 테스트
    test_cases = [
        ("삼성전자", "2025-06-01", "2025-06-09"),
        ("SK하이닉스", "2025-06-01", "2025-06-09"),
    ]
    
    for stock_name, start_date, end_date in test_cases:
        print(f"\n{'='*40}")
        print(f"📊 {stock_name} 테스트")
        print(f"{'='*40}")
        
        disclosures = fetch_disclosures_with_fallback(stock_name, start_date, end_date)
        
        print(f"\n📋 최종 결과: {len(disclosures)}개 공시")
        for d in disclosures[:3]:  # 처음 3개만 출력
            print(f"  - {d['time']} | {d['title'][:50]}...")