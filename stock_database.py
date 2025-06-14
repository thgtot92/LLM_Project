import json
import pandas as pd
import os
import requests

# stock_database.py

STOCK_JSON_PATH = "stocks.json"

def get_stock_database():
    """기본 종목 데이터베이스 (종목명 → (코드, 공식명))"""
    return {
        "삼성전자": ("005930", "삼성전자"),
        "sk하이닉스": ("000660", "SK하이닉스"),
        "네이버": ("035420", "NAVER"),
        "카카오": ("035720", "카카오"),
        "lg전자": ("066570", "LG전자"),
        "현대차": ("005380", "현대차"),
        "기아": ("000270", "기아"),
        "포스코홀딩스": ("005490", "POSCO홀딩스"),
        "삼성sdi": ("006400", "삼성SDI"),
        "lg화학": ("051910", "LG화학"),
        "셀트리온": ("068270", "셀트리온"),
        "삼성바이오로직스": ("207940", "삼성바이오로직스"),
        "현대모비스": ("012330", "현대모비스"),
        "kb금융": ("105560", "KB금융"),
        "신한지주": ("055550", "신한지주"),
        "한화에어로스페이스": ("012450", "한화에어로스페이스"),
        "한화에어로": ("012450", "한화에어로스페이스"),
        "한화": ("000880", "한화"),
        "대한항공": ("003490", "대한항공"),
        "한화시스템": ("272210", "한화시스템"),
        "한미반도체": ("042700", "한미반도체"),
        "원익iqe": ("090350", "원익IQE"),
        "테스": ("095610", "테스"),
        "동진쎄미켐": ("005290", "동진쎄미켐"),
        "솔브레인": ("357780", "솔브레인"),
        "실리콘웍스": ("108320", "실리콘웍스"),
        "엔씨소프트": ("036570", "엔씨소프트"),
        "넷마블": ("251270", "넷마블"),
        "크래프톤": ("259960", "크래프톤"),
        "하이브": ("352820", "하이브"),
        "아모레퍼시픽": ("090430", "아모레퍼시픽"),
        "lg생활건강": ("051900", "LG생활건강"),
        "kt": ("030200", "KT"),
        "skt": ("017670", "SK텔레콤"),
        "한국전력": ("015760", "한국전력공사"),
        "농심": ("004370", "농심"),
        "삼성생명": ("032830", "삼성생명"),
        "한화리츠": ("451800", "한화리츠")
    }

def get_stock_database_from_json(path="stocks.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_stock_database() -> dict:
    """JSON 파일에서 종목 DB 로드"""
    if os.path.exists(STOCK_JSON_PATH):
        with open(STOCK_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        return {}  # 초기 비어있을 수 있음

def save_stock_database(stock_db: dict):
    """종목 DB를 JSON 파일로 저장"""
    with open(STOCK_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(stock_db, f, ensure_ascii=False, indent=2)

def get_stock_database() -> dict:
    """최신 종목 DB 불러오기"""
    return load_stock_database()

def add_stock_entry(search_key: str, code: str, full_name: str):
    """새 종목 추가 및 저장"""
    db = load_stock_database()
    db[search_key] = [code, full_name]
    save_stock_database(db)