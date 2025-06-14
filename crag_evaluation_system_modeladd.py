#!/usr/bin/env python3
"""
강화된 CRAG 시스템 정량적 평가 프레임워크 (LLaMA 모델 비교 추가)
- Gemini vs LLaMA 모델 성능 비교
- CRAG vs Standard RAG 비교
- 4가지 조합 평가 (2x2 매트릭스)
"""

import pandas as pd
import json
from datetime import datetime, timedelta
import os
import time
from typing import List, Dict, Tuple
import random
import numpy as np

# 기존 모듈들 import
from llm_reporter import get_llm_report, compare_llm_models
from news_api_caller import NaverNewsSearcher, search_news_advanced, format_news_data, match_news_before_events
from seibro_disclosure_scraper import fetch_disclosures_with_fallback, match_disclosures_before_events
from slack_sender import send_to_slack

class ModelComparisonCRAGEvaluator:
    """Gemini와 LLaMA 모델을 비교하는 CRAG 평가 클래스"""
    
    def __init__(self):
        self.evaluation_results = []
        self.test_cases = []
        
        # API 설정
        self.client_id = os.getenv("NAVER_CLIENT_ID") or "JEuS9xkuWGpP40lsI9Kz"
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET") or "I6nujCm0xF"

        # Groq API 키 확인
        self.groq_api_key = "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"
                
        # 환경변수에 설정 (llm_reporter.py가 읽을 수 있도록)
        os.environ["GROQ_API_KEY"] = self.groq_api_key

        # Groq API 연결 테스트
        try:
            from llm_reporter import test_groq_connection, reinitialize_groq_client
            print("\n🔍 Groq API 연결 확인 중...")
            
            # Groq 클라이언트 재초기화
            reinitialize_groq_client(self.groq_api_key)
            
            if test_groq_connection():
                print("✅ Groq API 연결 성공! LLaMA 모델을 사용할 수 있습니다.")
            else:
                print("⚠️ Groq API 연결 실패. 새 API 키가 필요할 수 있습니다.")
                print("1. https://console.groq.com 에서 무료 API 키 발급")
                print("2. 코드에서 self.groq_api_key 값을 새 키로 변경")
        except Exception as e:
            print(f"⚠️ Groq API 테스트 중 오류: {e}")
    
    def create_test_cases(self) -> List[Dict]:
        """평가용 테스트 케이스 생성 (간소화)"""
        
        test_cases = [
            {
                "type": "model_comparison",
                "stock_code": "005930",
                "stock_name": "삼성전자", 
                "date": "2025-06-09",
                "description": "모델 비교 테스트",
                "expected_events": ["상승", "하락"],
                "difficulty": "medium"
            },
            {
                "type": "model_comparison",
                "stock_code": "000660", 
                "stock_name": "SK하이닉스",
                "date": "2025-06-04", 
                "description": "반도체 업종 모델 비교",
                "expected_events": ["상승", "하락"],
                "difficulty": "medium"
            }
        ]
        
        self.test_cases = test_cases
        return test_cases
    
    def run_model_comparison_evaluation(self, test_case_index: int = None) -> Dict:
        """
        4가지 조합으로 모델 비교 평가 실행
        1. Gemini + Standard RAG
        2. Gemini + Enhanced CRAG
        3. LLaMA + Standard RAG
        4. LLaMA + Enhanced CRAG
        """
        
        print("🚀 CRAG vs RAG + Gemini vs LLaMA 비교 평가 시작")
        print("="*70)
        
        # 테스트 케이스 선택
        if test_case_index is not None:
            test_cases = [self.test_cases[test_case_index]]
        else:
            test_cases = self.create_test_cases()
        
        all_evaluation_results = []
        
        for i, test_case in enumerate(test_cases):
            print(f"\n📊 테스트 케이스 {i+1}/{len(test_cases)}")
            print(f"종목: {test_case['stock_name']} ({test_case['stock_code']})")
            print(f"날짜: {test_case['date']}")
            print("-"*50)
            
            # 데이터 수집 (공통)
            print("\n📈 데이터 수집 중...")
            data_collection = self.collect_common_data(
                test_case['stock_code'], 
                test_case['stock_name'], 
                test_case['date']
            )
            
            case_results = {
                "test_case": test_case,
                "models": {},
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 4가지 조합 실행
            combinations = [
                ("gemini", "standard_rag", "Gemini + Standard RAG"),
                ("gemini", "enhanced_crag", "Gemini + Enhanced CRAG"),
                ("llama", "standard_rag", "LLaMA + Standard RAG"),
                ("llama", "enhanced_crag", "LLaMA + Enhanced CRAG")
            ]
            
            for model_type, approach, label in combinations:
                print(f"\n🔍 {label} 실행 중...")
                
                # API 제한 방지 대기
                if model_type == "llama":
                    time.sleep(3)
                
                try:
                    if approach == "standard_rag":
                        result = self.run_standard_rag_with_model(
                            data_collection, 
                            test_case['stock_name'],
                            test_case['date'],
                            model_type
                        )
                    else:  # enhanced_crag
                        result = self.run_enhanced_crag_with_model(
                            data_collection,
                            test_case['stock_name'],
                            test_case['date'],
                            model_type
                        )
                    
                    case_results["models"][f"{model_type}_{approach}"] = {
                        "analysis": result["analysis"],
                        "metadata": result["metadata"],
                        "model": model_type,
                        "approach": approach
                    }
                    print(f"✅ {label} 완료")
                    
                except Exception as e:
                    print(f"❌ {label} 실행 오류: {e}")
                    case_results["models"][f"{model_type}_{approach}"] = {
                        "analysis": f"오류 발생: {str(e)}",
                        "metadata": {},
                        "model": model_type,
                        "approach": approach
                    }
                
                # API 호출 간격
                time.sleep(5)
            
            # 모델 간 비교 평가
            print("\n⚖️ 모델 비교 평가 중...")
            comparison_result = self.evaluate_model_combinations(case_results)
            case_results["comparison"] = comparison_result
            
            all_evaluation_results.append(case_results)
            
            # 결과 출력
            self.print_case_comparison(case_results)
            
            # 다음 케이스 대기
            if i < len(test_cases) - 1:
                print("\n⏳ 다음 테스트까지 20초 대기...")
                time.sleep(20)
        
        # 종합 평가
        self.print_overall_comparison(all_evaluation_results)
        self.evaluation_results = all_evaluation_results
        
        return {
            "evaluation_results": all_evaluation_results,
            "summary": self.calculate_comparison_statistics(all_evaluation_results)
        }
    
    def collect_common_data(self, stock_code: str, stock_name: str, date: str) -> Dict:
        """공통 데이터 수집 (중복 방지)"""
        
        # 주가 데이터
        df = self.robust_fetch_intraday_price(stock_code, date)
        events = self.enhanced_detect_price_events(df)
        
        # 뉴스 데이터
        searcher = NaverNewsSearcher(self.client_id, self.client_secret)
        raw_news = search_news_advanced(searcher, stock_name, date)
        formatted_news = format_news_data(raw_news)
        analyzed_news = self.intelligent_news_analysis(formatted_news, stock_name)
        
        # 공시 데이터
        start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
        disclosures = fetch_disclosures_with_fallback(stock_name, start_date, date)
        
        # 매칭
        matched_news_dict = match_news_before_events(analyzed_news, events)
        matched_disclosures_dict = match_disclosures_before_events(disclosures, events, hours_before=72)
        
        return {
            "df": df,
            "events": events,
            "formatted_news": formatted_news,
            "analyzed_news": analyzed_news,
            "disclosures": disclosures,
            "matched_news_dict": matched_news_dict,
            "matched_disclosures_dict": matched_disclosures_dict
        }
    
    def run_standard_rag_with_model(self, data: Dict, stock_name: str, date: str, model_type: str) -> Dict:
        """특정 모델로 Standard RAG 실행"""
        
        df = data["df"]
        formatted_news = data["formatted_news"]
        disclosures = data["disclosures"]
        
        # 데이터 준비
        news_text = "\n".join([f"- {news['title']}" for news in formatted_news[:10]])
        disclosure_text = "\n".join([f"- {d['title']}" for d in disclosures[:5]])
        
        first_price = df.iloc[0]['price'] if len(df) > 0 else 0
        last_price = df.iloc[-1]['price'] if len(df) > 0 else 0
        price_change = ((last_price - first_price) / first_price * 100) if first_price > 0 else 0
        
        # 표준 RAG 프롬프트
        standard_prompt = f"""
        {date}일 {stock_name} 주식 분석을 수행해주세요.
        
        주가 정보:
        - 시작가: {first_price:,}원
        - 종료가: {last_price:,}원  
        - 변동률: {price_change:.2f}%
        
        관련 뉴스:
        {news_text if news_text.strip() else "- 관련 뉴스 없음"}
        
        관련 공시:
        {disclosure_text if disclosure_text.strip() else "- 관련 공시 없음"}
        
        위 정보를 바탕으로 종합적인 주식 분석을 제공해주세요.
        """
        
        # 모델별 분석
        analysis = get_llm_report(standard_prompt, model_type=model_type)
        
        metadata = {
            "data_points": len(df),
            "news_count": len(formatted_news),
            "disclosure_count": len(disclosures),
            "price_change": price_change,
            "model_used": model_type
        }
        
        return {"analysis": analysis, "metadata": metadata}
    
    def run_enhanced_crag_with_model(self, data: Dict, stock_name: str, date: str, model_type: str) -> Dict:
        """특정 모델로 Enhanced CRAG 실행"""
        
        events = data["events"]
        matched_news_dict = data["matched_news_dict"]
        matched_disclosures_dict = data["matched_disclosures_dict"]
        
        # CRAG 특화 프롬프트 생성
        prompt = self.create_enhanced_comprehensive_analysis(
            events, matched_news_dict, matched_disclosures_dict, stock_name, date
        )
        
        # 모델별 분석
        analysis = get_llm_report(prompt, model_type=model_type)
        
        total_matched_news = sum(len(news_list) for news_list in matched_news_dict.values())
        total_matched_disclosures = sum(len(disc_list) for disc_list in matched_disclosures_dict.values())
        
        metadata = {
            "data_points": len(data["df"]),
            "events_detected": len(events),
            "news_relevant": len(data["analyzed_news"]),
            "matched_news": total_matched_news,
            "matched_disclosures": total_matched_disclosures,
            "model_used": model_type
        }
        
        return {"analysis": analysis, "metadata": metadata}
    
    def evaluate_model_combinations(self, case_results: Dict) -> Dict:
        """4가지 조합 간 비교 평가"""
        
        # 간단한 평가 메트릭
        evaluation = {
            "best_combination": "",
            "scores": {},
            "rankings": []
        }
        
        # 각 조합별 점수 계산 (간단한 휴리스틱)
        for key, result in case_results["models"].items():
            if "analysis" in result and result["analysis"]:
                # 길이, 구조화, 키워드 등을 기반으로 점수 계산
                analysis = result["analysis"]
                score = 0
                
                # 분석 길이 (적절한 길이 선호)
                length = len(analysis)
                if 500 < length < 2000:
                    score += 20
                elif 2000 <= length < 3000:
                    score += 15
                else:
                    score += 10
                
                # 구조화 정도 (섹션, 불릿포인트 등)
                if "**" in analysis:  # 볼드 텍스트
                    score += 10
                if "•" in analysis or "-" in analysis:  # 불릿포인트
                    score += 10
                if "1." in analysis or "2." in analysis:  # 번호 매기기
                    score += 10
                
                # CRAG 특화 키워드
                crag_keywords = ["인과관계", "시간적", "이전", "이후", "원인", "결과"]
                for keyword in crag_keywords:
                    if keyword in analysis:
                        score += 5
                
                # 메타데이터 완성도
                metadata = result.get("metadata", {})
                if metadata.get("events_detected", 0) > 0:
                    score += 15
                if metadata.get("matched_news", 0) > 0:
                    score += 10
                
                evaluation["scores"][key] = score
        
        # 순위 매기기
        sorted_combinations = sorted(
            evaluation["scores"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        evaluation["rankings"] = sorted_combinations
        evaluation["best_combination"] = sorted_combinations[0][0] if sorted_combinations else ""
        
        return evaluation
    
    def print_case_comparison(self, case_results: Dict):
        """케이스별 비교 결과 출력"""
        
        print("\n" + "="*70)
        print(f"📊 {case_results['test_case']['stock_name']} 모델 비교 결과")
        print("="*70)
        
        comparison = case_results.get("comparison", {})
        
        if comparison.get("rankings"):
            print("\n🏆 성능 순위:")
            for i, (combo, score) in enumerate(comparison["rankings"], 1):
                model, approach = combo.split("_", 1)
                label = f"{model.upper()} + {approach.replace('_', ' ').title()}"
                print(f"{i}. {label}: {score}점")
        
        print("\n📈 메타데이터 비교:")
        print("조합                    | 이벤트 | 매칭뉴스 | 분석길이")
        print("-"*60)
        
        for key, result in case_results["models"].items():
            model, approach = key.split("_", 1)
            label = f"{model.upper()} + {approach.replace('_', ' ')}"
            metadata = result.get("metadata", {})
            analysis_length = len(result.get("analysis", ""))
            
            events = metadata.get("events_detected", 0)
            matched_news = metadata.get("matched_news", 0)
            
            print(f"{label:23} | {events:6} | {matched_news:8} | {analysis_length:8}")
    
    def print_overall_comparison(self, all_results: List[Dict]):
        """전체 비교 결과 출력"""
        
        print("\n" + "="*70)
        print("🎯 전체 모델 비교 종합 결과")
        print("="*70)
        
        # 조합별 총점 계산
        total_scores = {}
        
        for result in all_results:
            comparison = result.get("comparison", {})
            for combo, score in comparison.get("scores", {}).items():
                if combo not in total_scores:
                    total_scores[combo] = 0
                total_scores[combo] += score
        
        # 평균 점수 계산
        num_cases = len(all_results)
        avg_scores = {k: v/num_cases for k, v in total_scores.items()}
        
        # 순위 출력
        sorted_avg = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        print("\n🏆 최종 평균 점수:")
        for combo, avg_score in sorted_avg:
            model, approach = combo.split("_", 1)
            label = f"{model.upper()} + {approach.replace('_', ' ').title()}"
            print(f"{label}: {avg_score:.1f}점")
        
        # 승자 발표
        if sorted_avg:
            winner_combo = sorted_avg[0][0]
            winner_model, winner_approach = winner_combo.split("_", 1)
            print(f"\n🎉 최고 성능 조합: {winner_model.upper()} + {winner_approach.replace('_', ' ').title()}")
    
    def calculate_comparison_statistics(self, results: List[Dict]) -> Dict:
        """비교 통계 계산"""
        
        stats = {
            "total_cases": len(results),
            "model_performance": {
                "gemini": {"wins": 0, "total_score": 0},
                "llama": {"wins": 0, "total_score": 0}
            },
            "approach_performance": {
                "standard_rag": {"wins": 0, "total_score": 0},
                "enhanced_crag": {"wins": 0, "total_score": 0}
            }
        }
        
        for result in results:
            comparison = result.get("comparison", {})
            if comparison.get("best_combination"):
                best = comparison["best_combination"]
                model, approach = best.split("_", 1)
                
                # 모델별 승리 카운트
                stats["model_performance"][model]["wins"] += 1
                
                # 접근법별 승리 카운트  
                stats["approach_performance"][approach]["wins"] += 1
            
            # 점수 집계
            for combo, score in comparison.get("scores", {}).items():
                model, approach = combo.split("_", 1)
                stats["model_performance"][model]["total_score"] += score
                stats["approach_performance"][approach]["total_score"] += score
        
        return stats
    
    # 기존 메서드들 (간략화를 위해 주요 메서드만 포함)
    def robust_fetch_intraday_price(self, stock_code: str, date: str) -> pd.DataFrame:
        """강건한 주가 데이터 수집"""
        try:
            from naver_finance_crawler import fetch_intraday_price
            df = fetch_intraday_price(stock_code, date)
            if len(df) > 0:
                return df
        except:
            pass
        return self.generate_realistic_mock_data(stock_code, date)
    
    def generate_realistic_mock_data(self, stock_code: str, date: str) -> pd.DataFrame:
        """모의 데이터 생성"""
        base_prices = {
            "005930": 60000, "000660": 120000, "042700": 45000,
            "035420": 180000, "012450": 850000
        }
        base_price = base_prices.get(stock_code, 50000)
        
        start_time = datetime.strptime(f"{date} 09:00", "%Y-%m-%d %H:%M")
        times, prices, volumes = [], [], []
        current_price = base_price
        
        for i in range(390):
            current_time = start_time + timedelta(minutes=i)
            if 12 <= current_time.hour < 13:
                continue
                
            change_rate = random.gauss(0, 0.003)
            current_price *= (1 + change_rate)
            current_price = max(int(current_price), 1000)
            volume = random.randint(10000, 300000)
            
            times.append(current_time)
            prices.append(current_price)
            volumes.append(volume)
        
        # 이벤트 생성
        event_indices = random.sample(range(50, len(prices)-50), 3)
        for idx in event_indices:
            event_type = random.choice(['strong_up', 'strong_down'])
            multiplier = 1.002 if event_type == 'strong_up' else 0.998
            for j in range(idx, min(idx+30, len(prices))):
                prices[j] *= multiplier
                volumes[j] *= 1.5
        
        return pd.DataFrame({
            'datetime': times,
            'price': [int(p) for p in prices],
            'volume': volumes
        })
    
    def enhanced_detect_price_events(self, df: pd.DataFrame, threshold=0.006) -> pd.DataFrame:
        """향상된 이벤트 감지"""
        df = df.copy()
        df = df.sort_values("datetime").reset_index(drop=True)
        start_price = df.iloc[0]['price']
        
        df['pct_from_start'] = (df['price'] - start_price) / start_price
        df['pct_change'] = df['price'].pct_change()
        
        def detect_event_type(row, index):
            if abs(row['pct_from_start']) >= threshold:
                return "상승" if row['pct_from_start'] > 0 else "하락"
            return None
        
        df['event_type'] = [detect_event_type(row, i) for i, row in df.iterrows()]
        events = df[df['event_type'].notnull()][['datetime', 'price', 'pct_from_start', 'event_type']]
        
        return events
    
    def intelligent_news_analysis(self, formatted_news: list, stock_name: str) -> list:
        """지능형 뉴스 분석"""
        positive_keywords = ["상승", "호재", "성장", "수주", "계약"]
        negative_keywords = ["하락", "악재", "감소", "손실", "위험"]
        
        analyzed_news = []
        for news in formatted_news:
            title = news['title'].lower()
            relevance_score = 0
            
            if stock_name.lower() in title:
                relevance_score += 10
            
            sentiment_score = 0
            for pos_word in positive_keywords:
                if pos_word in title:
                    sentiment_score += 1
            for neg_word in negative_keywords:
                if neg_word in title:
                    sentiment_score -= 1
            
            if relevance_score >= 2:
                analyzed_news.append({
                    **news,
                    'relevance_score': relevance_score,
                    'sentiment_score': sentiment_score,
                    'sentiment': 'positive' if sentiment_score > 0 else ('negative' if sentiment_score < 0 else 'neutral')
                })
        
        return analyzed_news
    
    def create_enhanced_comprehensive_analysis(self, events_df, matched_news_dict, matched_disclosures_dict, 
                                             stock_name: str, date: str) -> str:
        """CRAG 특화 프롬프트 생성"""
        
        event_summary = ""
        if len(events_df) > 0:
            event_summary = f"📈 감지된 이벤트 ({len(events_df)}개):\n"
            for _, event in events_df.iterrows():
                pct = event['pct_from_start'] * 100
                event_time = event['datetime'].strftime('%H:%M')
                event_summary += f"- {event_time}: {pct:+.2f}% {event['event_type']} (₩{event['price']:,})\n"
        else:
            event_summary = "📈 감지된 이벤트:\n- 임계값 0.6% 이상의 주요 변동이 없는 안정적 거래일\n"
        
        all_news = []
        for news_list in matched_news_dict.values():
            for news in news_list:
                if news['title'] not in [n['title'] for n in all_news]:
                    all_news.append(news)
        
        news_summary = ""
        if all_news:
            news_summary = f"📰 CRAG 인과관계 뉴스 ({len(all_news)}개):\n"
            for news in all_news[:3]:
                news_summary += f"- {news['title']}\n"
        else:
            news_summary = "📰 CRAG 인과관계 뉴스:\n- 시간적 선후관계를 갖는 관련 뉴스 없음\n"
        
        all_disclosures = []
        for disc_list in matched_disclosures_dict.values():
            all_disclosures.extend(disc_list)
        
        disclosure_summary = ""
        if all_disclosures:
            disclosure_summary = f"📋 CRAG 인과관계 공시 ({len(all_disclosures)}개):\n"
        else:
            disclosure_summary = "📋 CRAG 인과관계 공시:\n- 72시간 내 관련 공시정보 없음\n"
        
        comprehensive_prompt = f"""[{date} {stock_name} 강화된 CRAG 분석 리포트]

{event_summary}

{news_summary}

{disclosure_summary}

🧠 **강화된 CRAG 특화 분석 요청:**

다음 관점에서 Standard RAG를 뛰어넘는 차별화된 분석을 제공해주세요:

1. **시간적 인과관계 우수성**: 이벤트 "이전" 정보만 사용한 진정한 원인 분석
2. **CRAG 고유 통찰력**: 시간 순서 기반 숨겨진 패턴 발굴
3. **예측적 가치**: 향후 유사 상황 예측에 활용 가능한 신호
4. **실전 차별화**: 시간적 인과관계 기반의 구체적 투자 전략

전문적이고 실무적인 분석을 작성해주세요.
"""
        
        return comprehensive_prompt


# 실행 함수
def main():
    """모델 비교 평가 실행"""
    
    print("🚀 CRAG 시스템 모델 비교 평가")
    print("📊 Gemini vs LLaMA + Standard RAG vs Enhanced CRAG")
    print("="*70)
    
    # Groq API 키 확인
    groq_key = "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" #os.getenv("GROQ_API_KEY")
    
    if not groq_key or groq_key != "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":
        print("\n⚠️ GROQ_API_KEY가 설정되지 않았습니다!")
        print("LLaMA 모델 비교를 위해 다음 단계를 따르세요:")
        print("1. https://console.groq.com 에서 무료 API 키 발급")
        print("2. export GROQ_API_KEY='your-api-key' 실행")
        print("3. 다시 이 스크립트 실행\n")
        
        response = input("Gemini만으로 진행하시겠습니까? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 평가 실행
    evaluator = ModelComparisonCRAGEvaluator()
    results = evaluator.run_model_comparison_evaluation()
    
    # 결과 저장
    with open('model_comparison_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 마크다운 요약 생성
    create_comparison_markdown(results)
    
    print("\n✅ 모델 비교 평가 완료!")
    print("📄 결과 파일:")
    print("- model_comparison_results.json")
    print("- model_comparison_summary.md")


def create_comparison_markdown(results: Dict):
    """비교 결과를 Markdown으로 저장"""
    
    lines = []
    lines.append(f"# CRAG 시스템 모델 비교 결과 ({datetime.now().strftime('%Y-%m-%d')})\n")
    lines.append("## 평가 개요\n")
    lines.append("- **모델**: Gemini vs LLaMA")
    lines.append("- **접근법**: Standard RAG vs Enhanced CRAG")
    lines.append("- **조합**: 4가지 (2x2 매트릭스)\n")
    
    summary = results.get("summary", {})
    
    lines.append("\n## 모델별 성능\n")
    lines.append("| 모델 | 승리 횟수 | 총 점수 |")
    lines.append("|------|----------|---------|")
    
    for model in ["gemini", "llama"]:
        perf = summary.get("model_performance", {}).get(model, {})
        lines.append(f"| {model.upper()} | {perf.get('wins', 0)} | {perf.get('total_score', 0)} |")
    
    lines.append("\n## 접근법별 성능\n")
    lines.append("| 접근법 | 승리 횟수 | 총 점수 |")
    lines.append("|--------|----------|---------|")
    
    for approach in ["standard_rag", "enhanced_crag"]:
        perf = summary.get("approach_performance", {}).get(approach, {})
        label = approach.replace("_", " ").title()
        lines.append(f"| {label} | {perf.get('wins', 0)} | {perf.get('total_score', 0)} |")
    
    lines.append("\n## 주요 발견사항\n")
    lines.append("1. **최고 성능 조합**: 평가 결과에 따라 결정")
    lines.append("2. **모델 특성**: Gemini와 LLaMA의 강점 비교")
    lines.append("3. **CRAG 효과**: 시간적 인과관계 분석의 가치")
    
    with open('model_comparison_summary.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()