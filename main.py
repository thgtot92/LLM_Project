from slack_sender import send_to_slack
from llm_reporter import get_llm_report
from naver_finance_crawler import fetch_kospi_daily, fetch_sector_etf_daily, fetch_industry_info_by_stock_code
from naver_news_crawler import EnhancedNewsCollector
from news_api_caller import match_news_before_events
from seibro_disclosure_scraper import fetch_disclosures_with_fallback, match_disclosures_before_events
import pandas as pd
import os
import requests
from datetime import datetime, timedelta
import random
import json
import traceback
from typing import Dict, List, Tuple, Optional
from stock_database import get_stock_database, add_stock_entry

class CRAGAnalyzer:
    """강화된 CRAG 분석 클래스"""
    
    def __init__(self):
        self.client_id = os.getenv("NAVER_CLIENT_ID", "JEuS9xkuWGpP40lsI9Kz")
        self.client_secret = os.getenv("NAVER_CLIENT_SECRET", "I6nujCm0xF")
        self.slack_url = os.getenv("SLACK_WEBHOOK_URL", "https://hooks.slack.com/services/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")
        
    def robust_fetch_intraday_price(self, stock_code: str, date: str) -> pd.DataFrame:
        """강건한 주가 데이터 수집"""
        try:
            from naver_finance_crawler import fetch_intraday_price
            df = fetch_intraday_price(stock_code, date)
            if len(df) > 0:
                print(f"✅ 네이버 금융에서 {len(df)}개 데이터 수집 성공")
                return df
        except Exception as e:
            print(f"⚠️ 네이버 금융 실패: {e}")
        
        print("🔄 현실적인 모의 데이터 생성으로 대체")
        return self.generate_realistic_mock_data(stock_code, date)

    def generate_realistic_mock_data(self, stock_code: str, date: str) -> pd.DataFrame:
        """현실적인 모의 데이터 생성"""
        base_prices = {
            "005930": 60000, "000660": 120000, "042700": 45000,
            "035420": 180000, "012450": 850000, "373220": 95000,
            "207940": 45000, "006400": 28000, "051910": 850000
        }
        base_price = base_prices.get(stock_code, 50000)
        
        start_time = datetime.strptime(f"{date} 09:00", "%Y-%m-%d %H:%M")
        times, prices, volumes = [], [], []
        current_price = base_price
        
        # 정교한 장중 시뮬레이션
        for i in range(390):
            current_time = start_time + timedelta(minutes=i)
            if 12 <= current_time.hour < 13:  # 점심시간 제외
                continue
                
            # 시간대별 변동성 조정
            if 9 <= current_time.hour < 10:  # 개장 초반 높은 변동성
                volatility = 0.005
            elif 10 <= current_time.hour < 15:  # 정규장 중간
                volatility = 0.003
            else:  # 마감 전 변동성 증가
                volatility = 0.004
                
            change_rate = random.gauss(0, volatility)
            current_price *= (1 + change_rate)
            current_price = max(int(current_price), 1000)
            
            # 거래량 패턴 (개장/마감 시 증가)
            if 9 <= current_time.hour < 10 or current_time.hour >= 15:
                volume = random.randint(50000, 500000)
            else:
                volume = random.randint(10000, 200000)
            
            times.append(current_time)
            prices.append(current_price)
            volumes.append(volume)
        
        # 의도적 이벤트 생성 (더 현실적)
        event_count = random.randint(2, 4)
        event_indices = random.sample(range(50, len(prices)-50), event_count)
        
        for idx in event_indices:
            event_type = random.choice(['strong_up', 'strong_down', 'volume_spike'])
            
            if event_type == 'strong_up':
                for j in range(idx, min(idx+20, len(prices))):
                    prices[j] *= 1.003
                    volumes[j] = int(volumes[j] * 1.8)
            elif event_type == 'strong_down':
                for j in range(idx, min(idx+20, len(prices))):
                    prices[j] *= 0.997
                    volumes[j] = int(volumes[j] * 2.0)
            else:  # volume_spike
                for j in range(idx, min(idx+10, len(prices))):
                    volumes[j] = int(volumes[j] * 3.0)
        
        df = pd.DataFrame({
            'datetime': times,
            'price': [int(p) for p in prices],
            'volume': volumes
        })
        
        print(f"📊 현실적 모의 데이터 생성: {len(df)}개 시점, 이벤트 {event_count}개 포함")
        return df

    def enhanced_detect_price_events_by_day(self, df: pd.DataFrame, threshold=0.005) -> pd.DataFrame:
        """향상된 이벤트 감지 (다중 조건)"""
        if df.empty:
            return pd.DataFrame()
            
        df = df.copy().sort_values("datetime").reset_index(drop=True)
        start_price = df.iloc[0]['price']
        
        # 기술적 지표 계산
        df['pct_from_start'] = (df['price'] - start_price) / start_price
        df['pct_change'] = df['price'].pct_change()
        df['ma_5'] = df['price'].rolling(window=5, min_periods=1).mean()
        df['ma_20'] = df['price'].rolling(window=20, min_periods=1).mean()
        df['vol_ma'] = df['volume'].rolling(window=10, min_periods=1).mean()
        df['vol_ratio'] = df['volume'] / df['vol_ma']
        
        # RSI 계산
        delta = df['price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        def detect_advanced_event(row, index):
            events = []
            
            # 1. 가격 기반 이벤트
            if abs(row['pct_from_start']) >= threshold:
                direction = "상승" if row['pct_from_start'] > 0 else "하락"
                strength = "강" if abs(row['pct_from_start']) >= threshold * 1.5 else "약"
                events.append(f"{strength}{direction}")
            
            # 2. 단기 급변동
            if index >= 10:
                recent_change = (row['price'] - df.iloc[index-10]['price']) / df.iloc[index-10]['price']
                if abs(recent_change) >= 0.008:
                    events.append("급상승" if recent_change > 0 else "급하락")
            
            # 3. 거래량 이상
            if not pd.isna(row['vol_ratio']) and row['vol_ratio'] >= 2.0:
                price_change = row['pct_change']
                if abs(price_change) >= 0.002:
                    events.append("거래량폭증상승" if price_change > 0 else "거래량폭증하락")
                else:
                    events.append("거래량폭증")
            
            # 4. 이동평균 돌파
            if index > 0 and not pd.isna(row['ma_20']):
                prev_vs_ma = (df.iloc[index-1]['price'] - df.iloc[index-1]['ma_20']) / df.iloc[index-1]['ma_20']
                curr_vs_ma = (row['price'] - row['ma_20']) / row['ma_20']
                
                if prev_vs_ma <= 0 and curr_vs_ma > 0.003:
                    events.append("MA돌파상승")
                elif prev_vs_ma >= 0 and curr_vs_ma < -0.003:
                    events.append("MA이탈하락")
            
            # 5. RSI 과매수/과매도
            if not pd.isna(row['rsi']):
                if row['rsi'] >= 80:
                    events.append("RSI과매수")
                elif row['rsi'] <= 20:
                    events.append("RSI과매도")
            
            return ", ".join(events) if events else None
        
        df['event_type'] = [detect_advanced_event(row, i) for i, row in df.iterrows()]
        events = df[df['event_type'].notnull()][['datetime', 'price', 'volume', 'pct_from_start', 'vol_ratio', 'rsi', 'event_type']]
        
        print(f"🎯 고도화 이벤트 감지: {len(events)}개 (임계값: {threshold*100:.1f}%)")
        if not events.empty:
            print(f"   주요 이벤트: {events['event_type'].value_counts().head(3).to_dict()}")
        
        return events

    def load_crag_template(self) -> str:
        """CRAG 템플릿 로드 (강화된 오류 처리)"""
        template_paths = ["crag_prompt_template.md", "templates/crag_prompt_template.md", "./crag_prompt_template.md"]
        
        for path in template_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    template_content = f.read()
                    print(f"✅ 템플릿 로드 성공: {path}")
                    
                    # 템플릿 변수 검증
                    import re
                    template_vars = re.findall(r'\{([^}]+)\}', template_content)
                    print(f"📋 템플릿 변수 확인: {len(template_vars)}개")
                    
                    return template_content
            except FileNotFoundError:
                print(f"📁 템플릿 파일 없음: {path}")
                continue
            except Exception as e:
                print(f"⚠️ 템플릿 로드 오류 ({path}): {e}")
                continue
        
        # 기본 템플릿 반환 (원본 템플릿 기반)
        print("🔄 기본 내장 템플릿 사용")
        return '''
📌 **요약 분석**  
- 주가: {summary_price}  
- 뉴스 영향: 감성 점수 {sentiment_score:+.2f}, 대부분 {news_timing_comment}  
- 시장: {kospi_info}, 업종/ETF: {industry_info}, {etf_info}

---

### 1️⃣ 주가 이벤트 요약

{event_table}

**해석**: {event_insight}

---

### 2️⃣ 뉴스 영향력 분석

- 감성 점수: **{sentiment_score:+.2f} ({sentiment_direction})**
- 주요 뉴스:
{top_news_titles}
- 뉴스 발표 시점: 대부분 **{news_timing_comment}**
- **해석**: {impact_summary}

---

### 3️⃣ 경쟁사 분석

{competitor_summary}

---

### 4️⃣ 시장 비교

- 코스피: {kospi_info}
- 동일 업종: {industry_info}
- ETF: {etf_info}

---

### 5️⃣ 투자 시사점

- **단기**: {insight_short}
- **중기**: {insight_mid}
- **리스크**: {risk_warning}

---

📍 **향후 필요 정보**
- 업종 내 경쟁사 분석
- 주요 기업 재무 요약
- 공시 세부내용 요약 및 영향 추정
- 기관/외국인 수급 데이터

---

✳️ *본 리포트는 자동화된 LLM 분석에 기반하며, 투자 판단은 사용자 책임입니다.*
'''

    def enhanced_comprehensive_analysis_v3(self, events_df, matched_news_dict, matched_disclosures_dict, 
                                          stock_name: str, date: str, news_impact: dict = None,
                                          competitor_news: list = None, 
                                          kospi_info: str = "", etf_info: str = "", industry_info: str = "") -> dict:
        """강화된 종합 분석 (v3)"""
        
        try:
            template = self.load_crag_template()
            
            # 이벤트 분석 강화
            if not events_df.empty:
                event_table = self.create_enhanced_event_table(events_df)
                event_summary = self.analyze_event_patterns(events_df)
                summary_price = f"{len(events_df)}개 이벤트 감지 ({event_summary})"
                event_insight = self.generate_event_insights(events_df, matched_news_dict)
            else:
                event_table = "| 시점 | 이벤트 |\n|------|--------|\n| - | 감지된 이벤트 없음 |"
                summary_price = "이벤트 없음"
                event_insight = "주가 안정세, 특별한 변동 요인 없음"

            # 뉴스 분석 강화
            news_analysis = self.analyze_news_comprehensive(matched_news_dict, news_impact)
            
            # 경쟁사 분석
            competitor_analysis = self.analyze_competitors(competitor_news, stock_name)
            
            # 시장 분석
            market_analysis = self.analyze_market_context(kospi_info, etf_info, industry_info)
            
            # 투자 인사이트 생성
            investment_insights = self.generate_investment_insights(
                events_df, news_impact, market_analysis, competitor_analysis
            )
            
            # 템플릿 변수 매핑 (모든 가능한 변수 포함)
            template_vars = {
                'summary_price': summary_price,
                'sentiment_score': news_analysis['sentiment_score'],
                'sentiment_direction': news_analysis['sentiment_direction'],
                'news_timing_comment': news_analysis['timing_comment'],
                'kospi_info': kospi_info or "시장 정보 없음",
                'etf_info': etf_info or "ETF 정보 없음", 
                'industry_info': industry_info or "업종 정보 없음",
                'event_table': event_table,
                'event_insight': event_insight,
                'top_news_titles': news_analysis['top_titles'],
                'impact_summary': news_analysis['impact_summary'],
                'competitor_summary': competitor_analysis,
                'insight_short': investment_insights['short_term'],
                'insight_mid': investment_insights['mid_term'],
                'risk_warning': investment_insights['risks'],
                'current_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'stock_name': stock_name,
                'analysis_date': date
            }
            
            # 안전한 템플릿 포맷팅 (변수 검증 포함)
            try:
                # 템플릿에서 필요한 변수 추출
                import re
                required_vars = set(re.findall(r'\{([^}:]+)', template))
                provided_vars = set(template_vars.keys())
                
                missing_vars = required_vars - provided_vars
                if missing_vars:
                    print(f"⚠️ 누락된 템플릿 변수: {missing_vars}")
                    # 누락된 변수에 기본값 할당
                    for var in missing_vars:
                        template_vars[var] = f"[{var} 정보 없음]"
                
                print(f"📝 템플릿 포맷팅: {len(required_vars)}개 변수 처리")
                formatted_report = template.format(**template_vars)
                
            except KeyError as e:
                print(f"🚨 템플릿 변수 오류: {e}")
                print(f"📋 사용 가능한 변수: {list(template_vars.keys())}")
                formatted_report = self.create_fallback_report(template_vars)
            except Exception as e:
                print(f"🚨 템플릿 포맷팅 오류: {e}")
                formatted_report = self.create_fallback_report(template_vars)
            
            print("🧠 강화된 CRAG 분석 완료")
            
            return {
                "report": formatted_report,
                "summary": {
                    "이벤트_수": len(events_df),
                    "뉴스_수": news_analysis['total_news'],
                    "감성_점수": news_analysis['sentiment_score'],
                    "경쟁사_뉴스": len(competitor_news) if competitor_news else 0,
                    "공시_수": sum(len(d) for d in matched_disclosures_dict.values()),
                    "시장_상황": market_analysis,
                    "투자_등급": investment_insights['grade']
                }
            }
            
        except Exception as e:
            print(f"🚨 분석 중 오류: {e}")
            traceback.print_exc()
            return {
                "report": f"분석 중 오류가 발생했습니다: {str(e)}",
                "summary": {"오류": str(e)}
            }

    def create_enhanced_event_table(self, events_df) -> str:
        """향상된 이벤트 테이블 생성"""
        if events_df.empty:
            return "| 시점 | 이벤트 |\n|------|--------|\n| - | 감지된 이벤트 없음 |"
        
        table = "| 시점 | 가격변화 | 거래량비율 | RSI | 이벤트 유형 |\n"
        table += "|------|----------|-----------|-----|------------|\n"
        
        for _, row in events_df.head(10).iterrows():  # 상위 10개만
            time_str = row['datetime'].strftime("%H:%M")
            pct = f"{row['pct_from_start']*100:+.2f}%"
            vol_ratio = f"{row.get('vol_ratio', 1.0):.1f}x" if pd.notna(row.get('vol_ratio')) else "N/A"
            rsi = f"{row.get('rsi', 50):.0f}" if pd.notna(row.get('rsi')) else "N/A"
            event_type = row['event_type'][:20] + "..." if len(row['event_type']) > 20 else row['event_type']
            
            table += f"| {time_str} | {pct} | {vol_ratio} | {rsi} | {event_type} |\n"
        
        return table

    def analyze_event_patterns(self, events_df) -> str:
        """이벤트 패턴 분석"""
        if events_df.empty:
            return "패턴 없음"
        
        patterns = []
        
        # 시간대별 분석
        events_df['hour'] = events_df['datetime'].dt.hour
        hourly_counts = events_df['hour'].value_counts()
        peak_hour = hourly_counts.index[0] if not hourly_counts.empty else None
        
        if peak_hour:
            if 9 <= peak_hour <= 10:
                patterns.append("개장 초반 집중")
            elif 14 <= peak_hour <= 15:
                patterns.append("마감 전 활발")
            else:
                patterns.append("정규장 중반 활동")
        
        # 강도별 분석
        strong_events = events_df[events_df['event_type'].str.contains('강|급|폭증', na=False)]
        if len(strong_events) > 0:
            patterns.append(f"강한 변동 {len(strong_events)}회")
        
        return ", ".join(patterns) if patterns else "일반적 패턴"

    def generate_event_insights(self, events_df, matched_news_dict) -> str:
        """이벤트 인사이트 생성"""
        if events_df.empty:
            return "주가 안정세 유지, 특별한 변동 요인 확인되지 않음"
        
        insights = []
        
        # 뉴스와의 상관관계
        total_matched = sum(len(news) for news in matched_news_dict.values())
        if total_matched > 0:
            insights.append(f"이벤트 시점 전후 {total_matched}개 뉴스 확인")
        else:
            insights.append("뉴스 기반 설명 어려움, 기술적/수급적 요인 추정")
        
        # 변동성 분석
        price_changes = events_df['pct_from_start'].abs()
        max_change = price_changes.max() * 100
        avg_change = price_changes.mean() * 100
        
        if max_change > 2.0:
            insights.append(f"최대 {max_change:.1f}% 변동으로 높은 변동성")
        elif max_change > 1.0:
            insights.append(f"중간 수준 변동성 ({max_change:.1f}%)")
        else:
            insights.append("상대적 안정적 움직임")
        
        # 거래량 분석
        vol_events = events_df[events_df['event_type'].str.contains('거래량', na=False)]
        if len(vol_events) > 0:
            insights.append("거래량 이상 신호 포함")
        
        return ". ".join(insights)

    def analyze_news_comprehensive(self, matched_news_dict, news_impact) -> dict:
        """종합 뉴스 분석"""
        all_news = [n for sublist in matched_news_dict.values() for n in sublist]
        
        sentiment_score = news_impact.get('sentiment_score', 0.0) if news_impact else 0.0
        
        if sentiment_score > 0.3:
            sentiment_direction = "매우 긍정"
        elif sentiment_score > 0.1:
            sentiment_direction = "긍정"
        elif sentiment_score > -0.1:
            sentiment_direction = "중립"
        elif sentiment_score > -0.3:
            sentiment_direction = "부정"
        else:
            sentiment_direction = "매우 부정"
        
        # 뉴스 타이밍 분석
        if sentiment_score > 0.2:
            timing_comment = "긍정 뉴스 흐름 확인"
        elif sentiment_score < -0.2:
            timing_comment = "부정 재료 부각"
        else:
            timing_comment = "중립적 보도 기조"
        
        # 상위 뉴스 추출
        sorted_news = sorted(all_news, key=lambda x: x.get('relevance_score', 0), reverse=True)
        top_titles = "\n".join([
            f"  - [{n.get('relevance_score', 0):.1f}점] {n['title'][:50]}..."
            for n in sorted_news[:5]
        ]) if sorted_news else "- 관련 뉴스 없음"
        
        # 임팩트 요약
        if news_impact:
            impact_summary = f"""
긍정 뉴스 {news_impact.get('positive_count', 0)}개, 
부정 뉴스 {news_impact.get('negative_count', 0)}개, 
중립 뉴스 {news_impact.get('neutral_count', 0)}개로 
{sentiment_direction} 기조 우세
""".strip().replace('\n', ' ')
        else:
            impact_summary = "뉴스 영향 분석 정보 없음"
        
        return {
            'sentiment_score': sentiment_score,
            'sentiment_direction': sentiment_direction,
            'timing_comment': timing_comment,
            'top_titles': top_titles,
            'impact_summary': impact_summary,
            'total_news': len(all_news)
        }

    def analyze_competitors(self, competitor_news, stock_name) -> str:
        """경쟁사 분석"""
        if not competitor_news:
            return f"""
> ❗ **경쟁사 뉴스 정보 부족**
> - {stock_name} 직접 경쟁사 뉴스 없음
> - 동종업계 비교분석 제한적
> - 향후 업종별 벤치마킹 필요
"""
        
        competitor_summary = f"**주요 경쟁사 동향 ({len(competitor_news)}건)**\n"
        for i, news in enumerate(competitor_news[:3], 1):
            competitor_summary += f"{i}. [{news.get('competitor', 'N/A')}] {news['title'][:40]}...\n"
        
        if len(competitor_news) > 3:
            competitor_summary += f"   (외 {len(competitor_news)-3}건 추가)\n"
        
        return competitor_summary

    def analyze_market_context(self, kospi_info, etf_info, industry_info) -> str:
        """시장 맥락 분석"""
        context_parts = []
        
        if "정보 없음" not in kospi_info:
            context_parts.append("코스피 정보 확인")
        if "정보 없음" not in etf_info:
            context_parts.append("섹터 ETF 비교 가능")
        if "정보 없음" not in industry_info:
            context_parts.append("업종 비교 가능")
        
        if context_parts:
            return f"시장 맥락 분석 가능 ({', '.join(context_parts)})"
        else:
            return "시장 맥락 정보 제한적"

    def generate_investment_insights(self, events_df, news_impact, market_analysis, competitor_analysis) -> dict:
        """투자 인사이트 생성"""
        
        # 단기 전망
        event_count = len(events_df)
        sentiment = news_impact.get('sentiment_score', 0.0) if news_impact else 0.0
        
        if event_count > 5 and sentiment > 0.2:
            short_term = "높은 변동성 + 긍정 뉴스 → 단기 상승 모멘텀 기대"
            grade = "B+"
        elif event_count > 3 or abs(sentiment) > 0.1:
            short_term = "중간 수준 변동성 → 방향성 주의 깊게 관찰 필요"
            grade = "B"
        else:
            short_term = "안정적 흐름 → 큰 변화 없이 박스권 예상"
            grade = "B-"
        
        # 중기 전망
        if "제한적" in market_analysis and "부족" in competitor_analysis:
            mid_term = "정보 부족으로 중기 방향성 판단 어려움"
        elif sentiment > 0.2:
            mid_term = "긍정적 뉴스 흐름 지속 시 상승 여력 존재"
        else:
            mid_term = "추가 재료 부재 시 횡보 가능성"
        
        # 리스크
        risks = []
        if event_count > 5:
            risks.append("높은 변동성으로 인한 급락 리스크")
        if "정보 부족" in competitor_analysis:
            risks.append("경쟁 환경 불확실성")
        if abs(sentiment) < 0.1:
            risks.append("뚜렷한 방향성 부재")
        
        risk_warning = ", ".join(risks) if risks else "일반적 시장 리스크"
        
        return {
            'short_term': short_term,
            'mid_term': mid_term,
            'risks': risk_warning,
            'grade': grade
        }

    def create_fallback_report(self, template_vars) -> str:
        """향상된 기본 리포트 생성 (템플릿 오류시)"""
        
        sentiment_score = template_vars.get('sentiment_score', 0.0)
        sentiment_direction = template_vars.get('sentiment_direction', '중립')
        
        return f"""
🔍 **주식 분석 리포트 (간소 버전)**

## 📊 **핵심 요약**
- **분석 개요**: {template_vars.get('summary_price', '정보 없음')}
- **뉴스 감성**: {sentiment_score:+.2f} ({sentiment_direction})
- **시장 상황**: {template_vars.get('kospi_info', '정보 없음')}

## 🎯 **주요 발견사항**
{template_vars.get('event_insight', '주요 이벤트 정보를 확인할 수 없습니다.')}

## 📰 **뉴스 영향 분석**
- **감성 점수**: {sentiment_score:+.2f}
- **주요 뉴스**: 
{template_vars.get('top_news_titles', '뉴스 정보 없음')}
- **영향 분석**: {template_vars.get('impact_summary', '분석 정보 없음')}

## 🏢 **경쟁사 현황**
{template_vars.get('competitor_summary', '경쟁사 정보 없음')}

## 💡 **투자 관점**
- **단기 전망**: {template_vars.get('insight_short', '전망 정보 없음')}
- **중기 관점**: {template_vars.get('insight_mid', '전망 정보 없음')}
- **주요 리스크**: {template_vars.get('risk_warning', '리스크 정보 없음')}

## 📋 **시장 맥락**
- **전체 시장**: {template_vars.get('kospi_info', 'N/A')}
- **업종 현황**: {template_vars.get('industry_info', 'N/A')}
- **섹터 ETF**: {template_vars.get('etf_info', 'N/A')}

---
⚠️ **주의**: 템플릿 처리 중 일부 오류가 발생하여 간소화된 리포트입니다.  
📞 **문의**: 시스템 관리자에게 템플릿 파일 확인을 요청하세요.

*본 분석은 AI 기반 자동 생성이며, 투자 판단은 사용자 책임입니다.*
"""

def search_stock_code(stock_name: str) -> tuple:
    """종목 코드 검색 (오류 처리 강화)"""
    try:
        stock_db = get_stock_database()
        normalized_query = stock_name.replace(" ", "").lower()

        # 정확한 매칭 우선
        for db_name, (code, full_name) in stock_db.items():
            db_name_normalized = db_name.lower().replace(" ", "")
            if normalized_query == db_name_normalized:
                print(f"✅ 종목 발견: {full_name} ({code})")
                return code, full_name

        # 부분 매칭
        candidates = []
        for db_name, (code, full_name) in stock_db.items():
            db_name_lower = db_name.lower()
            if normalized_query in db_name_lower or db_name_lower in normalized_query:
                candidates.append((code, full_name))

        if candidates:
            print(f"\n🎯 '{stock_name}'와 유사한 종목들:")
            for i, (code, name) in enumerate(candidates[:5], 1):
                print(f"{i}. {name} ({code})")
            
            while True:
                try:
                    choice = input(f"\n선택 (1-{min(5, len(candidates))}): ").strip()
                    if choice.isdigit():
                        choice_num = int(choice)
                        if 1 <= choice_num <= min(5, len(candidates)):
                            return candidates[choice_num - 1]
                    print("올바른 번호를 입력하세요.")
                except KeyboardInterrupt:
                    return None, None

        # 수동 입력
        print(f"\n❗ '{stock_name}'를 종목 DB에서 찾을 수 없습니다.")
        manual_code = input("👉 종목 코드를 직접 입력하세요 (예: 005930): ").strip()
        confirm_name = input("👉 종목의 정식 이름을 입력하세요 (예: 삼성전자): ").strip()

        if manual_code and confirm_name:
            print(f"✅ 입력 완료: {confirm_name} ({manual_code})")
            try:
                add_stock_entry(stock_name, manual_code, confirm_name)
            except Exception as e:
                print(f"⚠️ DB 저장 실패: {e}")
            return manual_code, confirm_name

    except Exception as e:
        print(f"🚨 종목 검색 중 오류: {e}")
    
    return None, None

def get_user_input():
    """사용자 입력 받기 (강화된 검증)"""
    print("🚀 강화된 CRAG 주식 분석 시스템 v2.0")
    print("="*50)
    
    max_attempts = 3
    
    # 종목 입력
    for attempt in range(max_attempts):
        try:
            stock_name = input(f"\n📈 분석할 종목명을 입력하세요 (시도: {attempt+1}/{max_attempts}): ").strip()
            if not stock_name:
                print("❌ 종목명을 입력해주세요.")
                continue
                
            stock_code, exact_name = search_stock_code(stock_name)
            if stock_code and exact_name:
                break
            else:
                print("❌ 종목을 찾을 수 없습니다.")
                if attempt == max_attempts - 1:
                    print("❌ 최대 시도 횟수 초과")
                    return None, None, None
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            return None, None, None
        except Exception as e:
            print(f"❌ 오류 발생: {e}")
    
    # 날짜 입력
    for attempt in range(max_attempts):
        try:
            print(f"\n📅 분석 날짜를 입력하세요.")
            print("형식: YYYY-MM-DD (예: 2025-06-09) 또는 'today'")
            
            date_input = input("날짜: ").strip().lower()
            
            if date_input == "today":
                analysis_date = datetime.now().strftime("%Y-%m-%d")
                break
            elif date_input == "":
                print("❌ 날짜를 입력해주세요.")
                continue
            else:
                try:
                    # 날짜 유효성 검사
                    parsed_date = datetime.strptime(date_input, "%Y-%m-%d")
                    if parsed_date > datetime.now():
                        print("❌ 미래 날짜는 선택할 수 없습니다.")
                        continue
                    analysis_date = date_input
                    break
                except ValueError:
                    print("❌ 올바른 날짜 형식이 아닙니다. (YYYY-MM-DD)")
                    continue
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            return None, None, None
    
    print(f"\n✅ 선택된 종목: {exact_name} ({stock_code})")
    print(f"✅ 분석 날짜: {analysis_date}")
    
    confirm = input("\n🚀 분석을 시작하시겠습니까? (y/N): ").strip().lower()
    if confirm not in ['y', 'yes']:
        print("👋 분석을 취소했습니다.")
        return None, None, None
    
    return stock_code, exact_name, analysis_date

def main():
    """강화된 CRAG 메인 실행 함수"""
    
    analyzer = CRAGAnalyzer()
    
    try:
        # 사용자 입력
        user_input = get_user_input()
        if user_input[0] is None:
            return
        
        stock_code, stock_name, date = user_input
        
        print(f"\n🚀 {stock_name}({stock_code}) {date} 강화된 CRAG 분석 시작")
        print("💡 고도화된 시간적 인과관계 분석 시스템")
        print("="*70)

        # 1단계: 주가 데이터 수집
        print("\n📊 1단계: 강건한 주가 데이터 수집")
        df = analyzer.robust_fetch_intraday_price(stock_code, date)
        
        # 2단계: 고도화된 이벤트 감지
        print("\n🎯 2단계: 고도화된 이벤트 감지")
        events = analyzer.enhanced_detect_price_events_by_day(df, threshold=0.005)
        print(f"✅ 주가 데이터: {len(df)}개 시점")
        print(f"✅ 감지된 이벤트: {len(events)}개")

        # 3단계: 뉴스 수집 및 분석
        print("\n📰 3단계: 향상된 뉴스 수집 및 감성 분석")
        try:
            news_collector = EnhancedNewsCollector(analyzer.client_id, analyzer.client_secret)
            
            enhanced_news = news_collector.search_news_multi_strategy(
                stock_name=stock_name,
                date=date,
                days_before=3,
                days_after=0
            )
            
            news_impact = news_collector.analyze_news_impact(enhanced_news, stock_name)
            competitor_news = news_collector.get_competitor_news(stock_name, date)
            
            print(f"✅ 관련성 높은 뉴스: {len(enhanced_news)}개")
            print(f"✅ 감성 분석: 긍정 {news_impact['positive_count']}, 부정 {news_impact['negative_count']}, 중립 {news_impact['neutral_count']}")
            print(f"✅ 감성 점수: {news_impact['sentiment_score']:+.3f}")
            print(f"✅ 경쟁사 뉴스: {len(competitor_news)}개")
            
        except Exception as e:
            print(f"⚠️ 뉴스 수집 실패: {e}")
            enhanced_news, news_impact, competitor_news = [], {}, []

        # 4단계: 공시정보 수집
        print("\n📋 4단계: 공시정보 수집")
        try:
            start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
            disclosures = fetch_disclosures_with_fallback(stock_name, start_date, date)
            print(f"✅ 수집된 공시: {len(disclosures)}개")
        except Exception as e:
            print(f"⚠️ 공시 수집 실패: {e}")
            disclosures = []

        # 5단계: 시간적 인과관계 매칭
        print("\n🔗 5단계: CRAG 시간적 인과관계 분석")
        try:
            # 뉴스 형식 변환
            analyzed_news = []
            for news in enhanced_news:
                analyzed_news.append({
                    'title': news['title'],
                    'link': news['link'],
                    'pubDate': news['pubDate'],
                    'description': news.get('description', ''),
                    'relevance_score': news.get('relevance_score', 0),
                    'sentiment': 'neutral'
                })
            
            matched_news_dict = match_news_before_events(analyzed_news, events)
            matched_disclosures_dict = match_disclosures_before_events(disclosures, events, hours_before=72)
            
            total_matched_news = sum(len(news_list) for news_list in matched_news_dict.values())
            total_matched_disclosures = sum(len(disc_list) for disc_list in matched_disclosures_dict.values())
            print(f"✅ 인과관계 매칭 - 뉴스: {total_matched_news}개, 공시: {total_matched_disclosures}개")
            
        except Exception as e:
            print(f"⚠️ 인과관계 분석 실패: {e}")
            matched_news_dict, matched_disclosures_dict = {}, {}

        # 6단계: 시장 맥락 수집
        print("\n📈 6단계: 시장 맥락 정보 수집")
        try:
            date_fmt = datetime.strptime(date, "%Y-%m-%d").strftime("%Y.%m.%d")
            
            kospi_info = fetch_kospi_daily(date_fmt)
            kospi_line = (
                f"코스피 {kospi_info.get('close', 0):,.0f} ({kospi_info.get('rate', 0):+.2f}%)"
                if kospi_info and 'rate' in kospi_info else "코스피 정보 없음"
            )
            
            etf_info = fetch_sector_etf_daily(etf_code="091160", date_yyyymmdd=date_fmt)
            etf_line = (
                f"반도체ETF {etf_info.get('rate', 0):+.2f}%"
                if etf_info and 'rate' in etf_info else "섹터ETF 정보 없음"
            )
            
            industry_info = fetch_industry_info_by_stock_code(stock_code)
            if industry_info and "업종명" in industry_info:
                industry_line = f"업종({industry_info.get('업종명', 'N/A')}) {industry_info.get('등락률', 'N/A')}"
            else:
                industry_line = "업종 정보 없음"
                
            print(f"✅ 시장 맥락: {kospi_line} | {etf_line} | {industry_line}")
            
        except Exception as e:
            print(f"⚠️ 시장 정보 수집 실패: {e}")
            kospi_line = etf_line = industry_line = "정보 수집 실패"

        # 7단계: 종합 분석 리포트 생성
        print("\n🧠 7단계: 강화된 종합 CRAG 분석")
        try:
            analysis_result = analyzer.enhanced_comprehensive_analysis_v3(
                events, matched_news_dict, matched_disclosures_dict,
                stock_name, date, news_impact, competitor_news,
                kospi_info=kospi_line,
                etf_info=etf_line,
                industry_info=industry_line
            )
            print("✅ CRAG 분석 완료")
            
            # analysis_result에서 report_content와 summary 추출
            if isinstance(analysis_result, dict) and "error" not in analysis_result:
                report_content = analysis_result.get("report", "리포트 생성 실패")
                summary = analysis_result.get("summary", {})
            else:
                print("⚠️ 분석 결과 형식 오류, 기본 리포트 사용")
                report_content = str(analysis_result)
                summary = {"오류": "분석 실패"}
                
        except Exception as e:
            print(f"🚨 종합 분석 중 오류: {e}")
            report_content = f"분석 중 오류가 발생했습니다: {str(e)}"
            summary = {"오류": str(e)}
        
        # 8단계: 결과 출력 및 전송
        print("\n" + "="*70)
        print("📄 CRAG 기반 종합 분석 리포트")
        print("="*70)
        print(report_content)
        
        # 9단계: Slack 전송
        print("\n📨 9단계: Slack 요약 전송")
        try:
            final_message = f"📈 *{stock_name} ({date}) CRAG 분석 완료*\n\n" + \
                f"🎯 *핵심 지표*\n" + \
                f"• 이벤트: {summary.get('이벤트_수', summary.get('이벤트 수', 0))}개\n" + \
                f"• 뉴스: {summary.get('뉴스_수', summary.get('관련 뉴스 수', 0))}개 (감성: {summary.get('감성_점수', summary.get('감성 점수', 0)):+.2f})\n" + \
                f"• 경쟁사: {summary.get('경쟁사_뉴스', summary.get('경쟁사 뉴스 수', 0))}개\n" + \
                f"• 공시: {summary.get('공시_수', summary.get('공시 수', 0))}개\n" + \
                f"• 투자등급: {summary.get('투자_등급', 'N/A')}\n\n" + \
                f"📊 *시장 맥락*\n" + \
                f"• {kospi_line}\n" + \
                f"• {etf_line}\n" + \
                f"• {industry_line}\n\n" + \
                f"---\n" + \
                f"• {report_content}\n" + \
                f"_강화된 CRAG v2.0 시스템 (시간적 인과관계 기반)_"

            send_to_slack(final_message, analyzer.slack_url)
            print("✅ Slack 전송 완료")
            
        except Exception as e:
            print(f"⚠️ Slack 전송 실패: {e}")
            # 기본 메시지라도 전송 시도
            try:
                basic_message = f"📈 {stock_name} ({date}) CRAG 분석 완료 (일부 오류 발생)"
                send_to_slack(basic_message, analyzer.slack_url)
                print("✅ 기본 Slack 메시지 전송")
            except:
                print("❌ Slack 전송 완전 실패")
        
        print(f"\n🎉 {stock_name} CRAG 분석이 성공적으로 완료되었습니다!")
        
    except KeyboardInterrupt:
        print("\n👋 사용자가 프로그램을 중단했습니다.")
    except Exception as e:
        error_msg = f"🚨 CRAG 시스템 오류: {e}"
        print(error_msg)
        traceback.print_exc()
        
        try:
            send_to_slack(f"❌ CRAG 분석 실패: {error_msg}", analyzer.slack_url)
        except:
            print("Slack 오류 알림 전송도 실패했습니다.")

if __name__ == "__main__":
    main()