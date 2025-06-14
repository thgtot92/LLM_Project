import google.generativeai as genai
import os
import time
from google.api_core.exceptions import ResourceExhausted

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    print("⚠️ groq 패키지를 찾을 수 없습니다.")
    print("LLaMA 모델을 사용하려면: pip install groq")
    GROQ_AVAILABLE = False
    Groq = None

from typing import Literal

# API 키 설정
genai.configure(api_key=os.getenv("GOOGLE_API_KEY") or "AIzaSyC-2aYPlx7he6pwCGmhvYYenvYGluEZApA")

# Groq 클라이언트 초기화 (LLaMA용)
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or "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"

if GROQ_AVAILABLE:
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"✅ Groq 클라이언트 초기화 성공 (키: {GROQ_API_KEY[:20]}...)")
    except Exception as e:
        print(f"❌ Groq 클라이언트 초기화 실패: {e}")
        groq_client = None
        GROQ_AVAILABLE = False
else:
    groq_client = None

def get_llm_report(prompt: str, max_retries=2, retry_delay=60, model_type: Literal["gemini", "llama"] = "gemini"):
    """
    Rate Limit을 고려한 안전한 LLM 리포트 생성 (Gemini/LLaMA 선택 가능)
    
    Args:
        prompt: LLM에 전달할 프롬프트
        max_retries: 최대 재시도 횟수 (기본 2회)
        retry_delay: 재시도 대기 시간(초, 기본 60초)
        model_type: 사용할 모델 ("gemini" 또는 "llama")
    
    Returns:
        str: LLM 응답 또는 오류 메시지
    """
    if model_type == "llama":
        return get_llama_report(prompt, max_retries, retry_delay)
    else:
        return get_gemini_report(prompt, max_retries, retry_delay)

def reinitialize_groq_client(api_key=None):
    """Groq 클라이언트를 재초기화하는 함수"""
    global groq_client, GROQ_API_KEY, GROQ_AVAILABLE
    
    if api_key:
        GROQ_API_KEY = api_key
        os.environ["GROQ_API_KEY"] = api_key
    else:
        # 환경변수에서 읽기
        GROQ_API_KEY = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
    
    if not GROQ_AVAILABLE:
        try:
            from groq import Groq
            GROQ_AVAILABLE = True
        except ImportError:
            print("⚠️ Groq 패키지가 설치되지 않았습니다.")
            return False
    
    try:
        groq_client = Groq(api_key=GROQ_API_KEY)
        print(f"✅ Groq 클라이언트 재초기화 성공 (키: {GROQ_API_KEY[:20]}...)")
        return True
    except Exception as e:
        print(f"❌ Groq 클라이언트 재초기화 실패: {e}")
        groq_client = None
        return False

def get_gemini_report(prompt: str, max_retries=2, retry_delay=60):
    """기존 Gemini 모델 사용"""
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"⏳ API 요청 한도로 인해 {retry_delay}초 대기 중... (재시도 {attempt}/{max_retries})")
                time.sleep(retry_delay)
            
            print(f"🧠 Gemini LLM 분석 진행 중... (시도 {attempt + 1}/{max_retries + 1})")
            response = model.generate_content(prompt)
            
            if response and response.text:
                print("✅ Gemini 분석 완료")
                return response.text
            else:
                print("⚠️ Gemini 응답이 비어있음")
                return "[분석 실패] Gemini 응답을 받을 수 없습니다."
            
        except ResourceExhausted as e:
            print(f"⚠️ Gemini API 요청 한도 초과 (시도 {attempt + 1})")
            print("💡 Gemini Free Tier는 분당 15회 요청 제한이 있습니다.")
            
            if attempt >= max_retries:
                return create_fallback_analysis(prompt)
                
        except Exception as e:
            print(f"❌ Gemini 호출 오류: {str(e)[:100]}...")
            
            if attempt >= max_retries:
                return create_fallback_analysis(prompt)
            else:
                print(f"🔄 10초 후 재시도...")
                time.sleep(10)
    
    return create_fallback_analysis(prompt)

def get_llama_report(prompt: str, max_retries=2, retry_delay=30, debug=False):
    """
    LLaMA 모델을 사용한 리포트 생성 (Groq API 사용)
    
    Args:
        prompt: LLM에 전달할 프롬프트
        max_retries: 최대 재시도 횟수
        retry_delay: 재시도 대기 시간(초)
        debug: 디버그 모드 (상세 정보 출력)
    
    Returns:
        str: LLaMA 응답 또는 오류 메시지
    """
    global groq_client, GROQ_API_KEY
    
    # Groq 사용 가능 여부 확인
    if not GROQ_AVAILABLE:
        print("❌ Groq 패키지가 설치되지 않았습니다.")
        print("해결 방법: pip install groq")
        return create_fallback_analysis(prompt)
    
    # groq_client가 None이면 재초기화 시도
    if not groq_client:
        print("⚠️ Groq 클라이언트가 초기화되지 않았습니다. 재초기화 시도 중...")
        # 환경변수에서 API 키 다시 읽기
        new_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)
        if reinitialize_groq_client(new_key):
            print("✅ Groq 클라이언트 재초기화 성공")
        else:
            print("❌ Groq 클라이언트 재초기화 실패")
            return create_fallback_analysis(prompt)
    
    if debug:
        print(f"🔍 Debug: Groq API Key = {GROQ_API_KEY[:20]}...")
        print(f"🔍 Debug: Groq Client = {groq_client}")
        print(f"🔍 Debug: Environment GROQ_API_KEY = {os.getenv('GROQ_API_KEY', 'Not set')[:20]}...")
    
    # 사용 가능한 LLaMA 모델들
    models = [
        "llama3-8b-8192",           # 가장 안정적인 모델
        "llama-3.1-8b-instant",     # 빠른 응답
        "llama3-70b-8192",          # 대형 모델
        "llama-3.1-70b-versatile"   # 최신 대형 모델
    ]
    
    selected_model = models[0]  # 기본적으로 가장 안정적인 모델 사용
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                print(f"⏳ 재시도 대기 중... {retry_delay}초 (시도 {attempt}/{max_retries})")
                time.sleep(retry_delay)
                # 재시도 시 더 가벼운 모델로 전환
                if attempt < len(models):
                    selected_model = models[attempt]
                    print(f"🔄 모델 변경: {selected_model}")
            
            print(f"🦙 LLaMA ({selected_model}) 분석 진행 중... (시도 {attempt + 1}/{max_retries + 1})")
            
            if debug:
                print(f"🔍 Debug: 모델 = {selected_model}")
                print(f"🔍 Debug: 프롬프트 길이 = {len(prompt)} 문자")
            
            # Groq API 호출
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 한국 주식시장 분석 전문가입니다. 정확하고 통찰력 있는 분석을 제공합니다."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=selected_model,
                temperature=0.7,
                max_tokens=2048,
                top_p=0.9,
                stream=False
            )
            
            response_text = chat_completion.choices[0].message.content
            
            if response_text:
                print(f"✅ LLaMA ({selected_model}) 분석 완료")
                return response_text
            else:
                print("⚠️ LLaMA 응답이 비어있음")
                return "[분석 실패] LLaMA 응답을 받을 수 없습니다."
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ LLaMA 호출 오류: {error_msg[:200]}...")
            
            # 자세한 에러 진단
            if "rate_limit_exceeded" in error_msg.lower():
                print("⚠️ Groq API 요청 한도 초과")
                print("- 무료 플랜: 분당 30 요청, 일일 14,400 토큰")
                if attempt < max_retries:
                    print(f"🔄 {retry_delay}초 후 재시도...")
                    continue
            
            # API 키 오류
            elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower() or "401" in error_msg:
                print("❌ Groq API 키 인증 오류!")
                print(f"현재 설정된 키: {GROQ_API_KEY[:20]}...")
                print("\n해결 방법:")
                print("1. 제공된 API 키가 만료되었을 수 있습니다")
                print("2. https://console.groq.com 에서 새 계정 생성")
                print("3. API Keys 메뉴에서 새 키 발급")
                print("4. 코드에서 직접 수정하거나 환경변수 설정:")
                print("   export GROQ_API_KEY='your-new-key'")
                return create_fallback_analysis(prompt)
            
            # 모델 오류
            elif "model" in error_msg.lower() or "not found" in error_msg.lower():
                print(f"❌ 모델 오류: {selected_model}")
                print("사용 가능한 모델: llama3-8b-8192, llama-3.1-8b-instant")
                if attempt < max_retries and attempt < len(models):
                    selected_model = models[attempt + 1]
                    print(f"🔄 다른 모델로 재시도: {selected_model}")
                    continue
            
            # 기타 오류
            else:
                print("❌ 알 수 없는 오류 발생")
                print("전체 오류 메시지:")
                print(error_msg)
            
            if attempt >= max_retries:
                return create_fallback_analysis(prompt)
    
    return create_fallback_analysis(prompt)

def compare_llm_models(prompt: str):
    """
    Gemini와 LLaMA 모델의 응답을 비교
    
    Args:
        prompt: 두 모델에 전달할 프롬프트
    
    Returns:
        dict: 두 모델의 응답과 메타데이터
    """
    print("🔬 LLM 모델 비교 분석 시작")
    print("="*50)
    
    # Gemini 분석
    print("\n1️⃣ Gemini 분석")
    start_time = time.time()
    gemini_response = get_llm_report(prompt, model_type="gemini", max_retries=1)
    gemini_time = time.time() - start_time
    
    # 잠시 대기 (API 제한 방지)
    print("\n⏳ 모델 전환 대기 (5초)...")
    time.sleep(5)
    
    # LLaMA 분석
    print("\n2️⃣ LLaMA 분석")
    start_time = time.time()
    llama_response = get_llm_report(prompt, model_type="llama", max_retries=1)
    llama_time = time.time() - start_time
    
    print("\n✅ 모델 비교 완료")
    print(f"⏱️ Gemini 응답 시간: {gemini_time:.2f}초")
    print(f"⏱️ LLaMA 응답 시간: {llama_time:.2f}초")
    
    return {
        "gemini": {
            "response": gemini_response,
            "time": gemini_time,
            "model": "gemini-1.5-flash"
        },
        "llama": {
            "response": llama_response,
            "time": llama_time,
            "model": "llama-3.1-70b-versatile"
        }
    }

def create_fallback_analysis(prompt: str) -> str:
    """
    LLM 호출 실패 시 대체 분석 생성
    프롬프트 내용을 바탕으로 기본적인 분석 제공
    """
    
    # 프롬프트에서 핵심 정보 추출
    lines = prompt.split('\n')
    
    # 종목명, 날짜, 이벤트 정보 추출
    stock_info = ""
    event_info = ""
    news_info = ""
    disclosure_info = ""
    
    for line in lines:
        if "이벤트:" in line:
            event_info = line.strip()
        elif "날짜:" in line or "시간:" in line:
            stock_info += line.strip() + " "
        elif "뉴스" in line and ":" in line:
            news_info = "뉴스 정보 포함"
        elif "공시" in line and ":" in line:
            disclosure_info = "공시 정보 포함"
    
    fallback_analysis = f"""[대체 분석 리포트]

{stock_info.strip()}

📊 **분석 상황:**
- LLM API 요청 한도 초과 또는 오류로 인해 대체 분석을 제공합니다.
- 수집된 데이터를 바탕으로 기본적인 분석을 수행했습니다.

📈 **주요 내용:**
{event_info if event_info else "• 주가 변동 이벤트가 감지되었습니다."}

📰 **뉴스 영향:**
• {news_info if news_info else "관련 뉴스가 수집되었습니다."}
• 뉴스 내용과 주가 변동 간의 연관성을 확인할 필요가 있습니다.

📋 **공시 영향:**
• {disclosure_info if disclosure_info else "관련 공시 정보가 수집되었습니다."}
• 공시 내용이 주가에 미친 영향을 분석해볼 필요가 있습니다.

🔍 **종합 평가:**
• 수집된 데이터를 바탕으로 추후 상세 분석이 권장됩니다.
• API 한도 복구 후 재분석을 통해 더 정확한 인사이트를 얻을 수 있습니다.

---
*대체 분석 시스템에 의해 생성된 리포트입니다.*
*정확한 분석을 위해서는 잠시 후 다시 시도해주세요.*
"""
    
    return fallback_analysis


def analyze_events_with_llm(events_df, matched_news_dict, stock_name, model_type="gemini"):
    """
    이벤트별 분석 - Rate Limit 고려 버전
    """
    summaries = []
    
    # 이벤트가 많을 경우 주요 이벤트만 분석 (API 호출 최소화)
    if len(events_df) > 3:
        print(f"⚠️ 이벤트가 {len(events_df)}개로 많아 상위 3개만 분석합니다. (API 한도 고려)")
        events_df = events_df.head(3)
    
    print(f"📊 총 {len(events_df)}개 이벤트 분석 시작 (모델: {model_type})")
    
    for i, row in events_df.iterrows():
        time_obj = row["datetime"]
        price = row["price"]
        event_type = row["event_type"]
        news_list = matched_news_dict.get(time_obj, [])
        news_summary_text = "\n".join([f"- {n['title']}" for n in news_list]) if news_list else "- 관련 뉴스 없음"

        prompt = f"""[이벤트 기반 뉴스 분석 보고서]

📅 날짜: {time_obj.strftime('%Y-%m-%d')}
🕒 시간: {time_obj.strftime('%H:%M')}
📈 이벤트: {stock_name}의 주가가 {event_type} (가격: {price}원)

📰 당시 뉴스 목록:
{news_summary_text}

이 뉴스와 주가 변화의 관련성을 분석하고, 가능한 원인과 시사점을 포함하여 3~5줄로 요약해주세요.
"""
        
        # API 호출 간격 조정 (Rate Limit 방지)
        if i > 0:
            print("⏳ API 요청 간격 조정 중...")
            time.sleep(5)  # 5초 대기
            
        result = get_llm_report(prompt, max_retries=1, retry_delay=30, model_type=model_type)
        summaries.append((time_obj.strftime('%H:%M'), event_type, result))
        print(f"✅ 이벤트 {i+1} 분석 완료")
    
    return summaries


def create_comprehensive_analysis(events_df, matched_news_dict, matched_disclosures_dict, stock_name, date, model_type="gemini"):
    """
    종합 분석 생성 - 모든 정보를 하나의 LLM 호출로 처리 (효율적)
    """
    
    # 이벤트 요약
    event_summary = ""
    if len(events_df) > 0:
        event_summary = "📈 주요 이벤트:\n"
        for _, event in events_df.iterrows():
            pct = event['pct_from_start'] * 100
            event_summary += f"- {event['datetime'].strftime('%H:%M')}: {pct:.2f}% {event['event_type']} (₩{event['price']:,})\n"
    else:
        event_summary = "📈 주요 이벤트:\n- 1% 이상의 주요 변동 이벤트가 감지되지 않았습니다.\n"
    
    # 뉴스 요약
    all_news = []
    for news_list in matched_news_dict.values():
        all_news.extend(news_list)
    
    # 중복 제거
    unique_news = []
    seen_titles = set()
    for news in all_news:
        if news['title'] not in seen_titles:
            seen_titles.add(news['title'])
            unique_news.append(news)
    
    news_summary = ""
    if unique_news:
        news_summary = "📰 관련 뉴스:\n"
        for news in unique_news[:6]:  # 상위 6개
            news_summary += f"- {news['title']}\n"
    else:
        news_summary = "📰 관련 뉴스:\n- 해당 시점과 연관된 뉴스가 없습니다.\n"
    
    # 공시 요약
    all_disclosures = []
    for disc_list in matched_disclosures_dict.values():
        all_disclosures.extend(disc_list)
    
    # 중복 제거
    unique_disclosures = []
    seen_disc_titles = set()
    for disc in all_disclosures:
        if disc['title'] not in seen_disc_titles:
            seen_disc_titles.add(disc['title'])
            unique_disclosures.append(disc)
    
    disclosure_summary = ""
    if unique_disclosures:
        disclosure_summary = "📋 관련 공시:\n"
        for disc in unique_disclosures[:4]:  # 상위 4개
            disclosure_summary += f"- {disc['title']}\n"
    else:
        disclosure_summary = "📋 관련 공시:\n- 해당 기간 관련 공시가 없습니다.\n"
    
    # 통합 프롬프트 생성
    comprehensive_prompt = f"""[{date} {stock_name} CRAG 종합 분석 리포트]

{event_summary}

{news_summary}

{disclosure_summary}

위 정보를 종합하여 다음 관점에서 분석해주세요:

1. **주가 동향 핵심**: 당일 주요 가격 변동과 거래 특징
2. **뉴스 영향 분석**: 뉴스가 주가에 미친 영향과 시장 반응  
3. **공시 영향 분석**: 공시정보가 주가에 미친 영향과 투자자 반응
4. **인과관계 종합**: 뉴스와 공시 중 주요 변동 요인 분석
5. **향후 전망**: 오늘의 흐름이 향후 주가에 미칠 영향

전문적이면서도 명확하고 간결하게 작성해주세요.
"""
    
    print(f"🧠 CRAG 종합 분석 진행 중... (모델: {model_type})")
    return get_llm_report(comprehensive_prompt, max_retries=2, retry_delay=60, model_type=model_type)

def test_groq_connection():
    """Groq API 연결 테스트"""
    print("🧪 Groq API 연결 테스트 시작")
    
    if not GROQ_AVAILABLE:
        print("❌ Groq 패키지가 설치되지 않았습니다!")
        print("설치: pip install groq")
        return False
    
    if not groq_client:
        print("❌ Groq 클라이언트가 초기화되지 않았습니다!")
        return False
    
    print(f"API 키: {GROQ_API_KEY[:20]}...")
    
    try:
        # 간단한 테스트 요청
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "user", "content": "Say 'Hello, I'm working!' in Korean"}
            ],
            model="llama3-8b-8192",  # 가장 안정적인 모델 사용
            max_tokens=50,
            temperature=0.5
        )
        
        result = response.choices[0].message.content
        print(f"✅ Groq API 연결 성공!")
        print(f"응답: {result}")
        return True
        
    except Exception as e:
        print(f"❌ Groq API 연결 실패!")
        print(f"오류: {str(e)}")
        
        error_str = str(e)
        if "authentication" in error_str.lower() or "401" in error_str:
            print("\n🔧 인증 오류 - 해결 방법:")
            print("1. API 키가 올바른지 확인하세요")
            print("2. https://console.groq.com/keys 에서 키 상태 확인")
            print("3. 새 키 발급 후 다시 시도")
            print(f"4. 현재 사용 중인 키: {GROQ_API_KEY}")
            print("   이 키가 유효하지 않을 수 있습니다.")
        elif "rate" in error_str.lower():
            print("\n⚠️ Rate Limit 초과")
            print("- 무료 티어: 분당 30 요청 제한")
            print("- 잠시 후 다시 시도하세요")
        elif "model" in error_str.lower():
            print("\n🔧 모델 오류")
            print("다른 모델을 시도해보세요:")
            print("- llama3-8b-8192")
            print("- llama-3.1-8b-instant")
        
        return False


# # 테스트 코드
# if __name__ == "__main__":
#     print("="*50)
#     print("LLM Reporter 테스트")
#     print("="*50)
    
#     # Groq 상태 확인
#     if not GROQ_AVAILABLE:
#         print("\n⚠️ Groq/LLaMA를 사용할 수 없습니다.")
#         print("Gemini만 테스트합니다.")
#         print("\nLLaMA를 사용하려면:")
#         print("1. pip install groq")
#         print("2. 스크립트 다시 실행")
#     else:
#         print(f"\n✅ Groq API 키 설정됨: {GROQ_API_KEY[:20]}...")
        
#         # API 연결 테스트
#         print("\n" + "="*50)
#         if test_groq_connection():
#             print("\n✅ API 연결 테스트 통과! 모델 비교를 진행합니다.\n")
#         else:
#             print("\n❌ Groq API 연결 실패")
#             print("새 API 키가 필요합니다:")
#             print("1. https://console.groq.com 접속")
#             print("2. 무료 계정 생성 (Google/GitHub 로그인 가능)")
#             print("3. API Keys 메뉴에서 'Create API Key' 클릭")
#             print("4. 생성된 키를 복사")
#             print("5. 코드의 GROQ_API_KEY 부분을 새 키로 교체")
#             print("\n또는 환경변수 설정:")
#             print("export GROQ_API_KEY='your-new-key'")
    
#     # 간단한 테스트
#     print("\n" + "="*50)
#     test_prompt = "삼성전자 주가가 2% 상승했습니다. 이에 대한 간단한 분석을 제공해주세요."
    
#     if GROQ_AVAILABLE and groq_client:
#         print("🧪 모델 비교 테스트")
#         comparison = compare_llm_models(test_prompt)
        
#         print("\n📊 Gemini 응답:")
#         print(comparison["gemini"]["response"][:200] + "...")
        
#         print("\n🦙 LLaMA 응답:")
#         print(comparison["llama"]["response"][:200] + "...")
#     else:
#         print("🧪 Gemini 단독 테스트")
#         response = get_gemini_report(test_prompt)
#         print("\n📊 Gemini 응답:")
#         print(response[:200] + "...")