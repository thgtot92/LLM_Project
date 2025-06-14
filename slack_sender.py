import requests

def send_to_slack(report_text: str, webhook_url: str):
    payload = {"text": report_text}
    headers = {"Content-Type": "application/json"}
    res = requests.post(webhook_url, json=payload, headers=headers)
    return res.status_code == 200


def send_event_summaries_to_slack(event_summaries, webhook_url):
    for i, summary in enumerate(event_summaries, 1):
        message = f"📍 **이벤트 {i} 요약 리포트** 📍\n\n{summary}"
        success = send_to_slack(message, webhook_url)
        if not success:
            print(f"❌ 이벤트 {i} 전송 실패")
        else:
            print(f"✅ 이벤트 {i} 전송 완료")
