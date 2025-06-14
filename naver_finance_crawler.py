import requests
from bs4 import BeautifulSoup
import pandas as pd

def fetch_intraday_price(stock_code: str, date: str):
    df_all = []
    for page in range(1, 41):  # 최대 40페이지 (09:00 ~ 15:30)
        url = f"https://finance.naver.com/item/sise_time.naver?code={stock_code}&thistime={date.replace('-', '')}153000&page={page}"
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.select_one("table.type2")

        if not table:
            continue

        rows = table.find_all('tr')
        data = []
        for row in rows:
            cols = row.find_all('td')
            if len(cols) < 6: continue
            try:
                time = cols[0].text.strip()
                price = int(cols[1].text.strip().replace(',', ''))
                volume = int(cols[5].text.strip().replace(',', ''))
                data.append([time, price, volume])
            except:
                continue
        if data:
            df = pd.DataFrame(data, columns=['time', 'price', 'volume'])
            df_all.append(df)

    if not df_all:
        raise ValueError("시계열 데이터를 수집하지 못했습니다.")

    result = pd.concat(df_all)
    result['datetime'] = pd.to_datetime(date + ' ' + result['time'])
    return result[['datetime', 'price', 'volume']]

def fetch_kospi_daily(date_yyyymmdd: str) -> dict:
    url = "https://finance.naver.com/sise/sise_index_day.naver?code=KOSPI"
    headers = {"User-Agent": "Mozilla/5.0"}

    for page in range(1, 5):
        res = requests.get(f"{url}&page={page}", headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        rows = soup.select("table.type_1 tr")
        for row in rows:
            cols = [td.text.strip().replace(',', '') for td in row.select("td")]
            if len(cols) >= 6 and cols[0] == date_yyyymmdd:
                try:
                    return {
                        "date": cols[0],
                        "close": float(cols[1]),
                        "change": float(cols[2]),
                        "rate": float(cols[3].replace('%', ''))
                    }
                except:
                    continue
    return {}
def fetch_sector_etf_daily(etf_code: str = "091160", date_yyyymmdd: str = "2025.06.09") -> dict:
    url = f"https://finance.naver.com/item/sise_day.naver?code={etf_code}"
    headers = {"User-Agent": "Mozilla/5.0"}

    for page in range(1, 5):
        res = requests.get(f"{url}&page={page}", headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")

        rows = soup.select("table.type2 tr")
        for row in rows:
            cols = [td.text.strip().replace(',', '') for td in row.select("td")]
            if len(cols) >= 6 and cols[0] == date_yyyymmdd:
                try:
                    return {
                        "date": cols[0],
                        "close": float(cols[1]),
                        "change": float(cols[2]),
                        "rate": float(cols[3].replace('%', ''))
                    }
                except:
                    continue
    return {"note": "해당일자 ETF 데이터 없음"}

def fetch_industry_info_by_stock_code(stock_code: str) -> dict:
    url = f"https://finance.naver.com/item/main.naver?code={stock_code}"
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    soup = BeautifulSoup(res.text, "html.parser")

    try:
        sector_tag = soup.select_one("div.wrap_company > a")
        sector_name = sector_tag.text.strip() if sector_tag else "N/A"

        table = soup.select_one("table.per_table")
        rows = table.select("tr") if table else []

        info = {"업종명": sector_name}

        for row in rows:
            cols = row.select("td")
            if len(cols) >= 2:
                label = row.select_one("th").text.strip()
                value = cols[1].text.strip()
                info[label] = value

        return info
    except Exception as e:
        return {"error": str(e)}