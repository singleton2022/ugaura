import sys
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup

def scrape_cushion_value():
    # 対象URL（※開催場によって index2.html, index3.html に変わる点に注意）
    url = "https://www.jra.go.jp/keiba/baba/index.html"
    
    # ヘッドレスモードの設定
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    
    try:
        driver = webdriver.Chrome(options=options)
    except Exception as e:
        print(f"エラー: WebDriverの初期化に失敗しました。\n詳細: {e}")
        sys.exit(1)

    try:
        driver.get(url)
        # JSによるDOMのレンダリング完了を待機
        time.sleep(2)
        
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # DOM構造の変更に耐えるため、「クッション値」というテキストを含む要素を探索
        elements = soup.find_all(string=lambda text: text and 'クッション値' in text)
        
        if not elements:
            print("指定されたページ内に『クッション値』に関する記述が見つかりません。非開催日か、URLが異なる可能性があります。")
            return

        print(f"URL: {url} からの抽出結果:\n")
        
        # 該当テキストを含む周辺のテーブル要素やテキストをごっそり取得
        for el in elements:
            # 親要素のさらに親（trやdiv等）のテキストを取得し、改行や空白を整理
            if el.parent and el.parent.parent:
                context_text = el.parent.parent.get_text(separator=' | ', strip=True)
                print(f"取得データ: {context_text}")
                
    except Exception as e:
         print(f"実行時エラー: {e}")
         
    finally:
        driver.quit()

if __name__ == '__main__':
    scrape_cushion_value()