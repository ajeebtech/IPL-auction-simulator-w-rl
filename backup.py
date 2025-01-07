stads = [
	"IND: I.S. Bindra Punjab Cricket Association Stadium",
	"IND: Sawai Mansingh Stadium, Jaipur",
	"IND: Eden Gardens, Kolkata",
	"IND: Wankhede Stadium, Mumbai",
	"IND: M.Chinnaswamy Stadium, Bengaluru",
	"IND: Narendra Modi Stadium, Motera, Ahmedabad",
	"IND: Arun Jaitley Stadium, Delhi",
	"IND: Rajiv Gandhi International Stadium, Uppal, Hyderabad",
	"IND: Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
	"IND: MA Chidambaram Stadium, Chepauk, Chennai"
]
homes = [
    "PunjabKings",
    "RajasthanRoyals",
    "KolkataKnightRiders",
    "MumbaiIndians",
    "RoyalChallengersBangalore",
    "GujaratTitans",
    "DelhiCapitals",
    "SunrisersHyderabad",
    "LucknowSupergiants",
    "ChennaiSuperKings"
]
import re
import json
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import Select
options = Options()
options.headless = True
import time
data = []
class Player:
    def __init__(self,name,age,role):
        self.name = name
        self.age = age   
        self.role = role
squad = {}
urls = ['https://www.espncricinfo.com/series/ipl-2016-968923/royal-challengers-bangalore-squad-969877/series-squads']
for url in urls:
    rqsts = requests.get(url)
    soup = BeautifulSoup(rqsts.content,'lxml')
    names = soup.find_all('div', class_ = 'ds-flex ds-flex-row ds-items-center ds-justify-between')
    names = [name.text.strip() for name in names]
    names = [name.replace('(c)', '').replace('†', '') for name in names if "Withdrawn" not in name]
    ages = soup.find_all('span', class_ = 'ds-text-compact-xxs ds-font-bold')
    ages = [age.text.strip() for age in ages]
    ages = [int(re.search(r'(\d+)y', age).group(1)) for age in ages if re.search(r'(\d+)y', age)]
    roles = soup.find_all('p',class_ = 'ds-text-tight-s ds-font-regular ds-mb-2 ds-mt-1')
    roles = [role.text.strip() for role in roles]
    batting_styles = soup.find_all('span', )
    for i in range(len(roles)):
        if 'Batter' in roles[i]:
            roles[i] = 'Batter'
        elif 'Allrounder' in roles[i]:
            roles[i] = 'AllRounder'
        elif 'Bowler' in roles[i]:
            roles[i] = 'Bowler'
    removes = len(names) - len(ages)
    n = removes
    del ages[n:]
    del roles[n:]
    for name, age, role in zip(names, ages, roles):
        squad[name] = Player(name=name,age=age,role=role)
def stats_taking(player, i):
    while len(data) <= i:
        data.append({})
    driver = webdriver.Chrome(options=options)
    driver.get('https://stats.espncricinfo.com/ci/engine/stats/index.html')
    driver.maximize_window()
    search_box = driver.find_element(By.NAME, "search")
    search_query = player.name
    search_box.send_keys(search_query)
    search_box.send_keys(Keys.RETURN)
    link = driver.find_element(By.XPATH, "//a[starts-with(text(), 'Players and Officials')]")
    link.click()
    link = driver.find_element(By.XPATH, "//a[text()='Twenty20 matches player']")
    link.click()
    menu_url = driver.current_url
    player_info = driver.find_element(By.XPATH, "//p[@style='padding-bottom:10px']").text
    try:
        details = player_info.split(' - ')[1].strip()
        styles = details.split('; ')  # Split into batting, bowling, and other styles
        print(f"Extracted Details: {styles}")
        for detail in styles:
            if "bat" in detail:  # Detect batting style
                batting_style = detail
                data[i]['bat_style'] = batting_style
                print(f"Batting Style: {batting_style}")
            elif "arm" in detail or "break" in detail:  # Detect bowling style
                bowling_style = detail
                data[i]['bowl_style'] = bowling_style
                print(f"Bowling Style: {bowling_style}")
            elif "wicketkeeper" in detail:  # Detect fielding role
                fielding_style = "wicketkeeper"
                data[i]['field_style'] = fielding_style
                print(f"Fielding Style: {fielding_style}")
    except IndexError:
        pass
    radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='cumulative']")
    radio_button.click()
    radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='awards_match']")
    radio_button.click()
    submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
    submit_button.click()
    try:
        player_of_match_elements = driver.find_elements(By.XPATH, "//td[text()='player of the match']")
        potm_awards = len(player_of_match_elements)
        data[i]['awards'] = int(potm_awards)
        print(f'potm awards {potm_awards}')
        driver.get(menu_url)
    except Exception as e:
        print('No awards found')
        driver.get(menu_url)
    
    radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='results']")
    radio_button.click()
    submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
    submit_button.click()
    
    try:
        wins = driver.find_elements(By.XPATH, "//td[text()='won']")
        losses = driver.find_elements(By.XPATH, "//td[text()='lost']")
        importance = len(wins) / len(losses)
        data[i]['importance'] = importance
        print(f"Importance: {importance}")
        driver.get(menu_url)
    except Exception as e:
        driver.get(menu_url)
        pass
    
    if player.role == 'AllRounder':
        radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='bowling']")
        radio_button.click()
        submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
        submit_button.click()
        row = driver.find_element(By.XPATH, "//tr[@class='data1']")
        cells = row.find_elements(By.TAG_NAME, "td")
        try:
            wickets = int(cells[7].text)
            data[i]['wickets'] = wickets
        except Exception:
            pass
        try:
            bowling_average = float(cells[9].text)
            data[i]['bowling_average'] = bowling_average
        except Exception:
            pass
        try:
            economy = float(cells[10].text)
            data[i]['economy_rate'] = economy
        except Exception:
            pass
        try:
            print(f"Wickets: {wickets}, Bowling Average: {bowling_average}, Economy: {economy}")
        except Exception:
            pass
        driver.get(menu_url)
        radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='batting']")
        radio_button.click()
        submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
        submit_button.click()
        table_row = driver.find_element(By.CLASS_NAME, "data1")
        cells = table_row.find_elements(By.TAG_NAME, "td")
        try:
            runs = int(cells[5].text)
            data[i]['runs'] = runs
        except Exception:
            pass
        try:
            batting_average = float(cells[7].text)
            data[i]['batting_average'] = batting_average
        except Exception:
            pass
        try:
            strike_rate = float(cells[9].text)
            data[i]['strike_rate'] = strike_rate
        except Exception:
            pass
        try:
            fours = int(cells[13].text)
            data[i]['fours'] = fours
        except:
            pass
        try:
            sixes = int(cells[14].text)
            data[i]['sixes'] = sixes
        except:
            pass
    elif player.role == 'Bowler':
        radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='bowling']")
        radio_button.click()
        submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
        submit_button.click()
        row = driver.find_element(By.XPATH, "//tr[@class='data1']")
        cells = row.find_elements(By.TAG_NAME, "td")
        try:
            wickets = int(cells[7].text)
            data[i]['wickets'] = wickets
        except Exception:
            pass
        try:
            bowling_average = float(cells[9].text)
            data[i]['bowling_average'] = bowling_average
        except Exception:
            pass
        try:
            economy = float(cells[10].text)
            data[i]['economy_rate'] = economy
        except Exception:
            pass
        try:
            print(f"Wickets: {wickets}, Bowling Average: {bowling_average}, Economy: {economy}")
        except Exception:
            pass
        driver.get(menu_url)
        for stadium, home in zip(stads, homes):
            dropdown = driver.find_element(By.NAME, "ground")
            try:
                select = Select(dropdown)
                select.select_by_visible_text(stadium)
                radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='bowling']")
                radio_button.click()
                try:
                    second_tbody = driver.find_elements(By.TAG_NAME, "tbody")[1]  
                    row = second_tbody.find_element(By.CLASS_NAME, "data1") 
                    value = row.find_elements(By.TAG_NAME, "td")[11].text 
                    print(f"{home} economy: {value}")
                    data[i][f'{home}_economy'] = float(value)
                except:
                    print(f"{home} economy: N/A")
                    pass
            except:
                print(f"{home} economy: N/A")
                continue
            driver.get(menu_url)
    elif player.role == 'Batter':
        driver.get(menu_url)
        radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='batting']")
        radio_button.click()
        submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
        submit_button.click()
        table_row = driver.find_element(By.CLASS_NAME, "data1")
        cells = table_row.find_elements(By.TAG_NAME, "td")
        try:
            runs = int(cells[5].text)
            data[i]['runs'] = runs
        except Exception:
            pass
        try:
            batting_average = float(cells[7].text)
            data[i]['batting_average'] = batting_average
        except Exception:
            pass
        try:
            strike_rate = float(cells[9].text)
            data[i]['strike_rate'] = strike_rate
        except Exception:
            pass
        try:
            fours = int(cells[13].text)
            data[i]['fours'] = fours
        except:
            pass
        try:
            sixes = int(cells[14].text)
            data[i]['sixes'] = sixes
        except:
            pass
        print(f"Runs: {runs}, Batting Average: {batting_average}, Strike Rate: {strike_rate}")
        driver.get(menu_url)
        for stadium, home in zip(stads, homes):
            dropdown = driver.find_element(By.NAME, "ground")
            select = Select(dropdown)
            try:
                select.select_by_visible_text(stadium)
                radio_button = driver.find_element(By.XPATH, "//input[@type='radio' and @value='batting']")
                radio_button.click()
                submit_button = driver.find_element(By.XPATH, "//input[@type='submit' and @value='Submit query']")
                second_tbody = driver.find_elements(By.TAG_NAME, "tbody")[1]  
                row = second_tbody.find_element(By.CLASS_NAME, "data1")
                value = row.find_elements(By.TAG_NAME, "td")[5].text
                print(f"{home} average: {value}")
                data[i][f'{home}_average'] = value
            except:
                print(f"{home} average: N/A")
                pass
            driver.get(menu_url)
    driver.quit()
for i, player in enumerate(list(squad.values())):
    stats_taking(player, i)

with open("rcb_dataset.json", "w") as f:
    json.dump(data, f, indent=4) 
