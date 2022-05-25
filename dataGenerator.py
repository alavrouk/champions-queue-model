from bs4 import BeautifulSoup
from selenium import webdriver
import time
import numpy as np
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By

# Scrapes from championsqueue.gg

# Gets the latest 20(numClicks + 1) Matches
# Then gets the champions in those games and patches
# Champions is [Team 1, Team 2]
def getChampionsAndPatchesAndPlayers(URL, numClicks):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(executable_path='chromedriver.exe', options=chrome_options)
    driver.get('{}?qty={}'.format(URL, 1346))
    time.sleep(2)
    button = driver.find_element(by=By.CLASS_NAME, value="block")
    for i in range(numClicks):
        button.click()
        time.sleep(2)
    html = driver.page_source
    driver.quit()
    page_content = BeautifulSoup(html, 'lxml')

    champions = []
    patches = []
    players = []
    for player in page_content.find('ol', class_='list').find_all('span', class_='player-name svelte-e4g8hu'):
        players.append(player)
    for champ in page_content.find('ol', class_='list').find_all('img', class_='svelte-j5wrz'):
        champions.append(champ['alt'])
    for patch in page_content.find('ol', class_='list').find_all('span', class_='stat patch svelte-e4g8hu'):
        patches.append(patch)
    players = np.asarray(players)
    players = players.reshape((players.shape[0] // 10, 10))
    champions = np.asarray(champions)
    champions = champions.reshape((champions.shape[0] // 10, 10))
    patches = np.asarray(patches)
    patches = patches.reshape((patches.shape[0], 1))
    return champions, patches, players


# WinLoss on this website is really terrible, you have to actually click on the match
# So I do that here
def getWinLoss(URL, numClicks):
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(executable_path='chromedriver.exe', options=chrome_options)
    driver.get('{}?qty={}'.format(URL, 1346))
    time.sleep(2)

    button = driver.find_element(by=By.CLASS_NAME, value="block")
    for i in range(numClicks):
        button.click()
        time.sleep(2)
    time.sleep(5)
    matchList = driver.find_element(by=By.CLASS_NAME, value='list')
    matches = matchList.find_elements_by_tag_name("li")

    winLosses = []
    for j in range(len(matches)):
        matches[j].click()
        time.sleep(2)
        html = driver.page_source
        soup = BeautifulSoup(html, 'lxml')
        outcomes = soup.find_all('h3', class_='outcome svelte-pgplua')
        winLosses.append(outcomes[0].getText())
        el = driver.find_element(by=By.XPATH, value="//div[@class=\'backdrop svelte-pgplua\']")
        action = ActionChains(driver)
        action.move_to_element_with_offset(el, 100, 100)
        action.click()
        action.perform()
    winLosses = np.asarray(winLosses)
    winLosses.reshape((winLosses.shape[0], 1))
    return winLosses


# Format the data however you please here
def formatData(players, champions, patches, winLosses):
    patches = patches.reshape((patches.shape[0], 1))
    winLosses = winLosses.reshape((winLosses.shape[0], 1))
    playersAndChampions = np.ravel([players, champions], order="F").reshape(np.shape(players)[0], np.shape(players)[1] + np.shape(players)[1])
    data = np.append(patches, winLosses, 1)
    data = np.append(data, playersAndChampions, 1)
    return data


# Puts it all together
def generateData(URL, numClicks):
    champions, patches, players = getChampionsAndPatchesAndPlayers(URL, numClicks)
    winLosses = getWinLoss(URL, numClicks)
    data = formatData(players, champions, patches, winLosses)
    print(data)
    np.savetxt(fname="champions_queue_data.csv", X=data, delimiter=",", fmt="%s")
    return data


# Main function, here I run generateData and turn it into a csv for later
# if __name__ == '__main__':
#     numClicks = 15
#     if len(sys.argv) > 1:
#         numClicks = np.int_(sys.argv[1])
#     data = generateData("https://championsqueue.gg/matches", numClicks)
#
