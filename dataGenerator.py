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

def generateData(URL, numClicks, logger):
    logger.info("Setting up webdriver")
    d0 = time.perf_counter()
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument('--headless')
    driver = webdriver.Chrome(
        executable_path='chromedriver.exe', options=chrome_options)
    driver.get('{}?qty={}'.format(URL, 1346))
    time.sleep(2)
    button = driver.find_element(by=By.CLASS_NAME, value="block")
    for i in range(numClicks):
        button.click()
        time.sleep(2)
    html = driver.page_source
    page_content = BeautifulSoup(html, 'lxml')
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Scraping champion info...")
    d0 = time.perf_counter()
    champions = getChampions(page_content)
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Scraping player info...")
    d0 = time.perf_counter()
    players = getPlayers(page_content)
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Scraping patch info...")
    d0 = time.perf_counter()
    patches = getPatches(page_content)
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    logger.info("Scraping winloss info...")
    d0 = time.perf_counter()
    winLosses = getWinLoss(driver)
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")

    driver.quit()

    # This one is not its own fucntion because I want to add more data, once all data has been added ill make it its own function
    logger.info("Formatting data...")
    d0 = time.perf_counter()
    patches = patches.reshape((patches.shape[0], 1))
    winLosses = winLosses.reshape((winLosses.shape[0], 1))
    playersAndChampions = np.ravel([players, champions], order="F").reshape(np.shape(players)[0],
                                                                            np.shape(players)[1] + np.shape(players)[1])
    data = np.append(patches, winLosses, 1)
    data = np.append(data, playersAndChampions, 1)
    d1 = time.perf_counter()
    logger.info(f"Done in {d1 - d0:0.4f} seconds")
    return data


def getWinLoss(driver):
    time.sleep(2)
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
        el = driver.find_element(
            by=By.XPATH, value="//div[@class=\'backdrop svelte-pgplua\']")
        action = ActionChains(driver)
        action.move_to_element_with_offset(el, 100, 100)
        action.click()
        action.perform()
    winLosses = np.asarray(winLosses)
    winLosses.reshape((winLosses.shape[0], 1))
    return winLosses


def getChampions(page_content):
    champions = []
    for champ in page_content.find('ol', class_='list').find_all('img', class_='svelte-j5wrz'):
        champions.append(champ['alt'])
    champions = np.asarray(champions)
    champions = champions.reshape((champions.shape[0] // 10, 10))
    return champions


def getPlayers(page_content):
    players = []
    for player in page_content.find('ol', class_='list').find_all('span', class_='player-name svelte-e4g8hu'):
        players.append(player)
    # Remove player team from their name (so team swaps do not affect data)
    players = np.asarray(players)
    for i in range(len(players)):
        spliced = players[i][0].split(' ')
        if len(spliced) > 1:
            del spliced[0]
            s = " "
            s = s.join(spliced)
            players[i][0] = s
    players = players.reshape((players.shape[0] // 10, 10))
    return players


def getPatches(page_content):
    patches = []
    for patch in page_content.find('ol', class_='list').find_all('span', class_='stat patch svelte-e4g8hu'):
        patches.append(patch)
    patches = np.asarray(patches)
    patches = patches.reshape((patches.shape[0], 1))
    return patches
