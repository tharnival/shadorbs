### IMPORTS ###

import numpy as np
import cv2 as cv
import re
import sys
import os
import json

from collections import deque

import skimage
from skimage.io import imread, imsave
from skimage.transform import hough_line
from skimage.feature import peak_local_max, canny

from PIL import Image
import pytesseract

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_opening, binary_closing, binary_dilation
from scipy.ndimage.morphology import binary_hit_or_miss

from fuzzywuzzy import fuzz, process


### FUNCTIONS ###

#check whether the distance between two colors fall within a given tolerance
def colcmp(c1, c2, tol):
    diff = (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 + (c1[2]-c2[2])**2
    return diff < tol

#search for the blue border in the vertical line where a played card ends up
def blueLine(frame):
    blue = (218,214,124)
    for i in np.arange(400, 470, 2):
        c = frame[i,960]
        diff = (c[0]-blue[0])**2 + (c[1]-blue[1])**2 + (c[2]-blue[2])**2
        if diff < 3000:
            return True
    return False

#look for the white of the slash in the energy display
def energyLine(frame):
    for i in np.arange(190, 200, 2):
        c = frame[890,i]
        if colcmp(c, (220,248,255), 6000):
            return True
    return False

#rotates an image around a specified pivot
def rotateImage(img, angle, pivot):
    pivot = [pivot[1], pivot[0]] #change to row-major order
    padX = [max(0, img.shape[1] - pivot[0]), pivot[0]]
    padY = [max(0, img.shape[0] - pivot[1]), pivot[1]]
    imgP = np.pad(img, [padY, padX, [0,0]], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

#returns a function that sets values below the given threshold to 0
def highpass(x):
    l = lambda c : c if c > x else 0
    return np.vectorize(l)

#remove connected components whose sizes are outside a given range
def compSizeFilter(img, A_min, A_max, connectivity):
    img = img.copy()
    img = img.astype('uint8')*255
    analysis = cv.connectedComponentsWithStats(img, connectivity, cv.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    img = np.zeros(img.shape, dtype='bool')
    for c in range(1, totalLabels):
        area = values[c, cv.CC_STAT_AREA]

        if A_min <= area and area <= A_max:
            componentMask = (label_ids == c)
            img = np.bitwise_or(img, componentMask)

    return img

#remove connected components which touch the right side of the image
def compRemRight(img):
    img = img.copy()
    img = img.astype('uint8')*255
    analysis = cv.connectedComponentsWithStats(img, 4, cv.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis
    right = label_ids[:,-1]
    right = np.unique(right)
    if 0 in right:
        right = right[1:]

    img = np.zeros(img.shape, dtype='bool')
    for c in range(1, totalLabels):

        if not c in right:
            componentMask = (label_ids == c)
            img = np.bitwise_or(img, componentMask)

    return img

#returns a list of the centers of the connected components
def compCentroids(img, connectivity):
    img = img.copy()
    img = img.astype('uint8')*255
    analysis = cv.connectedComponentsWithStats(img, connectivity, cv.CV_32S)
    (totalLabels, label_ids, values, centroid) = analysis

    return centroid[1:]

sqnorm = lambda x : np.inner(x,x)

#takes an image, a set of backgrounds, and an area
# and returns the average color distance between
# the image and each background
def bgdiff(img, bgs, crop):
    y1,y2,x1,x2 = crop
    img = img[y1:y2, x1:x2]
    img = img.astype('float')
    diffs = [0]*len(bgs)
    for i,bg in enumerate(bgs):
        bg = bg[y1:y2, x1:x2]
        diff = img - bg
        diff = np.apply_along_axis(sqnorm, 2, diff)
        roof = lambda x : min(2000, x)
        diff = np.vectorize(roof)(diff)
        diff = np.sum(diff.ravel())
        diffs[i] = diff/(x2-x1)/(y2-y1)
    return diffs

#load the background images
bgs = []
for i in range(8):
    bg = imread('bgs/bg{}.png'.format(i))
    bg = bg[:240, 180:1740]
    bg = cv.cvtColor(bg, cv.COLOR_RGB2BGR)
    bgs.append(bg)

#takes an image of a card header and uses Tesseract to do OCR on the image
def readHeader(crop):
    if crop.shape[0] == 0 or crop.shape[1] == 0:
        return ''

    tb = crop[:,:,0] < 200
    tg = crop[:,:,1] < 200
    tr = crop[:,:,2] < 200
    white = np.logical_or(np.logical_or(tb,tg),tr)
    green = np.logical_or(np.logical_or(np.logical_not(tb),tg),np.logical_not(tr))
    text = np.logical_and(white, green)

    text = np.logical_not(text)

    text = compSizeFilter(text, 0, 170, 4)
    text = compSizeFilter(text, 10, np.inf, 8)

    nowhite = np.logical_and(text, white)
    greentext = np.logical_and(nowhite, np.logical_not(green))
    upgraded = np.sum(greentext) > 225

    text = np.logical_not(text)

    header = Image.fromarray(text)
    str1 = pytesseract.image_to_string(header)
    upgraded = upgraded or str1.find('+') != -1
    str1 = re.sub('[^a-zA-Z. ]', '', str1)
    str1 = re.sub('^\.*', '', str1)
    str1 = str1.strip()
    if upgraded:
        str1 += '+'

    return str1

#takes a string and returns the card name which is the closest match
def read2card(str1, cardNames):
    upgraded = re.match('^.*\+$', str1)
    str1 = re.sub('\+', '', str1)
    l = lambda x : fuzz.ratio(str1, x)
    matches = [l(c) for c in cardNames]
    i = np.argmax(matches)
    ratio = np.max(matches)
    if ratio < 50:
        return ''
    str1 = cardNames[i]
    if upgraded:
        str1 += '+'
    return str1


### CARD NAMES ###

#Ironclad cards
ICards = {'Bash':True,
         'Defend':False,
         'Strike':True,
         'Anger':True,
         'Armaments':False,
         'Body Slam':True,
         'Clash':True,
         'Cleave':False,
         'Clothesline':True,
         'Flex':False,
         'Havoc':False,
         'Headbutt':True,
         'Heavy Blade':True,
         'Iron Wave':True,
         'Perfected Strike':True,
         'Pommel Strike':True,
         'Shrug It Off':False,
         'Sword Boomerang':False,
         'Thunderclap':False,
         'True Grit':False,
         'Twin Strike':True,
         'Warcry':False,
         'Wild Strike':True,
         'Battle Trance':False,
         'Blood for Blood':True,
         'Bloodletting':False,
         'Burning Pact':False,
         'Carnage':True,
         'Combust':False,
         'Dark Embrace':False,
         'Disarm':True,
         'Dropkick':True,
         'Dual Wield':False,
         'Entrench':False,
         'Evolve':False,
         'Feel No Pain':False,
         'Fire Breathing':False,
         'Flame Barrier':False,
         'Ghostly Armor':False,
         'Hemokinesis':True,
         'Infernal Blade':False,
         'Inflame':False,
         'Intimidate':False,
         'Metallicize':False,
         'Power Through':False,
         'Pummel':True,
         'Rage':False,
         'Rampage':True,
         'Reckless Charge':True,
         'Rupture':False,
         'Searing Blow':True,
         'Second Wind':False,
         'Seeing Red':False,
         'Sentinel':False,
         'Sever Soul':True,
         'Shockwave':False,
         'Spot Weakness':True,
         'Uppercut':True,
         'Whirlwind':False,
         'Barricade':False,
         'Berserk':False,
         'Bludgeon':True,
         'Brutality':False,
         'Corruption':False,
         'Demon Form':False,
         'Double Tap':False,
         'Exhume':False,
         'Feed':True,
         'Fiend Fire':False,
         'Immolate':False,
         'Impervious':False,
         'Juggernaut':False,
         'Limit Break':False,
         'Offering':False,
         'Reaper':False
         }

#colorless cards
GCards = {'Bandage Up':False,
         'Blind':True,
         'Dark Shackles':True,
         'Deep Breath':False,
         'Discovery':False,
         'Dramatic Entrance':False,
         'Enlightenment':False,
         'Finesse':False,
         'Flash of Steel':True,
         'Forethought':False,
         'Good Instincts':False,
         'Impatience':False,
         'Jack of All Trades':False,
         'Madness':False,
         'Mind Blast':True,
         'Panacea':False,
         'Panic Button':False,
         'Purity':False,
         'Swift Strike':True,
         'Trip':True,
         'Apotheosis':False,
         'Chrysalis':False,
         'Hand of Greed':True,
         'Magnetism':False,
         'Master of Strategy':False,
         'Mayhem':False,
         'Metamorphosis':False,
         'Panache':False,
         'Sadistic Nature':False,
         'Secret Technique':False,
         'Secret Weapon':False,
         'The Bomb':False,
         'Thinking Ahead':False,
         'Transmutation':False,
         'Violence':False,
         'Apparition':False,
         'Bite':True,
         'Expunger':True,
         'Insight':False,
         'J.A.X.':False,
         'Ritual Dagger':True,
         'Safety':False,
         'Smite':True,
         'Through Violence':True,
         'Slimed':False}

#unplayable cards
BCards = ['Burn',
          'Dazed',
          'Wound',
          'Void',
          "Ascender's Bane",
          'Clumsy',
          'Curse of the Bell',
          'Decay',
          'Doubt',
          'Injury',
          'Necronomicurse',
          'Normality',
          'Pain',
          'Parasite',
          'Pride',
          'Regret',
          'Shame',
          'Writhe']

playable = ICards | GCards
cardNames = list(playable.keys()) + BCards


### READ VIDEO ###

if len(sys.argv) != 2:
    print("invalid number of arguments")
    quit()

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"

cap = cv.VideoCapture('vids/'+sys.argv[1])
if not cap.isOpened():
    print("can't open video: vids/"+sys.argv[1])
    quit()

#was a card being played in the last frame
cardPlayed = False
#the last frame where an action was found
lastFind = 0
#current frame
f = -1

#is version known?
v_known = False
#version 2.0
v2p0 = False

#if there is an existing data file, continue where it left off
path = 'data/'+sys.argv[1]+'.dat'
if os.path.isfile(path) and os.path.getsize(path) > 0:
    dat = open(path, 'r')
    lines = dat.readlines()
    if lines[-1] == "done\n":
        cap.release()
        dat.close()
        quit()
    else:
        a = json.loads(lines[-1])
        lastFind = a['frame']
        f = a['frame']-1
        cap.set(cv.CAP_PROP_POS_FRAMES, a['frame'])

        v_known = True
        v2p0 = a['v2.0']

        print("continuing after frame {}...".format(a['frame']))
    dat.close()

dat = open(path, 'a', buffering=1)

#make a queue of the most recent frames to be able to quickly look back
frames = deque([np.zeros((1080,1920,3))]*6)

def finish():
    global cap
    global dat

    dat.write('done\n')
    print("done")

    cap.release()
    dat.close()
    quit()

#loop until the video is done
while True:

    #get next frame
    f += 1
    ret = cap.grab()
    if not ret:
        finish()

    #only read every other frame
    if f % 2 == 1:
        continue
    ret, frame = cap.retrieve()

    if not ret:
        finish()

    frames.append(frame)
    frames.popleft()

    #skip ahead a little if we just found an action
    if f < lastFind+15:
        continue


    ### FIND ACTIONS ###

    b = blueLine(frame) and energyLine(frames[0])
    #skip if we're still looking at the last action we noted
    if b == cardPlayed:
        continue

    cardPlayed = b
    if not b:
        continue

    lastFind = f

    a = {'frame':f,
        'card':'_',
        'hand':[],
        'floor':-1,
        'hp':-1,
        'maxhp':-1,
        'block':-1,
        'energy':-1,
        'v2.0':False}

    print("frame {}".format(f))


    ### FIND FLOOR ###

    floor = frame[19:49, 986:1026]
    floor = floor[:,:,2] < 200
    floor = Image.fromarray(floor)
    floor = pytesseract.image_to_string(floor, config='--psm 6')
    floor = re.sub('[^0-9]', '', floor)

    if floor == '':
        floor = -1
    else:
        floor = int(floor)

    a['floor'] = floor

    #the fourth act is outside the scope of the project
    if floor > 51:
        finish()


    ### FIND CARD ###

    card = frame[370:370+200, 960-300:960+300]

    #generate binary image showing which pixels
    # have the light blue color of the card border
    blues = np.broadcast_to([218,214,124], card.shape)
    border = card.astype('float') - blues
    border = np.apply_along_axis(sqnorm, 2, border)
    border = border < 6000

    #generate Hough line transformation
    angles = np.linspace(-np.pi/4, np.pi*3/4, 360, endpoint=False)
    border = skimage.img_as_ubyte(border)
    H,_,ds = hough_line(border, angles)

    #extract data for vertical and horizontal lines
    verl = H[:,90]
    horl = H[:,270]

    #find vertical lines
    verl = highpass(30)(verl)
    verl = gaussian_filter(verl, sigma=3)
    verp = peak_local_max(verl, 10)
    verp = np.ravel(verp)
    verp = verp[verp.argsort()]

    #find horizontal lines
    horl = highpass(30)(horl)
    horl = gaussian_filter(horl, sigma=3)
    horp = peak_local_max(horl, 10)
    horp = np.ravel(horp)
    horp = horp[horp.argsort()]

    #skip if we do not have two vertical lines and one horizontal one
    if len(verp) < 2 or len(horp) == 0:
        continue

    #get outline of card
    offset = int(ds[0])
    y1 = min(card.shape[0], horp[0]+offset+15)
    y2 = min(card.shape[0], horp[0]+offset+90)
    x1 = min(card.shape[1], verp[0]+offset+40)
    x2 = max(0, verp[-1]+offset-40)

    crop = card[y1:y2, x1:x2]

    #attempt to identify the card based on the crop of the header
    str1 = readHeader(crop)
    str1 = read2card(str1, list(playable.keys()))
    if str1 == '':
        continue

    a['card'] = str1


    ### MISC. DATA ###

    # VERSION
    if not v_known:
        version = frame[77:103, 1630:1910]
        version = skimage.color.rgb2gray(version)
        version = version < 0.15
        version = Image.fromarray(version)
        version = pytesseract.image_to_string(version, config='--psm 6')
        lb = version.find('[')
        rb = version.find(']')
        if rb-lb == 5 and version[lb:lb+4] == '[V2.':
            v_known = True
            if version[lb+4] == '0' or version[lb+4] == '1':
                v2p0 = True

    a['v2.0'] = v2p0

    frame = frames[0]

    # HP + MAX HP
    hp = frame[18:50, 358:446]
    hp = hp[:,:,2] < 200
    hp = Image.fromarray(hp)
    hp = pytesseract.image_to_string(hp, config='--psm 6')
    hp = re.sub('[^0-9/]', '', hp)
    hp = hp.strip()

    split = hp.find('/')
    if split != -1:
        a['hp'] = int(hp[:split]) if hp[:split] != '' else -1
        maxhp = re.sub('[^0-9]', '', hp[split+1:])
        a['maxhp'] = int(maxhp) if maxhp != '' else -1

    # ENERGY
    energy = frame[860:920, 130:195]
    energy = skimage.color.rgb2gray(energy)
    brightest = energy.max()
    energy = energy/brightest
    energy = energy > 0.8
    energy = compSizeFilter(energy, 50, np.inf, 8)
    energy = compRemRight(energy)
    energy = np.logical_not(energy)
    energy = Image.fromarray(energy)
    energy = pytesseract.image_to_string(energy, config='--psm 6')
    energy = re.sub('[^0-9]', '', energy)
    energy = energy.strip()
    if energy != '':
        a['energy'] = int(energy[0])

    # BLOCK
    block = frame[747:797, 327:377]
    block = skimage.color.rgb2gray(block)
    block = block > 0.9
    block = compSizeFilter(block, 50, np.inf, 4)
    block = np.logical_not(block)
    block = Image.fromarray(block)
    block = pytesseract.image_to_string(block, config='--psm 6')
    block = re.sub('[^0-9]', '', block)
    block = block.strip()
    if block == '':
        block = '0'
    block = int(block)
    a['block'] = block


    ### READ HAND ###

    #skip forward a few frames so the played card is out of the way
    for j in range(3):
        cap.grab()
        _,frame = cap.read()
        frames.append(frame)
        frames.popleft()
    f += 6

    im = frame[800:, 180:1740, :]
    im2 = im[140:,:,:]
    frame = frame[800:1040, 180:1740]

    ### BLUE METHOD ###

    #find pixels with the same color as the card borders
    blues = np.broadcast_to([218,214,124], im2.shape)
    borders = im2.astype('float') - blues
    borders = np.apply_along_axis(sqnorm, 2, borders)
    borders = borders < 6000

    #use a Hough line transform to find the vertical borders of the playable cards
    borders = skimage.img_as_ubyte(borders)
    H,_,ds = hough_line(borders)
    H = highpass(75)(H)
    H = gaussian_filter(H, sigma=3)
    peaks = peak_local_max(H, 50)
    peaks = peaks[peaks[:,0].argsort()]

    hand = []
    foundBlue = []

    #for each mostly vertical light blue line
    for i,da in enumerate(peaks):

        #get properly rotated crop of card
        x = int(ds[da[0]])
        x = max(0, x)

        x2 = min(100, x)
        card = im[:, max(x-100,0):x+228]
        card = rotateImage(card, da[1]-90, (140,x2))

        card = card[:, x2:x2+180]

        if card.shape[1] == 0:
            continue

        #find light blue pixels in the crop
        blues = np.broadcast_to([218,214,124], card.shape)
        borders = card.astype('float') - blues
        borders = np.apply_along_axis(sqnorm, 2, borders)
        borders = borders < 6000
        borders = skimage.img_as_ubyte(borders)

        #use a Hough line transform to find the top of the card
        angles = np.linspace(np.pi/4, np.pi*3/4, 90, endpoint=False)
        H,_,hs = hough_line(borders, angles)
        H = highpass(50)(H)
        H = gaussian_filter(H, sigma=3)
        top = peak_local_max(H, 10, num_peaks=1)

        if len(top) == 0:
            continue

        #get crop of card header
        y = int(hs[top[0,0]])
        crop = card[y+15:y+60, 45:]

        #get name of card
        str1 = readHeader(crop)
        str1 = read2card(str1, cardNames)

        if str1 != '':
            hand.append(str1)
            #note the placement that the card was found
            # so we won't count it again in the general method
            foundBlue.append(x+120)


    ### GENERAL METHOD ###

    #find the correct background for the current frame
    bgi = -1
    if floor <= 16:
        bgi = np.argmin(bgdiff(frame, [bgs[0],bgs[1]], (55,85,1240,1270)))
    elif floor <= 33:
        bgi = 2
        bgi += np.argmin(bgdiff(frame, [bgs[2],bgs[3]], (45,95,120,220)))
    else:
        ooze1 = np.argmin(bgdiff(frame, [bgs[4],bgs[7]], (25,55,1240,1330)))
        ooze2 = np.argmin(bgdiff(frame, [bgs[4],bgs[7]], (50,90,150,270)))
        bgi = 4 + ooze1 + 2 * ooze2

    #find out which pixels are not part of the background
    mask = frame.astype('float') - bgs[bgi]
    mask = np.apply_along_axis(sqnorm, 2, mask)
    mask = mask >= 600

    #remove the energy display and the 'end turn' button
    mask = compSizeFilter(mask, 25000, np.inf, 4)

    #fill out the gaps in the mask
    mask = np.logical_not(mask)
    mask = compSizeFilter(mask, 100000, np.inf, 4)
    mask = np.logical_not(mask)

    #smooth the edges of the mask with binary closing
    mask = binary_closing(mask, np.ones((30,30), np.uint8))

    edges = canny(mask[15:-15], sigma=10)

    #get left and right corners of any size of the hand outline
    leftcs = binary_hit_or_miss(edges, np.array([[0,1],[1,0]]), np.zeros((2,2)))
    rightcs = binary_hit_or_miss(edges, np.array([[1,0],[0,1]]), np.zeros((2,2)))
    cs = np.dstack((leftcs,rightcs,np.zeros_like(leftcs)))

    #remove small corners so we only have the actual corners of the cards
    leftcs = compSizeFilter(leftcs, 5, np.inf, 8)
    rightcs = compSizeFilter(rightcs, 5, np.inf, 8)
    cs = np.dstack((leftcs,rightcs,np.zeros_like(leftcs)))

    #find the centers of the left corners and represent them with 0
    leftcs = compCentroids(leftcs, 8).astype('int')
    zeros = np.zeros((leftcs.shape[0], 1), dtype='int')
    leftcs = np.hstack((leftcs, zeros))

    #find the centers fo the right corners and represent them with 1
    rightcs = compCentroids(rightcs, 8).astype('int')
    ones = np.ones((rightcs.shape[0], 1), dtype='int')
    rightcs = np.hstack((rightcs, ones))

    #check if corner `a` is below any of the corners `bs`
    # if it is, it is likely a false positive
    def under(a, bs):
        for b in bs:
            if abs(a[0]-b[0]) < 120 and a[1] > b[1]:
                return False
        return True

    cs = np.vstack((leftcs, rightcs))
    cs = np.array([c for c in cs if under(c, cs)])
    if cs.shape[0] > 0:
        cs = cs[cs[:,0].argsort()]

        #whether card at `x` was already found by the blue method
        def found(x):
            for c in foundBlue:
                if abs(x-c) < 100:
                    return True
            return False

        corners = []
        #iterate every neighboring pair of corners
        for i in range(1, len(cs)):

            prevRight = cs[i-1,2] == 1
            curRight = cs[i,2] == 1
            x1 = cs[i-1,0]
            x2 = cs[i,0]

            #choose a corner to use as point of reference for each card
            if not prevRight and not curRight and not found(x1+120):
                #two left corners
                corners.append(cs[i-1])
            elif not prevRight and curRight:
                #left and right corner
                if not found(x1+120):
                    corners.append(cs[i-1])
                if x2 - x1 > 300 and not found(x2-120):
                    corners.append(cs[i])
            elif prevRight and not curRight:
                #right and left corner
                continue
            elif prevRight and curRight and not found(x2-120):
                #two right corners
                corners.append(cs[i])

        frame = frame[15:]
        for i,c in enumerate(corners):

            #get a rough crop of the top of the card (excluding corners)
            y0 = max(0, c[1]-30)
            y1 = min(edges.shape[0], c[1]+30)

            top = 0
            if c[2] == 0:
                top = edges[y0:y1, c[0]+50:c[0]+120]
            else:
                top = edges[y0:y1, c[0]-120:c[0]-50]

            #find the rotation of the card
            theta = 0
            if np.sum(top) > 0:
                angles = np.linspace(0, np.pi, 180, endpoint=False)
                H,angs,_ = hough_line(top, angles)

                peak = np.unravel_index(np.argmax(H), H.shape)
                theta = 90-peak[1]

            #get crop of card header
            crop = []
            if c[2] == 0:
                crop = frame[:, c[0]:c[0]+230]
                crop = rotateImage(crop, -theta, (c[1],1))
            else:
                crop = frame[:, c[0]-230:c[0]]
                crop = rotateImage(crop, -theta, (c[1],229))

            crop = crop[c[1]:, 30:-30]
            crop = crop[:60]

            #find the name of the card
            str1 = readHeader(crop)
            str1 = read2card(str1, cardNames)

            if str1 != '':
                hand.append(str1)

    a['hand'] = hand

    ### WRITE DATA ###

    dat.write(json.dumps(a))
    dat.write('\n')

