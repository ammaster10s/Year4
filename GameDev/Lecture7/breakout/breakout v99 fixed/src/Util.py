import pygame
import json
import os

class Sprite:
    def __init__(self, image):
        self.image = image


class SpriteManager:
    def __init__(self):
        # Get the directory where this file (Util.py) is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the project root (breakout v99 fixed/)
        project_root = os.path.dirname(current_dir)
        
        self.spriteCollection = self.loadSprites(
            [
                os.path.join(project_root, "sprite/MiddlePaddle.json"),
                os.path.join(project_root, "sprite/SmallPaddle.json"),
                os.path.join(project_root, "sprite/Brick.json"),
                os.path.join(project_root, "sprite/Ball.json"),
                os.path.join(project_root, "sprite/Heart.json"),
                os.path.join(project_root, "sprite/Arrow.json"),
            ]
        )

    def loadSprites(self, urlList):
        resDict = {} #result dictionary
        for url in urlList:
            with open(url) as jsonData:
                data = json.load(jsonData)
                # Get the directory of the JSON file to resolve relative paths
                json_dir = os.path.dirname(url)
                # Remove leading ./ from the sprite sheet path if present
                sprite_sheet_url = data["spriteSheetURL"].lstrip("./")
                # Go up to project root (json_dir is in sprite/, we need to go up one level)
                project_root = os.path.dirname(json_dir)
                sprite_sheet_path = os.path.join(project_root, sprite_sheet_url)
                mySpritesheet = SpriteSheet(sprite_sheet_path)
                dic = {}
                for sprite in data["sprites"]:
                    try:
                        colorkey = sprite["colorKey"]
                    except KeyError:
                        colorkey = None
                    try:
                        xSize = sprite['xsize']
                        ySize = sprite['ysize']
                    except KeyError:
                        xSize, ySize = data['size']
                    dic[sprite["name"]] = Sprite(
                        mySpritesheet.image_at(
                            sprite["x"],
                            sprite["y"],
                            sprite["scalefactor"],
                            colorkey,
                            xTileSize=xSize,
                            yTileSize=ySize,
                        )
                    )
                resDict.update(dic)
                continue
        return resDict

class SpriteSheet(object):
    def __init__(self, filename):
        try:
            self.sheet = pygame.image.load(filename)
            self.sheet = pygame.image.load(filename)
            if not self.sheet.get_alpha():
                self.sheet.set_colorkey((0, 0, 0))
        except pygame.error:
            print("Unable to load spritesheet image:", filename)
            raise SystemExit

    def image_at(self, x, y, scalingfactor, colorkey=None,
                 xTileSize=16, yTileSize=16):
        rect = pygame.Rect((x, y, xTileSize, yTileSize))
        image = pygame.Surface(rect.size)
        image.blit(self.sheet, (0, 0), rect)
        if colorkey is not None:
            if colorkey == -1:
                colorkey = image.get_at((0, 0))
            image.set_colorkey(colorkey, pygame.RLEACCEL)
        return pygame.transform.scale(
            image, (xTileSize * scalingfactor, yTileSize * scalingfactor)
        )