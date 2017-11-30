import winsound




class HomerException(Exception):
    def __init__(self, message):
        super(HomerException, self).__init__(message)

        winsound.PlaySound("doh1.wav",winsound.SND_FILENAME)





class StarWarsException(Exception):
    def __init__(self, message):
        super(StarWarsException, self).__init__(message)

        winsound.PlaySound("swvader03.wav",winsound.SND_FILENAME)


class LightSaberException(Exception):
    def __init__(self, message):
        super(LightSaberException, self).__init__(message)

        winsound.PlaySound("light-saber-on.wav",winsound.SND_FILENAME)












if __name__ == "__main__":
    raise StarWarsException("Aloha")
    
