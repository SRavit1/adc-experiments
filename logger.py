class Logger:
    def __init__(self, file):
        self.file = file
    
    def log(self, s):
        print(s)
        #with open(self.file, 'a') as f:
        #    f.write(s + "\n")