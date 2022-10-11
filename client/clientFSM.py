from enum import Enum, auto

class ClientStateMachine:

    class States(Enum):
        InitWait = auto()
        DataGenWait = auto()
        DataReady = auto()
        ModelTraining = auto()

    def __init__(self):
        self.state = self.States.InitWait
    
    def onEvent(self, event):
        if self.state == self.States.InitWait:
            if event == "connected":
                self.state = self.States.DataGenWait
        elif self.state == self.States.DataGenWait:
            if event == "data generated":
                self.state = self.States.DataReady
        elif self.state == self.States.DataReady:
            if event == "received model":
                self.state = self.States.ModelTraining
        elif self.state == self.States.ModelTraining:
            if event == "returned model":
                self.state = self.States.DataGenWait
    
    def getState(self):
        return self.state
  