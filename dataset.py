
class IntentDataset:

    def __init__(self, x, y):
        self.x_data = x
        self.y_data = y

    def __len__(self):
        return len(self.x_data)

    def __get_item__(self, index):
        return self.x_data[index], self.y_data[index]
