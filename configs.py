def get_configs(attribute):
  configurations = {
                    "DIMS" : 256,
                    "CHANELS" : 1,
                    "EPOCHS" : 1,
                    "BASE" : "Fonts/English_Fonts/",
                    "X" : "Screen_Sans_Normal",
                    "layer_depths" : [4, 8, 16, 32, 32, 16, 8, 4],
                    "layer_pools" : [2, 2, 2, 2, 2, 2, 2, 2],
                    "layer_convs" : [3, 3, 3, 3, 3, 3, 3, 3]
                    }
  return configurations[attribute]