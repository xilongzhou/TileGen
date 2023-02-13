import os 
import pickle



class DataSourcer():
    def __init__(self, paths_list):
        """
        This class read and prepare all raw img, sem, ins and parsing, into four different lists.
        paths_list shold be a list containing multiple dict specifying paths to img, sem, ins and parsing
        """

        composition_exist = 'composition' in paths_list[0]

        self.img_bank = []
        self.sem_bank = []
        self.ins_bank = []

        if composition_exist:
            self.composition_bank = []

                
        for path in paths_list:
            self.img_bank += self.read_image_folder(path['img']) 
            # self.sem_bank += self.read_image_folder(path['sem']) 
            # self.ins_bank += self.read_image_folder(path['ins']) 
            if composition_exist:
                self.composition_bank += self.read_image_folder(path['composition']) 
            
        # assert len(self.img_bank) == len(self.sem_bank) == len(self.ins_bank)   
        if composition_exist:
            assert len(self.img_bank) == len(self.composition_bank)

        print('total data: ', len(self.img_bank) )

        
        
    def read_image_folder(self, folder_path):
        output = os.listdir(  folder_path  )
        output.sort()
        output = [  os.path.join(folder_path, item) for item in output  ]
        return output
    



