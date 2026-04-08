import os
import datetime
import pathlib

class CurrentRunFolder:
    def __init__(self, run_folder: str):
        self.run_folder = run_folder
        self._create_run_folders()
    
    def _create_run_folders(self):
        os.makedirs(self.run_folder, exist_ok=True)
    
    def get_file_name(self, filename:str, subfolder:str|None=None):
        if(subfolder is not None):
            os.makedirs(str(pathlib.Path(self.run_folder, subfolder)), exist_ok=True)
            return str(pathlib.Path(self.run_folder, subfolder, filename))
        return str(pathlib.Path(self.run_folder, filename))
    
    def get_date_file_name(self, extension:str, subfolder:str|None=None):
        filename = datetime.now().strftime("%Y-%m-%d|%H:%M:%S") + "." + extension
        return self.get_file_name(filename, subfolder)

if __name__ == "__main__":
    a = CurrentRunFolder("runs/a")

    print(a.get_file_name("b.txt", "c"))