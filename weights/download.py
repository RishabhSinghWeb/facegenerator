import os
os.system('cmd /c "pip install gdown"')

import gdown
    
url = "https://drive.google.com/uc?id=1afMN3e_6UuTTPDL63WHaA0Fb9EQrZceE'
gdown.download(url, './ffhq.npy',quiet=False) 